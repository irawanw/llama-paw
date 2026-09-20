from __future__ import annotations

import json
import logging
import os
from typing import Callable, Iterable, cast

import torch
from torch import Tensor

import gguf
import numpy as np

from .base import ModelBase
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin
from .qwen3vl import Qwen3VLVisionModel

logger = logging.getLogger(__name__)

# PAW overlay: substitute the routed experts and PLE table with packed tensors.
PAW_X3_PACKED_ENV = "PAW_X3_PACKED"
PAW_PLE_Q8_ENV = "PAW_PLE_Q8"
PAW_X3_PROJ = ("gate", "up", "down")
PAW_X3_TENSOR = {
    "gate": gguf.MODEL_TENSOR.FFN_GATE_EXP,
    "up": gguf.MODEL_TENSOR.FFN_UP_EXP,
    "down": gguf.MODEL_TENSOR.FFN_DOWN_EXP,
}


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.

    The checkpoint also carries a NextN/MTP draft head under `mtp.*`, exported as a
    trailing block; pass --no-nextn to leave it out.
    """

    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # only the shard names, so the table itself is never held
        self._ple_shards: dict[int, str] = {}
        self._ple_row_dim: int | None = None

        self._paw_x3_root = os.environ.get(PAW_X3_PACKED_ENV)
        self._paw_ple_q8_root = os.environ.get(PAW_PLE_Q8_ENV)
        for env, root in ((PAW_X3_PACKED_ENV, self._paw_x3_root),
                          (PAW_PLE_Q8_ENV, self._paw_ple_q8_root)):
            if root and not os.path.isdir(root):
                raise ValueError(f"{env}={root!r} is not a directory")
        if self._paw_x3_root:
            logger.info("PAW X3: routed experts come from %s", self._paw_x3_root)
        if self._paw_ple_q8_root:
            logger.info("PAW PLE Q8: n-gram table comes from %s", self._paw_ple_q8_root)

    _MTP_MIXER_PREFIX = "mtp.hyper_connection_mixer."

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if name.startswith("model." + cls._MTP_MIXER_PREFIX):
            name = name.replace("model.", "", 1)
        if name.startswith(cls._MTP_MIXER_PREFIX):
            if cls.no_mtp:
                return None
            assert cls._original_block_count is not None
            return f"model.layers.{cls._original_block_count}.{name[len('mtp.'):]}", gen
        return super().filter_tensors((name, gen))

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors = super().index_tensors(remote_hf_model_id=remote_hf_model_id)

        emb = tensors.pop("mtp.fc_embedding.weight", None)
        hid = tensors.pop("mtp.fc_hidden.weight", None)
        if emb is None and hid is None:
            return tensors
        if emb is None or hid is None:
            raise ValueError(
                "the qwen4exp MTP combiner needs both mtp.fc_embedding.weight and "
                "mtp.fc_hidden.weight; pass --no-nextn to convert without the draft head"
            )

        assert self._original_block_count is not None
        name = f"model.layers.{self._original_block_count}.eh_proj.weight"
        tensors[name] = lambda: torch.cat([emb(), hid()], dim=1)
        return tensors

    def _read_hash_constants(self, suffix: str) -> list[int]:
        """Read an int64 PLE constant straight from the checkpoint.

        prepare_tensors() casts every non-float dtype to float32 before
        modify_tensors() sees it (base.py), which would silently round these
        45-bit multipliers. Reading the lazy tensor here bypasses that.
        """
        for name, gen in self.model_tensors.items():
            if name.endswith(suffix):
                t = gen()
                if t.dtype != torch.int64:
                    t = t.to(torch.int64)
                return [int(x) for x in t.tolist()]
        raise ValueError(f"PLE constant {suffix!r} missing from the checkpoint")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_hyper_connection_count(hp["hc_count"])
        self.gguf_writer.add_hyper_connection_low_rank(hp["hc_lowrank"])

        n_layer = hp["num_hidden_layers"]
        self.gguf_writer.add_indexer_head_count(hp["indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["indexer_budget"])
        ratio = hp["indexer_compress_ratio"]
        layer_types = hp["layer_types"]
        ratios = [ratio if layer_types[i] == "full_attention" else 0 for i in range(n_layer)]
        ratios += [0] * (self.block_count - n_layer)
        self.gguf_writer.add_attention_compress_ratios(ratios)

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones
        ple_layers = [i - 1 for i in hp["ple_layer_ids"]]
        if not ple_layers or self.mtp_only:
            return
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(hp["ngram_size"])
        self.gguf_writer.add_ple_heads_per_ngram(hp["heads_per_ngram"])
        self.gguf_writer.add_ple_conv_kernel(hp["ple_conv_kernel_size"])
        self.gguf_writer.add_ple_eos_token_id(self._eos_token_id())
        # an image is decoded as an embeddings-only batch, so the graph has no placeholder
        # ids to hash; carry the id and let it stand in for those positions
        _img = self._image_token_id()
        if _img is not None:
            self.gguf_writer.add_ple_image_token_id(int(_img))
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        self.gguf_writer.add_ple_layer_multipliers(
            self._read_hash_constants("ple_embedding.layer_multipliers"))
        self.gguf_writer.add_ple_head_offsets(
            self._read_hash_constants("ple_embedding.ngram_heads_offsets"))
        self.gguf_writer.add_ple_head_vocab_sizes(
            self._read_hash_constants("ple_embedding.ngram_heads_vocab_sizes"))

    def _image_token_id(self) -> int | None:
        img = self.hparams.get("image_token_id")
        return None if img is None else int(img)

    def _eos_token_id(self) -> int:
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            # the PLE hash resets n-grams on the primary EOS
            return int(eos[-1])
        if eos is None:
            raise ValueError("eos_token_id is required: the PLE hash resets its n-grams on it")
        return int(eos)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if self._paw_x3_root and ".mlp.experts." in name:
            return []

        if self._paw_ple_q8_root and ".ngram_embedding.shard_" in name:
            return []

        # int64 hash constants must stay exact; 1-D tensors force F32, so use KV
        if name.endswith("ple_embedding.layer_multipliers"):
            self._ple_multipliers = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_offsets"):
            self._ple_head_offsets = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_vocab_sizes"):
            self._ple_head_vocab_sizes = [int(x) for x in data_torch.tolist()]
            return []

        if ".ngram_embedding.shard_" in name:
            return self._place_ple_shard(data_torch, name)

        # one projection feeds indexer q and k; split it, as minimax-m3 does
        if ".indexer.index_qk_proj.weight" in name:
            n_q = self.hparams["indexer_n_heads"] * self.hparams["indexer_head_dim"]
            q = data_torch[:n_q]
            k = data_torch[n_q:]
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ".weight"), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ".weight"), k),
            ]

        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]

        if name.endswith(".ple.conv1d.weight"):
            return [(self.map_tensor_name(name), data_torch.squeeze())]

        return super().modify_tensors(data_torch, name, bid)

    # the shards concatenate into a tensor of well over 100 GB
    # use LazyChunkedTensor here, a single shard resident at a time
    def _place_ple_shard(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:

        idx = int(name.rpartition(".shard_")[2].partition(".")[0])
        n_parts = self.hparams["split_ngram_parts"]

        self._ple_shards[idx] = name
        self._ple_row_dim = int(data_torch.shape[-1])

        if len(self._ple_shards) < n_parts:
            return []

        # the checkpoint may yield the shards in any order, the row order is by index
        shards = [self._ple_shards[i] for i in sorted(self._ple_shards)]
        rows = 0
        for shard in shards:
            shape = self.model_tensors[shard]().shape
            if int(shape[-1]) != self._ple_row_dim:
                raise ValueError(
                    f"PLE shard {shard} has row dim {int(shape[-1])}, expected {self._ple_row_dim}")
            rows += int(shape[0])

        table = gguf.LazyChunkedTensor(
            [self._load_ple_shard(shard) for shard in shards],
            shape=(rows, self._ple_row_dim),
            dtype=np.float32,
        )
        gguf_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD]
        return [(gguf_name + ".weight", cast(Tensor, table))]

    def _load_ple_shard(self, name: str):
        def load() -> np.ndarray:
            from .base import LazyTorchTensor

            # a fresh lazy tensor every call, or to_eager() memoizes every shard
            eager = LazyTorchTensor.to_eager(self.model_tensors[name]())
            return eager.to(torch.float32).contiguous().numpy()
        return load

    def prepare_tensors(self):
        super().prepare_tensors()
        n_parts = self.hparams.get("split_ngram_parts", 0)
        if self._ple_shards and len(self._ple_shards) != n_parts:
            raise ValueError(
                f"got {len(self._ple_shards)} PLE embedding shards, expected {n_parts}"
            )
        if self._paw_x3_root:
            self._add_paw_x3_tensors()
        if self._paw_ple_q8_root:
            self._add_paw_ple_q8_tensors()

    @staticmethod
    def _memmap(path: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.memmap:
        expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
        actual = os.path.getsize(path)
        if actual != expected:
            raise ValueError(f"{path} has {actual} bytes, expected {expected}")
        return np.memmap(path, dtype=dtype, mode="r", shape=shape)

    def _add_paw_x3_tensors(self) -> None:
        assert self._paw_x3_root is not None
        n_layer = self.hparams["num_hidden_layers"]
        n_expert = self.hparams["num_experts"]

        for il in range(n_layer):
            layer_dir = os.path.join(self._paw_x3_root, f"L{il:02d}")
            index_path = os.path.join(layer_dir, "index.json")
            if not os.path.isfile(os.path.join(layer_dir, "DONE")):
                raise ValueError(f"PAW X3 layer {il} is incomplete: {layer_dir}")
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            if index["n_expert"] != n_expert:
                raise ValueError(
                    f"PAW X3 layer {il} has {index['n_expert']} experts, expected {n_expert}")

            for proj in PAW_X3_PROJ:
                base = self.format_tensor_name(PAW_X3_TENSOR[proj], il, suffix="")
                shapes = index["proj"][proj]["shapes"]
                arrays = {
                    "m3_trellis": self._memmap(
                        os.path.join(layer_dir, f"{proj}_trellis.bin"), np.dtype("<i2"),
                        (shapes["trellis"][0],)),
                    "m3_meta": self._memmap(
                        os.path.join(layer_dir, f"{proj}_meta.bin"), np.dtype("<i4"),
                        (shapes["meta"][1], shapes["meta"][0])),
                    "m3_suh": self._memmap(
                        os.path.join(layer_dir, f"{proj}_suh.bin"), np.dtype("<f2"),
                        (shapes["suh"][1], shapes["suh"][0])),
                    "m3_svh": self._memmap(
                        os.path.join(layer_dir, f"{proj}_svh.bin"), np.dtype("<f2"),
                        (shapes["svh"][1], shapes["svh"][0])),
                }
                for suffix, array in arrays.items():
                    self.gguf_writer.add_tensor(f"{base}.{suffix}", array)

        logger.info("PAW X3: added %d packed expert layers", n_layer)

    def _add_paw_ple_q8_tensors(self) -> None:
        assert self._paw_ple_q8_root is not None
        layout_path = os.path.join(self._paw_ple_q8_root, "layout.json")
        with open(layout_path, "r", encoding="utf-8") as f:
            layout = json.load(f)

        n_rows = int(layout["n_rows"])
        row_dim = int(layout["dim"])
        if len(layout["shards"]) != self.hparams["split_ngram_parts"]:
            raise ValueError(
                f"PAW PLE Q8 has {len(layout['shards'])} shards, expected "
                f"{self.hparams['split_ngram_parts']}")

        q8 = self._memmap(
            os.path.join(self._paw_ple_q8_root, "ngram_q8.bin"), np.dtype("i1"),
            (n_rows, row_dim))
        scale = self._memmap(
            os.path.join(self._paw_ple_q8_root, "ngram_scale.bin"), np.dtype("<f2"),
            (n_rows, 1))
        base = self.format_tensor_name(gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD, suffix="")
        self.gguf_writer.add_tensor(base + ".q8", q8)
        self.gguf_writer.add_tensor(base + ".scale", scale)
        self._ple_row_dim = row_dim
        logger.info("PAW PLE Q8: added %d rows x %d", n_rows, row_dim)


@ModelBase.register("Qwen4ExpForConditionalGeneration")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpVisionModel(Qwen3VLVisionModel):
    """The vision tower is an unmodified Qwen3-VL ViT."""
