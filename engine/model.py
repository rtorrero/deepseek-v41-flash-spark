"""
model.py -- DeepSeek-V4.1-Flash text model for one-box serving: chunked prefill, multi-token
decode blocks (DSpark verification) and cache rollback, batch size 1.

Ported from the reference `inference/model.py`; the tilelang kernels are replaced by torch ops
and the routed experts by `engine.experts.ExpertStore` (+ the Triton FP4 grouped-MoE kernel in
`tools/fp4_moe.py`). Deviations from the reference, all towards MORE precision:
  * activations are not fake-quantized to fp8 (optional flag),
  * window KV and compressed KV caches are kept in bf16 instead of fp8 / FP4-E4M3
    (DSV41_PREFILL_KV_FP8=1 gives that back for the prefill GATHER only -- the caches themselves
    stay bf16, so decode reads what it reads today),
  * the indexer's Q/K are not FP4-quantized.
Position semantics are the reference's: a chunk of T tokens at absolute start position S.
Any (S, T) with T <= 512 works, which is what chunked prefill and 6-token verify blocks need.
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import v41_ref as R  # noqa: E402

from engine import prefill_attn_gemm as AG  # noqa: E402  (torch-free; fp32 unless the env asks)
from engine import prefill_topk as PT  # noqa: E402  (torch-free; off unless the env asks)
from engine import prefill_fp8 as PF  # noqa: E402  (torch-free; off unless the env asks)
from engine import prefill_sinkhorn as PS  # noqa: E402  (torch-free; off unless the env asks)

# Longest prefill chunk. Bigger chunks are strictly cheaper on this recipe: a prefill chunk streams
# nearly every expert of every layer through the transient ring whatever its length (a 512-token
# chunk already touches ~370 of 384), so the NVMe traffic of a prompt is ~chunks x layers x 384
# experts and quadrupling the chunk quarters it. The same holds in pruned all-resident mode, where
# nothing is read from NVMe at all: the grouped MoE still unpacks every resident expert of a layer
# once per chunk, so halving the number of chunks halves the unpack passes.
# The ceiling is activation memory: at T=2048 the gathered window+compressed KV of one layer is
# ~2.7 GB in bf16 (see docs/memory-budget.md, "Where a prefill chunk's memory goes"), which is why
# 4096 wants DSV41_PREFILL_KV_FP8=1 alongside it.
# Keep it a multiple of 128: MM_TILE/ATTN_TILE pad the last tile of every GEMM, and a chunk that
# overruns a tile boundary pays ~30 % more iteration time for the rows that only pad.
MAX_CHUNK = int(os.environ.get("DSV41_PREFILL_CHUNK", 2048))
# Window ring slots. Must exceed window_size + the longest chunk a single forward sees, because
# `attention` gathers a query's window out of the ring AFTER writing the whole chunk into it
# (128 + 2048 here). 4096 slots x 512 dims x bf16 x (40 layers + 3 MTP) = 180 MB.
# The default follows MAX_CHUNK, or raising DSV41_PREFILL_CHUNK alone would wrap the ring inside a
# single chunk and silently give the first queries of it whatever the last ones wrote.
RING = int(os.environ.get("DSV41_RING", max(4096, MAX_CHUNK + 512)))

# Prefill-only fp8 working copies of the gathered window / compressed KV (DSV41_PREFILL_KV_FP8=1;
# default 0, and with it off not one byte of the path below changes).
#
# What is quantised is ONLY the per-chunk gather that `attention` builds for the softmax. The ring
# (`Caches.win`) and the compressed/index caches (`Caches.ckv`, `Caches.ik`) stay bf16 exactly as
# they are, so everything decode later reads is in the format it is in today.
#
# e4m3 and not e5m2, for two reasons. (1) Precision: 3 mantissa bits against 2, so the worst-case
# relative rounding error is 2^-4 = 6.25 % instead of 2^-3 = 12.5 %, and error here lands in an
# attention score. (2) The extra range e5m2 buys is range these tensors do not use: both are the
# output of an rmsnorm (plus RoPE on the last 64 dims), i.e. O(1), and e4m3's +-448 is nine binades
# above that. It is also what the reference implementation uses for these very tensors -- this
# engine's module docstring lists "window KV and compressed KV caches are kept in bf16 instead of
# fp8 / FP4-E4M3" as a deviation TOWARDS more precision, and this flag gives that deviation back
# for the prefill working copy alone.
PREFILL_KV_FP8 = os.environ.get("DSV41_PREFILL_KV_FP8", "0") == "1"
# Looked up rather than named, so that a torch without the dtype still imports this module and only
# refuses the flag -- the default path does not need it.
FP8_KV_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_KV_MAX = 448.0   # largest finite e4m3; the cast NaNs above it, so clamp first
if PREFILL_KV_FP8 and FP8_KV_DTYPE is None:
    raise RuntimeError(f"DSV41_PREFILL_KV_FP8=1 needs torch.float8_e4m3fn; torch is {torch.__version__}")
# Rows gathered per pass when filling the fp8 buffer. The gather is an index_select, not a GEMM, so
# no invariance rule applies to the tile size; it only bounds the bf16 scratch, which is
# tile x (window_size + index_topk) x head_dim x 2 = 84 MB at 128.
KV_GATHER_TILE = 128

# Chunk invariance requires every GEMM to give the same row whatever the batch length M. cuBLAS
# picks split-K kernels for small M and, with this flag on, reduces the K-splits in bf16, so
# F.linear(x[:6], w) != F.linear(x, w)[:6] by ~2.4e-3 for the N=512 wkv projection -- which the
# attention softmax then amplifies ~2x per layer. fp32 reduction cuts that to ~9e-5.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# ... and the same GEMM must be issued with the same M whatever the chunk length, or cuBLAS
# switches tiling/split-K and a row comes out a few ulps different. See v41_ref.mm().
# 16 and not something larger: for several of these shapes (wq_a, the expert w1/w3) cuBLAS gives a
# row a slightly different result depending on its OFFSET inside the tile, and a token's offset is
# chunk-relative. 8 and 16 are offset-invariant for every shape the model uses; 32/64/128 are not.
MM_TILE = 16
# The attention softmax is a *batched* GEMM (one independent problem per token), which is already
# offset-invariant, so it can use a bigger tile.
ATTN_TILE = 64
KEY_BLOCK = 512  # indexer score tile along the compressed-key axis (= index_topk)
R.MM_TILE = MM_TILE

# The two prefill switches of docs/gemm-dispatch.md, pushed into v41_ref the same way MM_TILE is.
# Both default to off and off is byte-identical: PREFILL_FP8_MODE "off" is FP8Weight.dequant(), and
# with HC_SINKHORN_FUSED never asked for, hc_mixes keeps the tiled torch Sinkhorn.
R.PREFILL_FP8_MODE = PF.MODE
try:  # the fused Sinkhorn decode has used all along; absent only where triton is not importable
    from engine.hc_sinkhorn import hc_split_sinkhorn as _hc_fused  # noqa: E402
    R.HC_SINKHORN_FUSED = _hc_fused
except Exception:  # noqa: BLE001
    _hc_fused = None


# ----------------------------------------------------------------------------- weights
class IndexerWeights:
    def __init__(self, get, p: str, owns_k: bool, device: str):
        w, sc = get(p + "indexer.wq_b.weight").to(device), get(p + "indexer.wq_b.scale").to(device)
        self.wq_b = R.FP8Weight(w, sc) if (R.FP8Weight is not None and os.environ.get("DSV41_DENSE_FP8", "1") == "1") else R.dequant_fp8_block(w, sc)
        self.weights_proj = get(p + "indexer.weights_proj.weight").to(device).to(torch.bfloat16)
        self.owns_k = owns_k
        if owns_k:
            self.wk = get(p + "indexer.wk.weight").to(device).to(torch.bfloat16)
            self.k_norm = get(p + "indexer.k_norm.weight").to(device).to(torch.bfloat16)


class Weights:
    """All non-routed-expert weights on the GPU, bf16 (fp32 where the reference uses fp32)."""

    def __init__(self, model_dir: str, index: dict, args: R.Args, device: str, log=print, act_quant: bool = False,
                 n_layers: int | None = None, load_mtp: bool = True, engram_dir: str | None = None):
        self.args, self.device = args, device
        n_load = args.n_layers if n_layers is None else n_layers
        wm = index["weight_map"]
        handles = {}

        def get(name):
            f = wm[name]
            if f not in handles:
                path = os.path.join(model_dir, f)
                if not os.path.exists(path) and ".engram." in name:
                    # engram shard not (yet) on disk: the small non-table tensors fetched by tools/engram_rows.py
                    L = name.split(".")[1]
                    path = os.path.join(engram_dir or "engram_rows", f"layer{L}_weights.safetensors")
                handles[f] = safe_open(path, "pt", device="cpu")
            return handles[f].get_tensor(name)

        t0 = time.time()
        self.embed = get("embed.weight").to(device).to(torch.bfloat16)
        # bf16 (the stored dtype) unless DSV41_HEAD_FP32=1. The reference keeps the LM head in fp32
        # ("so the logits come out in fp32 directly"); the fast decode path already ran a bf16 copy
        # (fp32 accumulate, logits rounded to bf16), so with a bf16 head here the two paths use the
        # same weights and the 2.65 GB fp32 copy disappears (= ~140 more expert slots).
        # ... and, with DSV41_HEAD_FMT, in fp8 or fp4 instead: the head is read in full on every
        # decode step, so its stored format is worth as much as a dense projection group's.
        self.head = R.make_head(get("head.weight").to(device))
        self.norm = get("norm.weight").to(device).to(torch.bfloat16)
        self.layers = []
        self.indexers = {}
        self.engram = {}
        for L in range(n_load):
            self.layers.append(R.LayerWeights(get, L, args, device))
            if L in args.index_source_layers:
                self.indexers[L] = IndexerWeights(get, f"layers.{L}.attn.", L in args.kv_source_layers, device)
            if L in args.engram_layer_ids:
                self.engram[L] = R.EngramWeights(get, L, device)
            if L % 10 == 9:
                log(f"weights: layer {L} loaded ({time.time() - t0:.0f}s)")
            for f in list(handles):
                if f.endswith(f"{L + 3:05d}-of-00048.safetensors"):
                    del handles[f]
        # DSpark blocks
        self.mtp = []
        for k in range(3 if load_mtp else 0):
            self.mtp.append(MTPWeights(get, k, args, device))
        # The three blocks are the only consumers of the override, and it is up to 1.3 GB of host
        # tensors. Holding it past this point would come straight off the arena, which is sized from
        # what is free after the weights are loaded.
        _drop_mtp_override()
        self.dspark_experts = None  # filled by the engine (arena of 3 x 128 experts)
        log(f"weights: all non-expert weights on GPU in {time.time() - t0:.0f}s")


# A fine-tuned DSpark head instead of the checkpoint's (tools/train_mtp.py, docs/architecture.md
# "Fine-tuning the drafter"). The file holds bf16 tensors under the checkpoint's own names
# (`mtp.0.attn.wq_a.weight`, ...); everything it does NOT name comes from the checkpoint, so a file
# with three tensors in it is a legal three-tensor override. Unset = the shipped head.
#
# The tensors are re-quantized on load into the format the shipped path serves in -- fp8 e4m3 with
# UE8M0 block scales, and then FP4 for whichever groups DSV41_DENSE_FP4 names -- so a fine-tuned
# head reads exactly as many bytes per draft as the shipped one and the speed of the draft graph
# does not move. That round trip is lossy, which is why tools/train_mtp.py reports the acceptance
# proxy after quantizing as well as before.
MTP_WEIGHTS = os.environ.get("DSV41_MTP_WEIGHTS", "").strip()
_MTP_OVERRIDE = None


def _mtp_override():
    """{name: tensor} from DSV41_MTP_WEIGHTS, read once. Keys outside `mtp.` are refused rather than
    ignored: a file whose names are wrong would otherwise load as "the shipped head" and the only
    evidence would be an acceptance number that did not move."""
    global _MTP_OVERRIDE
    if _MTP_OVERRIDE is None:
        if not MTP_WEIGHTS:
            _MTP_OVERRIDE = {}
        else:
            f = safe_open(MTP_WEIGHTS, "pt", device="cpu")
            keys = list(f.keys())
            bad = [k for k in keys if not k.startswith("mtp.")]
            if bad:
                raise ValueError(f"DSV41_MTP_WEIGHTS={MTP_WEIGHTS}: {len(bad)} tensor(s) outside the "
                                 f"DSpark blocks, e.g. {bad[:3]}")
            _MTP_OVERRIDE = {k: f.get_tensor(k) for k in keys}
            print(f"DSV41_MTP_WEIGHTS: {len(keys)} tensors from {MTP_WEIGHTS}", flush=True)
    return _MTP_OVERRIDE


def _drop_mtp_override():
    """Release the host tensors once the DSpark blocks have been built. Anything that built an
    MTPWeights after this would silently get the shipped head, so nothing does: `Weights.__init__`
    is the only place they are constructed."""
    global _MTP_OVERRIDE
    _MTP_OVERRIDE = {}


class MTPWeights:
    """One DSpark block (`mtp.k.*`): a Block with a 128-expert MoE, plus stage-specific heads."""

    def __init__(self, get, k: int, args: R.Args, device: str):
        p = f"mtp.{k}."
        self.k = k

        class _A(R.Args):
            pass

        a = R.Args(**{f: getattr(args, f) for f in R.Args.__dataclass_fields__})
        a.n_routed_experts = 128
        a.n_activated_experts = 3
        self.args = a
        # LayerWeights expects "layers.{L}." prefix; build the same fields by hand
        dev = device

        over = _mtp_override()

        def ov(name):
            """The fine-tuned tensor for `mtp.k.<name>`, or None."""
            t = over.get(p + name)
            return None if t is None else t.to(dev)

        def bf(name):
            t = ov(name)
            return (get(p + name) if t is None else t).to(dev).to(torch.bfloat16)

        def f32(name):
            t = ov(name)
            return (get(p + name) if t is None else t).to(dev).to(torch.float32)

        fp4_groups = R.dense_fp4_groups()

        def fp8lin(name):
            t = ov(name + ".weight")
            if t is not None:
                # bf16 in, the checkpoint's own stored format out -- same kernels, same bytes/step
                if R.quantize_to_fp8 is None:
                    return t.to(torch.bfloat16)
                return R.maybe_fp4(R.quantize_to_fp8(t.to(torch.bfloat16)), name, fp4_groups)
            w, sc = get(p + name + ".weight").to(dev), get(p + name + ".scale").to(dev)
            if R.FP8Weight is not None and os.environ.get("DSV41_DENSE_FP8", "1") == "1":
                return R.maybe_fp4(R.FP8Weight(w, sc), name, fp4_groups)
            return R.dequant_fp8_block(w, sc)

        def wo_a():
            t = ov("attn.wo_a.weight")
            if t is None:
                return R.make_wo_a(get(p + "attn.wo_a.weight").to(dev),
                                   get(p + "attn.wo_a.scale").to(dev), args, fp4_groups)
            # [G, R, K] bf16 -> the grouped stored format, addressed as [G*R, K]
            if R.FP8GroupedWeight is None or R.quantize_to_fp8 is None:
                return t.to(torch.bfloat16).view(args.o_groups, args.o_lora_rank, -1)
            q = R.quantize_to_fp8(t.to(torch.bfloat16).reshape(-1, t.shape[-1]))
            g = R.FP8GroupedWeight(q.w, q.s, args.o_groups, args.o_lora_rank)
            return R.quantize_fp8_grouped_to_fp4(g) if "wo_a" in fp4_groups else g

        self.attn_norm = bf("attn_norm.weight"); self.ffn_norm = bf("ffn_norm.weight")
        self.attn_sink = f32("attn.attn_sink"); self.q_norm = bf("attn.q_norm.weight"); self.kv_norm = bf("attn.kv_norm.weight")
        self.wq_a = fp8lin("attn.wq_a"); self.wq_b = fp8lin("attn.wq_b"); self.wkv = fp8lin("attn.wkv")
        self.wo_a = wo_a(); self.wo_b = fp8lin("attn.wo_b")
        self.hc_attn_fn = f32("hc_attn_fn"); self.hc_ffn_fn = f32("hc_ffn_fn")
        self.hc_attn_base = f32("hc_attn_base"); self.hc_ffn_base = f32("hc_ffn_base")
        self.hc_attn_scale = f32("hc_attn_scale"); self.hc_ffn_scale = f32("hc_ffn_scale")
        self.gate_w = f32("ffn.gate.weight"); self.gate_bias = f32("ffn.gate.bias")
        self.sh_w1 = fp8lin("ffn.shared_experts.w1"); self.sh_w2 = fp8lin("ffn.shared_experts.w2"); self.sh_w3 = fp8lin("ffn.shared_experts.w3")
        self.ratio = 0
        self.is_kv_source = False
        self.layer = args.n_layers + k
        if k == 0:
            self.main_proj = fp8lin("main_proj")
            self.main_norm = bf("main_norm.weight")
        if k == 2:
            self.norm = bf("norm.weight")
            self.markov_embed = bf("markov_head.embed.weight")
            # fp32 once, for the same reason as the LM head: this one is applied once per drafted
            # token, i.e. five times per DSpark step.
            self.markov_head = bf("markov_head.head.weight")
            self.conf_proj = f32("confidence_head.proj.weight")


# ----------------------------------------------------------------------------- caches
class Caches:
    def __init__(self, args: R.Args, max_seq: int, device: str):
        self.args, self.max_seq, self.device = args, max_seq, device
        d = args.head_dim
        # The window gather runs AFTER the whole chunk has been written into the ring, so a ring
        # that is not longer than window_size + MAX_CHUNK hands the first queries of a chunk the
        # KV of its last tokens -- wrong output, no exception. The 0.88 relative error the 64-slot
        # ring produced (docs/architecture.md) is what this refuses to repeat silently.
        if RING < args.window_size + MAX_CHUNK:
            raise ValueError(
                f"DSV41_RING={RING} is too small for DSV41_PREFILL_CHUNK={MAX_CHUNK}: the window "
                f"gather needs more than window_size + chunk = {args.window_size + MAX_CHUNK} "
                f"slots. Raise DSV41_RING (it costs {(args.n_layers + 3) * d * 2 / 1e3:.1f} MB per "
                f"1,000 slots) or lower DSV41_PREFILL_CHUNK.")
        self.win = [torch.zeros(RING, d, dtype=torch.bfloat16, device=device) for _ in range(args.n_layers)]
        self.mtp_win = [torch.zeros(RING, d, dtype=torch.bfloat16, device=device) for _ in range(3)]
        self.ckv = {}
        self.ik = {}
        self.pending = {}  # ratio-2 sources: (kv fp32 [512], score fp32 [512]) of an unpaired position, or None
        for L in args.kv_source_layers:
            r = args.compress_ratios[L]
            n = max_seq // r + 1
            self.ckv[L] = torch.zeros(n, d, dtype=torch.bfloat16, device=device)
            self.ik[L] = torch.zeros(n, args.index_head_dim, dtype=torch.bfloat16, device=device)
            self.pending[L] = None
        self.len = 0  # number of valid positions
        # per-chunk memory for rollback of the compressor state
        self._chunk_inputs = {}  # L -> (S, kv [T,512] fp32, score [T,512] fp32, pending_before)

    def rollback(self, n: int):
        """Discard everything at positions >= n.

        Only the compressor carries state across positions, so only `pending` has to be restored:
        the window ring and the compressed/index caches are append-only and every slot at or after
        n is rewritten by the next forward before anything can read it.
        """
        assert n <= self.len
        for L, (S, kv, sc, before) in self._chunk_inputs.items():
            r = self.args.compress_ratios[L]
            if r == 1:
                continue
            if n % r == 0:
                self.pending[L] = None  # n positions = n/r whole groups, nothing left over
                continue
            p = n - 1  # the position left unpaired at n
            if p >= S:
                self.pending[L] = (kv[p - S], sc[p - S])
            elif p == S - 1:
                self.pending[L] = before  # exactly back to the start of the last chunk
            else:
                raise ValueError(f"rollback({n}) reaches before the last chunk (start {S}); the "
                                 f"compressor input for position {p} is no longer kept")
        self.len = n


class Shared:
    def __init__(self):
        self.ckv = None
        self.ik = None
        self.ratio = 0
        self.topk = None  # [T, k] absolute compressed positions or -1
        self.candidates = None  # [T, n_c] bool


# ----------------------------------------------------------------------------- model
class Model:
    def __init__(self, W: Weights, store, caches: Caches, moe_fn, act_quant: bool = False):
        self.W, self.store, self.c, self.moe_fn = W, store, caches, moe_fn
        self.args = W.args
        self.dev = W.device
        a = self.args
        self.freqs_c = R.precompute_freqs_cis(a.rope_head_dim, caches.max_seq + 8, a.original_seq_len, a.compress_rope_theta,
                                              a.rope_factor, a.beta_fast, a.beta_slow, self.dev)
        self.freqs_w = R.precompute_freqs_cis(a.rope_head_dim, caches.max_seq + 8, 0, a.rope_theta, a.rope_factor,
                                              a.beta_fast, a.beta_slow, self.dev)
        self.tap = None  # optional diagnostic hook: callable(name, L, tensor)
        self.engram_rows = None  # callable (layer, hashes [T,24]) -> [T,24,256] float32
        self.hash_state = None  # reference NgramHashState
        if not act_quant:
            R.act_qdq_fp8 = lambda x, block=32: x.to(torch.bfloat16)
        self.stats = {"attn_s": 0.0, "moe_s": 0.0, "engram_s": 0.0, "tokens": 0}
        self.begin_prompt()

    def _tap(self, name, L, t):
        if self.tap is not None:
            self.tap(name, L, t)

    # ------------------------------------------------------------------ attention
    def _window_positions(self, pos: torch.Tensor):
        """[T, 128] absolute positions each query may see in its sliding window, -1 if none."""
        w = self.args.window_size
        p = pos[:, None] - torch.arange(w - 1, -1, -1, device=self.dev)[None, :]
        return torch.where(p >= 0, p, torch.full_like(p, -1))

    def attention(self, x: torch.Tensor, w, L: int, S: int, sh: Shared, ring: torch.Tensor,
                  freqs: torch.Tensor, mtp_extra=None, win_lo: int = 0, prefill: bool = False):
        """x: [T, d] normed input. Returns [T, d]. `ring` is this layer's window KV ring.

        `win_lo` is the first position whose window KV this ring actually holds. It is 0 everywhere
        except in the decoder replay (SWA Bounded Replay, tech report 3.2.2), where the decoder
        layers have only seen the last 128 prompt tokens and a query near the start of the replay
        would otherwise gather whatever the ring happens to hold below it.

        `prefill` only selects the fp8 gather (DSV41_PREFILL_KV_FP8); the arithmetic is otherwise
        identical for a chunk and for a decode block.
        """
        a = self.args
        T = x.size(0)
        pos = torch.arange(S, S + T, device=self.dev)
        rd = a.rope_head_dim
        fq = freqs[S:S + T]

        self._tap("attn_x", L, x)
        qr = R.rmsnorm(R.qlinear(x, w.wq_a), w.q_norm, a.norm_eps)
        q = R.qlinear(qr, w.wq_b).view(T, a.n_heads, a.head_dim)
        q = torch.cat([q[..., :-rd], R.apply_rotary(q[..., -rd:], fq)], dim=-1)
        self._tap("q", L, q)

        kv = R.rmsnorm(R.qlinear(x, w.wkv), w.kv_norm, a.norm_eps)
        kv = torch.cat([kv[:, :-rd], R.apply_rotary(kv[:, -rd:], fq)], dim=-1)
        self._tap("kv_new", L, kv)
        if mtp_extra is None:
            wpos = self._window_positions(pos)  # [T, 128]
            ring[pos % RING] = kv
            wmask = wpos >= win_lo if win_lo else wpos >= 0
            # The fp8 working copy is worth building only for a real prefill chunk. The bounded
            # replay is window_size queries -- 42 MB of gathered KV at 128 rows -- so there is
            # nothing to save there, and it is the pass that produces the prompt's final logits.
            if PREFILL_KV_FP8 and prefill and T > a.window_size:
                cidx, cmask = self._compressed_idx(x, qr, w, L, S, T, pos, sh) if w.ratio else (None, None)
                kv_all = self._gather_kv_fp8(ring, wpos, sh, cidx, T)
                self._tap("win_kv", L, kv_all[:, :a.window_size]); self._tap("win_mask", L, wmask)
                mask = wmask
                if cmask is not None:
                    self._tap("ckv_rows", L, kv_all[:, a.window_size:]); self._tap("c_mask", L, cmask)
                    mask = torch.cat([wmask, cmask], dim=1)
            else:
                # gather the window AFTER writing this chunk into the ring: a query's window
                # reaches into the chunk itself, which is why RING > window_size + MAX_CHUNK
                wkv = ring[wpos.clamp_min(0) % RING]  # [T, 128, d]
                self._tap("win_kv", L, wkv); self._tap("win_mask", L, wmask)
                kv_all, mask = wkv, wmask
                if w.ratio:
                    ckv_rows, cmask = self._compressed(x, qr, w, L, S, T, pos, sh)
                    self._tap("ckv_rows", L, ckv_rows); self._tap("c_mask", L, cmask)
                    kv_all = torch.cat([wkv, ckv_rows], dim=1)
                    mask = torch.cat([wmask, cmask], dim=1)
        else:
            # DSpark draft attention: window from the main stream's ring (positions <= S-1) + all draft kvs
            main_last = mtp_extra  # position of the last main token in the ring
            wpos = main_last - torch.arange(a.window_size - 1, -1, -1, device=self.dev)
            wpos = torch.where(wpos >= 0, wpos, torch.full_like(wpos, -1))  # [128]
            wkv = ring[wpos.clamp_min(0) % RING][None].expand(T, -1, -1)
            kv_all = torch.cat([wkv, kv[None].expand(T, -1, -1)], dim=1)
            mask = torch.cat([(wpos >= 0)[None].expand(T, -1), torch.ones(T, T, dtype=torch.bool, device=self.dev)], dim=1)

        # The softmax runs on fixed-size token tiles for the same reason the GEMMs do: the batched
        # score/PV products are not invariant to the number of query rows in the call.
        # fp32 PV product, like tools/v41_ref: rounding the probabilities to bf16 first is a
        # cliff that turns 1e-7 fp32 GEMM jitter into 1e-4 output jitter, which is enough to flip
        # a borderline router top-k and make the MoE output depend on the chunk length.
        o = self._softmax_attn(q, kv_all, mask, w.attn_sink, prefill)
        o = torch.cat([o[..., :-rd], R.apply_rotary(o[..., -rd:], fq, inverse=True)], dim=-1)
        o = o.reshape(T, a.o_groups, -1)
        # grouped output projection: "sgd,grd->sgr" is a GEMM with M = number of tokens, so it too
        # has to run on fixed-size token tiles (it differs most visibly at a 1-token chunk).
        o = R.wo_a_proj(o, w.wo_a, tiled=True)
        out = R.qlinear(o.flatten(1), w.wo_b)
        self._tap("attn_out", L, out)
        return out

    def _softmax_attn(self, q, kv_all, mask, sink, prefill: bool = False):
        """Sinked softmax attention over [T, n, d] KV, in fixed-size query tiles.

        Padding rows are all-masked: their scores are -inf, so m clamps to -1e30, p is 0 and the
        sink term makes the denominator +inf -- 0/inf = 0, no NaN.

        These two einsums are the two largest kernels of a prefill chunk: at ATTN_TILE=64 a
        2,048-token chunk is 32 tiles, each tile issues one score product and one PV product, and
        the encoder is 21 layers -- 672 of each, 855 ms together in the profile behind
        docs/gemm-dispatch.md, against 893 ms for both FP4 MoE kernels. They are fp32, and fp32
        means the SIMT pipeline: `cutlass_80_simt_sgemm_128x32_8x5_tn_align1` for the score
        product and the `_nn_` variant for the PV product, neither of which touches a tensor core.
        DSV41_PREFILL_ATTN_GEMM (engine/prefill_attn_gemm.py) is the switch that puts them there;
        `mode_for` returns fp32 for every call that is not a prefill call, so the decode path is
        untouched, and fp32 is byte-identical to the engine that shipped -- `.to(torch.float32)`
        of a bf16 tensor is `.float()`, and the two `.to(dt)` calls are no-ops at dt=float32.
        """
        T = q.size(0)
        scale = self.args.head_dim ** -0.5
        B = ATTN_TILE if ATTN_TILE > 0 else T
        mode = AG.mode_for(prefill)
        dt = torch.bfloat16 if mode == "bf16" else torch.float32

        def tile(qt, kvt, mt):
            # .float() on the score product so the softmax is fp32 in every mode: bf16 operands
            # still accumulate in fp32 on the tensor cores, it is only the epilogue that rounds.
            scores = torch.einsum("thd,tnd->thn", qt, kvt).float() * scale
            scores = scores.masked_fill(~mt[:, None, :], float("-inf"))
            mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(scores - mx)
            denom = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - mx)
            return torch.einsum("thn,tnd->thd", (p / denom).to(dt), kvt)

        outs = []
        with AG.math_scope(mode):
            for i in range(0, T, B):
                j = min(i + B, T)
                qt, kvt, mt = q[i:j].to(dt), kv_all[i:j].to(dt), mask[i:j]
                n = j - i
                if n < B:  # pad the last tile so every call sees exactly B query rows
                    qt = torch.cat([qt, qt.new_zeros(B - n, *qt.shape[1:])])
                    kvt = torch.cat([kvt, kvt.new_zeros(B - n, *kvt.shape[1:])])
                    mt = torch.cat([mt, mt.new_zeros(B - n, mt.size(1))])
                outs.append(tile(qt, kvt, mt)[:n])
        return torch.cat(outs).to(torch.bfloat16)

    def _gather_kv_fp8(self, ring, wpos, sh: Shared, cidx, T: int):
        """The same [T, window_size (+ index_topk), head_dim] gather the bf16 path builds, written
        straight into one fp8 buffer instead of two bf16 gathers plus a concatenation of both.

        That is where the memory goes: at T=2048 the bf16 path holds the window gather (128 KB a
        token), the compressed gather (512 KB) and their concatenation (640 KB) at the same moment,
        1,280 KB a token; this holds 320 KB a token and a tile of bf16 scratch that does not grow
        with T. The values are the bf16 ones rounded to e4m3 -- elementwise, so the tiling is
        invisible in the result -- and `_softmax_attn` widens them back to fp32 per query tile
        exactly as it widens bf16 today.
        """
        a = self.args
        d = a.head_dim
        n = a.window_size + (a.index_topk if cidx is not None else 0)
        kv_all = torch.empty(T, n, d, dtype=FP8_KV_DTYPE, device=self.dev)
        for i in range(0, T, KV_GATHER_TILE):
            j = min(i + KV_GATHER_TILE, T)
            g = ring[wpos[i:j].clamp_min(0) % RING]
            kv_all[i:j, :a.window_size].copy_(g.clamp_(-FP8_KV_MAX, FP8_KV_MAX))
            if cidx is not None:
                g = sh.ckv[cidx[i:j].clamp_min(0)]
                kv_all[i:j, a.window_size:].copy_(g.clamp_(-FP8_KV_MAX, FP8_KV_MAX))
        return kv_all

    def _compressed(self, x, qr, w, L, S, T, pos, sh: Shared):
        """Produce/read the shared compressed KV for this chunk; run/reuse the indexer; return the
        gathered rows [T, k, d] and their mask [T, k]."""
        idx, m = self._compressed_idx(x, qr, w, L, S, T, pos, sh)
        return sh.ckv[idx.clamp_min(0)], m

    def _compressed_idx(self, x, qr, w, L, S, T, pos, sh: Shared):
        """Everything `_compressed` does except the final gather: returns the [T, index_topk]
        absolute compressed positions (-1 = none) and their mask."""
        a = self.args
        r = w.ratio
        c = self.c
        rd = a.rope_head_dim
        if w.is_kv_source:
            # latent for every position of the chunk (plus the pending unpaired one)
            if r > 1:
                xf = x.float()
                kvl, sc = R.mm(xf, w.comp_wkv), R.mm(xf, w.comp_wgate)
                before = c.pending[L]
                c._chunk_inputs[L] = (S, kvl, sc, before)
                if before is not None:
                    kvl = torch.cat([before[0][None], kvl]); sc = torch.cat([before[1][None], sc])
                    first = S - 1
                else:
                    first = S
                n_tok = kvl.size(0)
                cut = n_tok - n_tok % r
                if n_tok % r:
                    c.pending[L] = (kvl[-1], sc[-1])
                else:
                    c.pending[L] = None
                if cut > 0:
                    g_kv = kvl[:cut].unflatten(0, (-1, r)); g_sc = sc[:cut].unflatten(0, (-1, r))
                    latent = (g_kv * g_sc.softmax(dim=1)).sum(dim=1)
                    latent = R.rmsnorm(latent.to(torch.bfloat16), w.comp_norm, a.norm_eps)
                    j0 = first // r
                else:
                    latent, j0 = None, first // r
            else:
                latent = R.rmsnorm(R.mm(x, w.comp_wkv), w.comp_norm, a.norm_eps)
                j0 = S
            if latent is not None:
                nj = latent.size(0)
                jpos = (j0 + torch.arange(nj, device=self.dev)) * r
                fj = self.freqs_c[jpos]
                if L in self.W.indexers:  # index key from the pre-RoPE latent
                    iw = self.W.indexers[L]
                    k = R.rmsnorm(R.mm(latent, iw.wk), iw.k_norm, a.norm_eps)
                    k = torch.cat([k[:, :-rd], R.apply_rotary(k[:, -rd:], fj)], dim=-1)
                    c.ik[L][j0:j0 + nj] = k
                lat = torch.cat([latent[:, :-rd], R.apply_rotary(latent[:, -rd:], fj)], dim=-1)
                c.ckv[L][j0:j0 + nj] = lat
                self._tap("latent", L, (j0, lat))
            sh.ckv, sh.ik, sh.ratio = c.ckv[L], c.ik[L], r
        assert sh.ratio == r, (L, sh.ratio, r)
        compress_lens = (pos + 1) // r  # visible compressed positions per query
        n_c = int((S + T) // r)
        if L in self.W.indexers:
            sh.topk = self._indexer(x, qr, L, pos, compress_lens, n_c, sh)
        idx = sh.topk  # [T, k] absolute compressed positions, -1 = none
        self._tap("topk", L, idx); self._tap("n_c", L, n_c)
        return idx, idx >= 0

    def _indexer(self, x, qr, L, pos, compress_lens, n_c, sh: Shared):
        a = self.args
        iw = self.W.indexers[L]
        T = x.size(0)
        rd = a.rope_head_dim
        if n_c == 0:
            return self._pad_topk(torch.full((T, 0), -1, dtype=torch.int64, device=self.dev))
        q = R.qlinear(qr, iw.wq_b).view(T, a.index_n_heads, a.index_head_dim)
        q = torch.cat([q[..., :-rd], R.apply_rotary(q[..., -rd:], self.freqs_c[pos])], dim=-1)
        wts = (R.mm(x, iw.weights_proj).float() * (a.index_head_dim ** -0.5 * a.index_n_heads ** -0.5))  # [T, H]
        # fixed [MM_TILE queries x KEY_BLOCK keys] score tiles: n_c grows with the chunk, so a
        # single GEMM over sh.ik[:n_c] would be a different shape in every chunk. Columns past
        # n_c are masked out below, so the padding cannot be selected.
        NB = KEY_BLOCK
        n_pad = max(NB, (n_c + NB - 1) // NB * NB)
        k = sh.ik[:n_pad]
        if k.size(0) < n_pad:
            k = torch.cat([k, k.new_zeros(n_pad - k.size(0), k.size(1))])
        B = ATTN_TILE if ATTN_TILE > 0 else T
        score = torch.empty(T, n_pad, dtype=torch.float32, device=self.dev)
        for i in range(0, T, B):
            j = min(i + B, T)
            qt, wt = q[i:j], wts[i:j]
            if j - i < B:
                qt = torch.cat([qt, qt.new_zeros(B - (j - i), *qt.shape[1:])])
                wt = torch.cat([wt, wt.new_zeros(B - (j - i), wt.size(1))])
            for jb in range(0, n_pad, NB):
                sc = torch.einsum("thd,nd->thn", qt, k[jb:jb + NB])  # bf16
                sc = sc.float().relu_() * wt[:, :, None]
                score[i:j, jb:jb + NB] = sc.sum(dim=1)[:j - i]
        cpos = torch.arange(n_pad, device=self.dev)
        score.masked_fill_(cpos[None, :] >= compress_lens[:, None], float("-inf"))
        is_cand_src = L == a.candidate_source_layer
        if is_cand_src:
            sh.candidates = self._select_candidates(score, compress_lens, a.candidate_topk_blocks, a.candidate_block_size)
        elif 0 <= a.candidate_source_layer < L and sh.candidates is not None:
            score = score.masked_fill(~sh.candidates, float("-inf"))
        k_ = min(a.index_topk, n_c)
        idx = score.topk(k_, dim=-1, sorted=False).indices.sort(dim=-1).values
        idx = torch.where(idx < compress_lens[:, None], idx, torch.full_like(idx, -1))
        return self._pad_topk(idx)

    def _pad_topk(self, idx: torch.Tensor) -> torch.Tensor:
        """Always hand back exactly index_topk columns, the tail filled with -1 (= masked off).

        The number of compressed rows a chunk can reach, min(index_topk, (S+T)//ratio), grows with
        the chunk, so without this the concatenated KV of the attention softmax would be a
        different width in a short chunk than in a long one. The extra columns are fully masked
        and change nothing mathematically, but a different N makes cuBLAS pick a different kernel
        for the score GEMM, and the resulting ulp differences flip router decisions downstream.
        """
        pad = self.args.index_topk - idx.size(1)
        return idx if pad <= 0 else F.pad(idx, (0, pad), value=-1)

    @staticmethod
    def _select_candidates(logits, compress_lens, topk_blocks, block_size):
        width = logits.size(-1)
        scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
        scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
        num_blocks = scores.size(-1)
        last = ((compress_lens - 1) // block_size)[:, None]
        scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device)[None, :] == last, float("inf"))
        top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
        return keep.repeat_interleave(block_size, dim=-1)[..., :width]

    # ------------------------------------------------------------------ blocks
    def moe(self, y: torch.Tensor, w, L: int, prefill: bool, store, arena, n_experts: int):
        a = self.args
        self._tap("moe_in", L, y)
        scores = F.softplus(R.mm(y.float(), w.gate_w)).sqrt()
        k = 3 if n_experts == 128 else a.n_activated_experts
        logits = scores + w.gate_bias
        pm = getattr(self, "prune_mask", None)
        # `n_experts == 128` is the DSpark drafter's own router over its 3 x 128 fully resident
        # experts; nothing is ever pruned there, so neither mode touches it.
        pruned = pm is not None and n_experts != 128 and L in pm
        # DSV41_PRUNE_MODE: what happens to a routing pick whose expert is not resident.
        # `substitute` (default) hides the evicted experts from the router, so the token is computed
        # with six experts it did not ask for, at full renormalised weight. `drop` keeps the
        # router's real six and gives the ones that did not survive a weight of exactly zero.
        # Because the survivors are renormalised to route_scale (norm_topk_prob is on in this
        # checkpoint, and the technical report keeps the bias for selection only), `drop` is
        # top-k' routing with k' = the picks that survived -- about four of six at the keep
        # fractions used here -- not an attenuation toward the shared expert; only a token that
        # loses all six falls through to the shared expert alone. The case for it is that a
        # WRONG expert injects a signal the model was never trained to receive, and the mHC
        # residual then feeds that error into the next layer's mixing coefficients as well.
        # Measured on the generation gate 2026-09-12, substitution at ~30 % displaced
        # routing mass corrupts rare tokens at subword boundaries (`clearTimeout` -> `cleartimeout`,
        # `OSError` -> `oenerror`, `.some` -> `.s.s`) and the model then loops trying to repair
        # them. Dropping is what the REAP-style pruning literature does; substituting is what this
        # engine did. See docs/keep-sets.md and env.example.
        drop = pruned and getattr(self, "prune_drop", False)
        # DSV41_ESCAPE_K -- the escape hatch (engine/escape.py). Before the mask goes on, the
        # router's own scores nominate the best non-resident expert(s) of this layer-step; if one
        # clears the margin the runtime streams it into a transient slot and sets its bit in
        # `pm[L]` IN PLACE, so the mask below is already the patched one and nothing else in this
        # function changes. Decode only: a prefill chunk of 2,048 tokens touches ~370 of a layer's
        # 384 experts, so the rule would fire on all 40 layers of every chunk, thrash an 8-slot
        # ring and pay for tokens that are not where the failure shows.
        # With the hatch off (the default) `escape_rt` does not exist and this is one getattr.
        esc = getattr(self, "escape_rt", None)
        if esc is not None and pruned and not drop and not prefill:
            esc.consider(L, logits, scores, k)
        if pruned and not drop:
            # the router may only pick surviving experts
            logits = logits.masked_fill(~pm[L], float("-inf"))
        # DSV41_PREFILL_TOPK_TEST (engine/prefill_topk.py): measurement-only, prefill-only override
        # of the routed k. Unset -- every served configuration -- this returns `k` unchanged and
        # the topk below is the one this engine has always issued.
        if prefill and n_experts != 128:
            k = PT.prefill_topk(k, drop=drop)
        indices = logits.topk(k, dim=-1)[1]
        weights = scores.gather(1, indices)
        slot_idx = indices
        if drop:
            live = pm[L][indices]                          # bool [T, k]: is this pick resident?
            weights = weights.masked_fill(~live, 0.0)      # exactly 0 -- the term leaves the sum
            # The slot lookup below still has to name an expert that HAS a slot: `lut[L][e]` is -1
            # for an evicted expert and `store.resolve` would fetch it from NVMe, which is the one
            # thing all-resident mode must never do. Which resident expert it is cannot matter --
            # its contribution is multiplied by 0 -- so every displaced pick in column j takes
            # `prune_fallback[L][j]` (see `build_prune_fallback` in engine/v41_engine.py: k distinct
            # resident ids, one per column, which keeps at most two of a token's pairs on one slot).
            slot_idx = torch.where(live, indices, self.prune_fallback[L])
        # Renormalising over what is left: with every pick dropped the sum is 0, the weights stay 0
        # (0 / 1e-20), and only the shared expert contributes to this layer's output.
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
        # The taps carry the router's TRUE choice and the post-zeroing weights, so a profile sees
        # what actually happened rather than the substitution.
        self._tap("route_idx", L, indices); self._tap("route_w", L, weights)
        t0 = time.perf_counter()
        # All-resident configurations carry a device slot table (built by the engine from the
        # store's LRU). Using it here turns routing into one GPU gather instead of a host round-trip
        # plus a Python pass over every (layer, expert) pair in the chunk -- at a 2,048-token chunk
        # that pass was 77 % of prefill (NOTES 2026-09-12). The table is only valid while nothing is
        # evicted, which is exactly the pruned all-resident case; anything else takes the host path.
        lut = getattr(self, "slot_lut", None)
        if lut is not None and n_experts != 128:
            slots = lut[L][slot_idx]
            # every entry of `slot_idx` is resident in both modes, so this is still all hits
            self.stats["hits"] = self.stats.get("hits", 0) + slot_idx.numel()
        else:
            slots = store.resolve(L, slot_idx, prefill)
        routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit).float()
        shared = R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()
        self._tap("moe_routed", L, routed); self._tap("moe_shared", L, shared)
        out = routed + shared
        self.stats["moe_s"] += time.perf_counter() - t0
        return out.to(y.dtype)

    def block(self, h, pre_mix, w, L, S, sh, ring, freqs, prefill, store, arena, n_experts, mtp_extra=None,
              win_lo: int = 0):
        a = self.args
        residual = h
        fused_hc = PS.fused_prefill(prefill)
        attn_pre, attn_post, attn_comb = R.hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, a,
                                                    fused=fused_hc)
        y = R.hc_pre(h, pre_mix)
        y = R.rmsnorm(y, w.attn_norm, a.norm_eps)
        t0 = time.perf_counter()
        y = self.attention(y, w, L, S, sh, ring, freqs, mtp_extra, win_lo=win_lo, prefill=prefill)
        self.stats["attn_s"] += time.perf_counter() - t0
        h = R.hc_post(y, residual, attn_post, attn_comb)
        residual = h
        ffn_pre, ffn_post, ffn_comb = R.hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, a,
                                                 fused=fused_hc)
        y = R.hc_pre(h, attn_pre)
        y = R.rmsnorm(y, w.ffn_norm, a.norm_eps)
        y = self.moe(y, w, L, prefill, store, arena, n_experts)
        h = R.hc_post(y, residual, ffn_post, ffn_comb)
        return h, ffn_pre

    # ------------------------------------------------------------------ SWA bounded replay
    def begin_prompt(self):
        """Drop whatever the previous prompt left in the replay buffer."""
        self._rep = {"h": [], "pre_mix": [], "topk": [], "cand": []}
        self._rep_end = 0
        R.prefill_fp8_cache_clear()   # DSV41_PREFILL_FP8_DEQUANT=cached; a no-op otherwise

    def _rep_keep(self, h, pre_mix, sh: Shared, S: int, T: int):
        """Remember the last `window_size` encoder outputs of the prompt so far.

        Only the tail is ever needed, so each chunk contributes at most `window_size` rows and the
        buffer is trimmed as soon as it has more than that.
        """
        w = self.args.window_size
        n = min(w, T)
        sl = slice(T - n, T)
        r = self._rep
        r["h"].append(h[sl]); r["pre_mix"].append(pre_mix[sl])
        # layers 21..23 reuse layer 20's top-k and layers 24..39 search inside layer 20's candidate
        # pool, and both are computed per query -- so they belong to the queries, not to the caches,
        # and the replay has to carry them across from the encoder pass instead of recomputing them.
        r["topk"].append(sh.topk[sl])
        r["cand"].append(sh.candidates[sl] if sh.candidates is not None else None)
        self._rep_end = S + T
        while sum(t.size(0) for t in r["h"]) - r["h"][0].size(0) >= w:
            for k in r:
                r[k].pop(0)

    def _rep_tail(self):
        w = self.args.window_size
        r = self._rep
        h = torch.cat(r["h"])[-w:]
        pre_mix = torch.cat(r["pre_mix"])[-w:]
        topk = torch.cat(r["topk"])[-w:]
        cands = r["cand"]
        cand = None
        if cands and cands[0] is not None:
            width = max(c.size(1) for c in cands)
            # older chunks scored fewer compressed columns; a query can only ever see columns below
            # its own position, all of which are inside its own chunk's width, so padding the rest
            # with False changes nothing that is reachable.
            cand = torch.cat([c if c.size(1) == width else F.pad(c, (0, width - c.size(1)), value=False)
                              for c in cands])[-w:]
        return h, pre_mix, topk, cand, self._rep_end - h.size(0)

    @torch.inference_mode()
    def decoder_replay(self, need_logits: bool = True):
        """Decoder SWA Bounded Replay (tech report 2.2 / 3.2.2).

        Under CED the decoder's global KV is projected from the last encoder layer's hidden state,
        which the encoder pass has already written for every prompt position. The only thing the
        decoder layers still owe the first decode steps is their own sliding-window KV -- so they
        are run over the last `window_size` prompt tokens only, with SWA truncated to that segment,
        instead of over the whole prompt. The prompt's final logits come from this pass.
        """
        a = self.args
        h, pre_mix, topk, cand, S = self._rep_tail()
        T = h.size(0)
        sh = Shared()
        src = a.candidate_source_layer
        sh.ckv, sh.ik, sh.ratio = self.c.ckv[src], self.c.ik[src], self.args.compress_ratios[src]
        sh.topk, sh.candidates = topk, cand
        main_hiddens = []
        for L in range(src + 1, len(self.W.layers)):
            w = self.W.layers[L]
            if L in a.dspark_target_layer_ids:
                main_hiddens.append(h.float().mean(dim=1))
            freqs = self.freqs_c if w.ratio else self.freqs_w
            h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, True, self.store,
                                    self.store.arena, a.n_routed_experts, win_lo=S)
            self._tap("h", L, h); self._tap("pre_mix", L, pre_mix)
        self.last_h, self.last_pre_mix = h, pre_mix
        logits = None
        if need_logits:
            x = R.hc_pre(h, pre_mix)
            x = R.rmsnorm(x, self.W.norm, a.norm_eps)
            logits = R.head_logits(x, self.W.head)
        self.stats["replay_tokens"] = self.stats.get("replay_tokens", 0) + T
        # The replay is the last thing prefill does, so the cached bf16 dequants are given back
        # here -- before the first decode step, not at the next prompt.
        R.prefill_fp8_cache_clear()
        return logits, (torch.cat(main_hiddens, dim=-1) if main_hiddens else None), S

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, S: int, prefill: bool, need_logits: bool = True,
                encoder_only: bool = False):
        """ids: [T] token ids at positions S..S+T-1. Returns (logits [T, V] fp32 or None, main_hidden [T, 15360]).
        Caches must be valid for positions < S (self.c.len == S).

        ``encoder_only`` stops after the candidate-source layer (the last layer that writes global
        KV, layer 20 here): everything above it is replayed once over the prompt tail by
        `decoder_replay`. It returns (None, None) -- there are no logits and no DSpark hidden
        states below layer 37.
        """
        a = self.args
        assert self.c.len == S, (self.c.len, S)
        T = ids.size(0)
        assert T <= MAX_CHUNK, (T, MAX_CHUNK)
        t0 = time.perf_counter()
        hashes = self.hash_state(ids[None], S)[0] if self.hash_state is not None else None  # [T, 2, 24]
        self.stats["engram_s"] += time.perf_counter() - t0
        h = self.W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1)
        pre_mix = torch.zeros(T, a.hc_mult, device=self.dev)
        pre_mix[:, 0] = 1.0
        sh = Shared()
        main_hiddens = []
        n_layers = len(self.W.layers)
        last = a.candidate_source_layer if encoder_only else n_layers - 1
        for L in range(last + 1):
            w = self.W.layers[L]
            if L in self.W.engram:
                t0 = time.perf_counter()
                li = list(a.engram_layer_ids).index(L)
                rows = self.engram_rows(L, hashes[:, li, :])
                self._tap("engram_rows", L, rows)
                h = R.engram_forward(h, rows, self.W.engram[L], a)
                self._tap("engram_out", L, h)
                self.stats["engram_s"] += time.perf_counter() - t0
            if L in a.dspark_target_layer_ids:
                main_hiddens.append(h.float().mean(dim=1))
            freqs = self.freqs_c if w.ratio else self.freqs_w
            h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, prefill, self.store,
                                    self.store.arena, a.n_routed_experts)
            self._tap("h", L, h); self._tap("pre_mix", L, pre_mix)
        self.c.len = S + T
        self.stats["tokens"] += T
        if encoder_only:
            self._rep_keep(h, pre_mix, sh, S, T)
            return None, None
        logits = None
        self.last_h, self.last_pre_mix = h, pre_mix
        if need_logits and n_layers == a.n_layers:
            x = R.hc_pre(h, pre_mix)
            x = R.rmsnorm(x, self.W.norm, a.norm_eps)
            logits = R.head_logits(x, self.W.head)
        if prefill:
            # A full (non-encoder-only) prefill forward is the whole prompt in one pass, so there
            # is no later chunk to reuse a cached dequant: give the memory back here. The chunk
            # loop runs with encoder_only=True and is freed by decoder_replay instead.
            R.prefill_fp8_cache_clear()
        return logits, (torch.cat(main_hiddens, dim=-1) if main_hiddens else None)

    # ------------------------------------------------------------------ DSpark
    @torch.inference_mode()
    def dspark_seed(self, main_hidden: torch.Tensor, S: int):
        """Write the drafter's window KV for main positions S..S+M-1 from their main hiddens [M, 15360]."""
        a = self.args
        m0 = self.W.mtp[0]
        main_x = R.rmsnorm(R.qlinear(main_hidden.to(torch.bfloat16), m0.main_proj), m0.main_norm, a.norm_eps)
        M = main_x.size(0)
        pos = torch.arange(S, S + M, device=self.dev)
        rd = a.rope_head_dim
        for k, w in enumerate(self.W.mtp):
            kv = R.rmsnorm(R.qlinear(main_x, w.wkv), w.kv_norm, a.norm_eps)
            kv = torch.cat([kv[:, :-rd], R.apply_rotary(kv[:, -rd:], self.freqs_w[S:S + M])], dim=-1)
            self.c.mtp_win[k][pos % RING] = kv

    @torch.inference_mode()
    def dspark_draft(self, tok: int, last_main_pos: int, temperature: float):
        """Draft block: returns (draft ids [B], draft probs [B, V] fp32 at the given temperature,
        confidence [B]) for B = the DSpark block size. Queries sit at last_main_pos+1 .. +B."""
        a = self.args
        from engine.fastdecode import T_DRAFT as B   # DSV41_BLOCK, default the checkpoint's 5
        ids = torch.full((B,), 128799, dtype=torch.long, device=self.dev)
        ids[0] = tok
        h = self.W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1)
        pre_mix = torch.zeros(B, a.hc_mult, device=self.dev); pre_mix[:, 0] = 1.0
        S = last_main_pos + 1
        sh = Shared()
        for k, w in enumerate(self.W.mtp):
            h, pre_mix = self.block(h, pre_mix, w, a.n_layers + k, S, sh, self.c.mtp_win[k], self.freqs_w, False,
                                    self.W.dspark_store, self.W.dspark_arena, 128, mtp_extra=last_main_pos)
        w = self.W.mtp[2]
        x = R.hc_pre(h, pre_mix)
        # The reference feeds the UN-normed hc_pre output to the confidence head and the normed one
        # to the LM head (inference/model.py::DSparkBlock.forward_head), so keep both.
        x_pre = x
        x = R.rmsnorm(x, w.norm, a.norm_eps)
        logits = R.head_logits(x, self.W.head)  # [B, V] fp32
        out = torch.empty(B + 1, dtype=torch.long, device=self.dev)
        out[0] = tok
        probs = []
        embeds = []
        for i in range(B):
            e = w.markov_embed[out[i]]  # [256]
            bias = F.linear(e.to(torch.bfloat16)[None], w.markov_head).float()[0]  # [V]
            lg = logits[i] + bias
            if temperature <= 0:
                p = torch.zeros_like(lg); p[lg.argmax()] = 1.0
                nxt = lg.argmax()
            else:
                p = torch.softmax(lg / temperature, dim=-1)
                nxt = torch.multinomial(p, 1)[0]
            out[i + 1] = nxt
            probs.append(p)
            embeds.append(e.float())
        # DSparkConfidenceHead returns the raw projection (no sigmoid); adaptive verification is off
        # in this engine, so it is reported, not acted on.
        conf = (torch.cat([x_pre.float(), torch.stack(embeds)], dim=-1) @ w.conf_proj.T).squeeze(-1)
        return out[1:], torch.stack(probs), conf
