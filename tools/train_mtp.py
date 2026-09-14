#!/usr/bin/env python3
"""train_mtp.py -- fine-tune the shipped DSpark head on the target's own outputs (FastMTP).

    python3 tools/train_mtp.py --plan --data data/draft          # sizes and memory, no torch needed
    python3 tools/train_mtp.py --data data/draft --steps 3000 --out weights/mtp_finetuned.safetensors
    DSV41_MTP_WEIGHTS=weights/mtp_finetuned.safetensors ./start.sh

Why this and not something else. The verify step is ~145 ms and flat across workloads, so served
tok/s is `accept_len_mean / 0.145` and nothing else. Draft trees, EAGLE-style heads and adaptive
draft lengths were measured and did not pay. What is left is the acceptance of the head that is
already in the checkpoint: 5.07 accepted tokens a step on a single-file HTML page, 2.46 on a long
story (RESULTS.md 4.3). The head was trained by its author on its author's data; this fine-tunes it
on what THIS target actually emits, on the registers where it is weak.

The recipe is FastMTP (Red Hat / vLLM, 2026-09), which is four decisions:

  1. **Start from the shipped MTP weights.** Not a fresh head: `mtp.*` already carries a working
     drafter and the fine-tune is a few thousand steps of adaptation, not a training run.
  2. **Freeze the target.** It is never loaded. The recorder (`engine/draft_record.py`) wrote the
     target's last hidden state, its settled token and its top-32 next-token logits for every
     position it decoded, so a training step reads 31 KB per token off NVMe instead of running
     510 GB of weights.
  3. **Share the full, unreduced LM head.** The drafter's logits come from the target's own
     `head.weight`, all 129,280 rows of it, frozen. Slicing the vocabulary would train a head that
     cannot be verified against the target it has to agree with.
  4. **A teacher-forced recursive loop that mirrors serving, with an exponentially decayed
     per-step loss.** `beta = 0.6`: step 1 of the draft is worth 1.0, step 5 is worth 0.13. That is
     not arbitrary -- acceptance is a LEADING-prefix quantity. A draft that is right at step 4 and
     wrong at step 1 accepts nothing, so the steps are worth what they contribute.

What "recursive" means in THIS architecture, exactly, because it is not the same as a chain of
independent single-token heads. One DSpark draft is a single forward over the whole block: the
token ids are `[t_p0, noise, noise, noise, noise]`, the attention sees a 128-position window of the
TARGET's hidden states (ending at p0-1) plus the block's own five positions, and the only serial
dependence is the rank-256 Markov head, which biases step d's logits by the drafter's own prediction
at step d-1. So the loop here is: window from recorded target hidden states, block forward,
Markov chain over the previous predictions -- exactly `engine/fastdecode.py::_draft`, in plain
autograd-friendly torch. `--markov-input` chooses whether the chain is fed the drafter's own
argmax (`self`, the default -- what serving does) or the target's token (`target`, teacher forcing
of the chain as well).

The loss per draft step d, against the target's own distribution:

    L_d = beta^d * ( KL( p_target || p_draft )  +  ce_weight * CE(draft_d, token_d) )

The KL is forward KL over the target's recorded top-32, renormalized over those 32 (the tail is
neither recorded nor worth chasing: what acceptance depends on is agreement at the top of the
distribution). The cross-entropy on the token the target actually settled on is what keeps the
argmax sharp, which is what greedy verification compares.

**Memory.** The one number that decides whether this fits: the drafter's own routed experts are
3 x 128 x 35.4 M = 13.6 G parameters -- 7.2 GB packed FP4 in the checkpoint, 27.2 GB dequantized to
bf16. They are FROZEN (their gradient would be another 27 GB and their AdamW state 109 GB) but they
still have to be resident, because the draft runs through them. Against a box with ~120 GB while the
engine is NOT running:

    frozen DSpark routed experts, bf16            27.2 GB
    frozen embed + LM head, bf16                   2.6 GB
    trainable dense MTP parameters, fp32           2.5 GB   (636 M parameters)
    AdamW: fp32 grad + two moments                 7.6 GB
    bf16 casts, activations, logits, workspace     ~2.5 GB  (at --batch 16)
                                                 --------
                                                  ~43 GB

`--plan` prints that table from the checkpoint's own headers before anything is allocated, and the
run refuses to start if the box has less free than the plan asks for. Training the routed experts
too is not supported: it would add 190 GB of gradient and AdamW state, which `--plan` says as well.

**What comes out.** `mtp_finetuned.safetensors` -- the TRAINED tensors as bf16, under the
checkpoint's own names, so everything else keeps the checkpoint's own bytes -- plus a manifest, and
the acceptance proxy before and after. The proxy is top-1 agreement at draft step d on a held-out
split of the recorded data: exactly what greedy verification tests, so the mean leading agreement
+ 1 is directly comparable to the engine's `accept_len_mean`. It is a proxy and not the number:
`tools/verify_mtp.sh` runs the real server and reports the real one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))

import draft_record as DR  # noqa: E402  (torch-free: the record format only)

# The dense tensors of one DSpark block, by group. `--train` selects groups; everything not selected
# keeps its checkpoint value and is not written to the output file at all -- the engine's override
# is per tensor, so a file with three tensors in it is a legal three-tensor override.
GROUPS = {
    "attn": ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b"),
    "shared": ("ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"),
    "gate": ("ffn.gate",),
    "norms": ("attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm", "norm",
              "hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
              "attn.attn_sink"),
    "main": ("main_proj", "main_norm"),
    "markov": ("markov_head.embed", "markov_head.head"),
}
DEFAULT_GROUPS = ("attn", "shared", "gate", "norms", "main", "markov")

# In a selected group and still frozen: the router's correction bias steers SELECTION only -- it is
# added to the scores before the top-3 and never to the weights that multiply an expert's output --
# so no gradient reaches it and an optimizer step on it would be noise on a dead parameter. The
# checkpoint's value is kept. (Same reasoning as DeepSeek-V3 eq. 13, which the engine follows.)
NO_GRAD_NAMES = ("ffn.gate.bias",)


# =============================================================================
# the plan: what is in the checkpoint, and what training it would cost
# =============================================================================

def safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def mtp_inventory(model_dir: str) -> dict:
    """{name: (shape, dtype)} for every `mtp.*` tensor, plus embed/head, from the headers alone."""
    wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    want = [k for k in wm if k.startswith("mtp.") or k in ("embed.weight", "head.weight")]
    by_file = {}
    for k in want:
        by_file.setdefault(wm[k], []).append(k)
    out = {}
    for fn, keys in by_file.items():
        hdr = safetensors_header(os.path.join(model_dir, fn))
        for k in keys:
            h = hdr[k]
            out[k] = (tuple(h["shape"]), h["dtype"])
    return out


def _elements(shape, name: str) -> int:
    """Dequantized element count. A packed-FP4 weight is stored [N, K/2] and a `.scale` table is
    bookkeeping, not parameters."""
    n = 1
    for s in shape:
        n *= s
    return n * 2 if name.endswith(".weight") and ".experts." in name else n


def plan(model_dir: str, groups, record_bytes: int, tokens: int = 300000, batch: int = 16) -> dict:
    inv = mtp_inventory(model_dir)
    dense, experts, shared_head = 0, 0, 0
    trainable = 0
    sel = set()
    for g in groups:
        sel.update(GROUPS[g])
    for name, (shape, _dt) in inv.items():
        if name.endswith(".scale"):
            continue
        n = _elements(shape, name)
        if name in ("embed.weight", "head.weight"):
            shared_head += n
            continue
        if ".experts." in name:
            experts += n
            continue
        dense += n
        suffix = name[len("mtp.0."):]       # every `mtp.K.` prefix is the same six characters
        base = suffix.rsplit(".weight", 1)[0].rsplit(".bias", 1)[0]
        if base in sel and suffix not in NO_GRAD_NAMES:
            trainable += n
    gb = 1e9
    mem = {
        "frozen routed experts (bf16)": experts * 2 / gb,
        "frozen embed + head (bf16)": shared_head * 2 / gb,
        "trainable parameters (fp32)": trainable * 4 / gb,
        "AdamW grad + two moments (fp32)": trainable * 12 / gb,
        "frozen dense + bf16 casts": (dense - trainable) * 2 / gb + trainable * 2 / gb,
        "activations, logits, workspace": 0.25 + batch * 0.06,
    }
    return {
        "dense_params": dense, "expert_params": experts, "shared_params": shared_head,
        "trainable_params": trainable, "memory_gb": mem, "total_gb": sum(mem.values()),
        "train_experts_would_add_gb": experts * 14 / gb,
        "data_gb": tokens * record_bytes / gb,
    }


def print_plan(p: dict, groups, tokens: int):
    print(f"DSpark head: {p['dense_params'] / 1e6:.0f} M dense parameters, "
          f"{p['expert_params'] / 1e9:.1f} G routed-expert parameters (frozen)")
    print(f"trainable ({', '.join(groups)}): {p['trainable_params'] / 1e6:.0f} M parameters")
    print(f"\n{'what':<36} {'GB':>7}")
    for k, v in p["memory_gb"].items():
        print(f"{k:<36} {v:>7.1f}")
    print(f"{'total':<36} {p['total_gb']:>7.1f}")
    print(f"\ntraining the routed experts too would add {p['train_experts_would_add_gb']:.0f} GB "
          f"(grad + AdamW state): not supported")
    print(f"recorded data for {tokens} positions: {p['data_gb']:.1f} GB on disk")


# =============================================================================
# the recorded data
# =============================================================================

def usable_starts(positions: np.ndarray, flags: np.ndarray, window: int, draft: int) -> np.ndarray:
    """Indices i of a shard's records that can start a training sample.

    A sample needs records i-window .. i+draft to be ONE contiguous run of positions -- the window
    is the drafter's attention window and a gap in it is a different sequence -- and a recorded
    distribution at i .. i+draft-1, which is the supervision for draft steps 1..draft. The records
    written for the prompt tail carry hidden states and no distribution, which is why they are
    usable as window and never as a target.
    """
    n = len(positions)
    if n < window + draft + 1:
        return np.zeros(0, dtype=np.int64)
    step = np.zeros(n, dtype=bool)
    step[1:] = positions[1:] == positions[:-1] + 1
    idx_all = np.arange(n, dtype=np.int64)
    # run_start[i] = the index of the first record of i's contiguous run: the running maximum of
    # "this record starts a run" over the indices
    run_start = np.maximum.accumulate(np.where(step, 0, idx_all))
    has = (flags & DR.FLAG_HAS_DIST).astype(bool)
    # cumulative count of records with a distribution, to test a whole span in O(1)
    cum = np.concatenate([[0], np.cumsum(has)])
    idx = np.arange(n, dtype=np.int64)
    ok = (idx >= window) & (idx + draft < n)
    ok &= run_start <= (idx - window)                      # the window is in the same run
    ok &= np.roll(run_start, -draft) <= idx                # ... and so is i+draft
    ok &= (cum[np.clip(idx + draft, 0, n)] - cum[idx]) == draft
    return idx[ok]


class Shards:
    """The recorded shards of one or more directories, and every sample start in them."""

    def __init__(self, dirs, window: int, draft: int, hidden_dim: int = DR.HIDDEN_DIM,
                 topk: int = DR.TOPK):
        self.window, self.draft = window, draft
        self.hidden_dim, self.topk = hidden_dim, topk
        self.shards, self.samples = [], []
        for d in dirs:
            if not os.path.isdir(d):
                raise SystemExit(f"no such recorder directory: {d}")
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".bin"):
                    continue
                path = os.path.join(d, fn)
                man = os.path.join(d, fn[:-4] + ".json")
                if os.path.exists(man):
                    m = DR.load_manifest(man)
                    if m.get("hidden_dim") != hidden_dim or m.get("topk") != topk:
                        raise ValueError(f"{path}: recorded hidden_dim/topk "
                                         f"{m.get('hidden_dim')}/{m.get('topk')} != "
                                         f"{hidden_dim}/{topk}")
                rec = DR.read_shard(path, hidden_dim, topk)
                if len(rec) < window + draft + 1:
                    continue
                si = len(self.shards)
                self.shards.append(rec)
                for i in usable_starts(np.asarray(rec["position"]), np.asarray(rec["flags"]),
                                       window, draft):
                    self.samples.append((si, int(i)))

    def __len__(self):
        return len(self.samples)

    def split(self, held_out: float, seed: int):
        """A held-out split by SHARD, not by sample: two samples 3 positions apart share 125 of
        their 128 window rows, so a random split over samples would hold out nothing."""
        n_shards = len(self.shards)
        order = list(range(n_shards))
        random.Random(seed).shuffle(order)
        n_hold = max(1, int(round(n_shards * held_out)))
        hold = set(order[:n_hold])
        train = [s for s in self.samples if s[0] not in hold]
        test = [s for s in self.samples if s[0] in hold]
        return train, test

    def batch(self, picks, device):
        """Assemble a batch on the device. Returns a dict of tensors:
        hidden [B, W, H] bf16, win_pos [B, W], tok0 [B], pos0 [B],
        tgt_ids [B, D, K] int64, tgt_vals [B, D, K] fp32, tgt_tok [B, D] int64.
        """
        import torch

        W, D, K = self.window, self.draft, self.topk
        B = len(picks)
        hidden = np.empty((B, W, self.hidden_dim), dtype=np.uint16)
        win_pos = np.empty((B, W), dtype=np.int64)
        tok0 = np.empty(B, dtype=np.int64)
        pos0 = np.empty(B, dtype=np.int64)
        tgt_ids = np.empty((B, D, K), dtype=np.int64)
        tgt_vals = np.empty((B, D, K), dtype=np.float32)
        tgt_tok = np.empty((B, D), dtype=np.int64)
        for b, (si, i) in enumerate(picks):
            r = self.shards[si]
            hidden[b] = r["hidden"][i - W:i]
            win_pos[b] = r["position"][i - W:i]
            tok0[b] = r["token"][i]
            pos0[b] = r["position"][i]
            tgt_ids[b] = r["top_ids"][i:i + D]
            tgt_vals[b] = r["top_vals"][i:i + D]
            tgt_tok[b] = r["token"][i + 1:i + D + 1]
        t = lambda x, dt: torch.as_tensor(x, device=device, dtype=dt)  # noqa: E731
        return {
            "hidden": torch.from_numpy(hidden.view(np.int16)).to(device).view(torch.bfloat16),
            "win_pos": t(win_pos, torch.long), "tok0": t(tok0, torch.long), "pos0": t(pos0, torch.long),
            "tgt_ids": t(tgt_ids, torch.long), "tgt_vals": t(tgt_vals, torch.float32),
            "tgt_tok": t(tgt_tok, torch.long),
        }


# =============================================================================
# the head
# =============================================================================

# The tensors the engine reads in fp32 (v41_ref/MTPWeights `f32(...)`). Everything else is bf16.
FP32_NAMES = ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale",
              "hc_ffn_scale", "attn.attn_sink", "ffn.gate.weight", "ffn.gate.bias",
              "confidence_head.proj.weight")


def load_head(model_dir: str, device: str, groups, cfg, log=print):
    """Load `mtp.*` -- dense tensors dequantized from their stored fp8, routed experts from packed
    FP4 -- plus the shared embed and LM head.

    Training happens in bf16 with fp32 masters for what is trainable, and the result is re-quantized
    on the way back into the engine (engine/model.py::_mtp_override). That round trip is the honest
    cost of fine-tuning a head whose serving format is fp8/fp4, and it is measured at the end of the
    run rather than assumed away.

    Returns (trainable {name: fp32 tensor}, frozen {name: tensor, "mtp.k.experts": [w1, w2, w3]}).
    """
    import torch
    from safetensors import safe_open
    import v41_ref as R

    wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(name):
        fn = wm[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(model_dir, fn), "pt", device="cpu")
        return handles[fn].get_tensor(name)

    sel = set()
    for g in groups:
        sel.update(GROUPS[g])

    P, frozen = {}, {}
    t0 = time.time()
    for name in ("embed.weight", "head.weight"):
        frozen[name] = get(name).to(device).to(torch.bfloat16)
    log(f"embed + head on the GPU ({time.time() - t0:.0f}s)")

    for k in range(3):
        p = f"mtp.{k}."
        for full in sorted(n for n in wm if n.startswith(p)):
            if full.endswith(".scale") or ".experts." in full:
                continue
            suffix = full[len(p):]
            base = suffix.rsplit(".weight", 1)[0].rsplit(".bias", 1)[0]
            t = get(full).to(device)
            scale = full[: -len(".weight")] + ".scale" if full.endswith(".weight") else None
            if scale in wm:
                t = R.dequant_fp8_block(t, get(scale).to(device))
                if suffix == "attn.wo_a.weight":   # [G*R, K] as stored -> [G, R, K] as used
                    t = t.view(cfg.o_groups, cfg.o_lora_rank, -1)
            t = t.float() if suffix in FP32_NAMES else t.to(torch.bfloat16)
            if base in sel and suffix not in NO_GRAD_NAMES:
                P[full] = t.float().detach().requires_grad_(True)
            else:
                frozen[full] = t
        frozen[p + "experts"] = _load_experts(get, p, device, R, torch)
        log(f"DSpark block {k} loaded ({time.time() - t0:.0f}s)")
    return P, frozen


def _load_experts(get, p: str, device: str, R, torch):
    """The block's 128 routed experts, packed FP4 -> bf16, into three preallocated [128, N, K]
    tensors. Preallocated and filled one expert at a time on purpose: a list of 384 tensors
    followed by a `torch.stack` would need the 9 GB of a block twice."""
    out = []
    for w in ("w1", "w2", "w3"):
        first = R.dequant_fp4_packed(get(f"{p}ffn.experts.0.{w}.weight").to(device),
                                     get(f"{p}ffn.experts.0.{w}.scale").to(device))
        buf = torch.empty(128, *first.shape, dtype=torch.bfloat16, device=device)
        buf[0] = first
        del first
        for e in range(1, 128):
            q = f"{p}ffn.experts.{e}.{w}."
            buf[e] = R.dequant_fp4_packed(get(q + "weight").to(device), get(q + "scale").to(device))
        out.append(buf)
    return out


def _w(P, frozen, name, dtype=None):
    """One weight, trainable (an fp32 master, cast per use) or frozen."""
    t = P.get(name)
    if t is None:
        t = frozen[name]
    return t if dtype is None else t.to(dtype)


def _rope(R, x, freqs, rd: int, inverse: bool = False):
    """Rotate the last `rd` dims. x: [S, d] or [S, H, d]; freqs: [S, rd/2]."""
    import torch

    return torch.cat([x[..., :-rd], R.apply_rotary(x[..., -rd:], freqs, inverse=inverse)], dim=-1)


def draft_forward(cfg, P, frozen, b, freqs_w, markov_input: str = "self"):
    """One DSpark draft for a batch, mirroring engine/fastdecode.py::_draft.

    b: the dict from Shards.batch. Returns (logits [B, D, V] fp32, with the Markov bias included,
    argmax [B, D]). Everything but the attention runs with the batch flattened into the row
    dimension, which is what the engine's own shapes are: the draft is 5 rows, not a sequence.
    """
    import torch
    import torch.nn.functional as F
    import v41_ref as R

    dev = b["tok0"].device
    B, Wn = b["win_pos"].shape
    D = b["tgt_tok"].size(1)
    hc, dim, rd = cfg.hc_mult, cfg.dim, cfg.rope_head_dim
    nh, hd = cfg.n_heads, cfg.head_dim
    N = B * D

    def lin(x, name):
        return F.linear(x, _w(P, frozen, name, torch.bfloat16))

    # ---- the window: the target's hidden states -> main_proj -> each block's KV ring.
    # This is Model.dspark_seed, and it is inside the trained graph on purpose: main_proj is the
    # drafter's entire interface to the target, so it is the tensor most worth moving.
    f_win = freqs_w[b["win_pos"].reshape(-1)]
    hid = b["hidden"].reshape(B * Wn, -1)
    main_x = R.rmsnorm(lin(hid, "mtp.0.main_proj.weight"),
                       _w(P, frozen, "mtp.0.main_norm.weight"), cfg.norm_eps)
    win_kv = []
    for k in range(3):
        kv = R.rmsnorm(lin(main_x, f"mtp.{k}.attn.wkv.weight"),
                       _w(P, frozen, f"mtp.{k}.attn.kv_norm.weight"), cfg.norm_eps)
        win_kv.append(_rope(R, kv, f_win, rd).view(B, Wn, hd))

    # ---- the block: [t_p0, noise x (D-1)], the ids the engine drafts from
    ids = torch.full((B, D), cfg.dspark_noise_token_id, dtype=torch.long, device=dev)
    ids[:, 0] = b["tok0"]
    h = frozen["embed.weight"][ids].unsqueeze(2).expand(B, D, hc, dim).reshape(N, hc, dim)
    pre_mix = torch.zeros(N, hc, device=dev)
    pre_mix[:, 0] = 1.0
    pos = (b["pos0"][:, None] + torch.arange(D, device=dev)[None, :]).reshape(-1)
    f_blk = freqs_w[pos]
    scale = hd ** -0.5
    # Every window row of a sample is a real recorded position, so the engine's `wpos >= 0` mask is
    # all true here. It is kept because the mask is part of the math, not of the data.
    mask = torch.cat([b["win_pos"] >= 0, torch.ones(B, D, dtype=torch.bool, device=dev)], dim=1)

    for k in range(3):
        p = f"mtp.{k}."
        residual = h
        pre, post, comb = _hc(cfg, P, frozen, h, p + "hc_attn")
        y = R.rmsnorm(R.hc_pre(h, pre_mix), _w(P, frozen, p + "attn_norm.weight"), cfg.norm_eps)
        qr = R.rmsnorm(lin(y, p + "attn.wq_a.weight"), _w(P, frozen, p + "attn.q_norm.weight"), cfg.norm_eps)
        q = _rope(R, lin(qr, p + "attn.wq_b.weight").view(N, nh, hd), f_blk, rd).view(B, D, nh, hd)
        kv = _rope(R, R.rmsnorm(lin(y, p + "attn.wkv.weight"),
                                _w(P, frozen, p + "attn.kv_norm.weight"), cfg.norm_eps),
                   f_blk, rd).view(B, D, hd)
        kv_all = torch.cat([win_kv[k], kv], dim=1)                       # [B, W + D, hd]
        scores = torch.einsum("bthd,bnd->bthn", q.float(), kv_all.float()) * scale
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
        pr = torch.exp(scores - mx)
        sink = _w(P, frozen, p + "attn.attn_sink", torch.float32)
        denom = pr.sum(-1, keepdim=True) + torch.exp(sink[None, None, :, None] - mx)
        o = torch.einsum("bthn,bnd->bthd", pr / denom, kv_all.float()).to(torch.bfloat16)
        o = _rope(R, o.reshape(N, nh, hd), f_blk, rd, inverse=True).reshape(N, cfg.o_groups, -1)
        o = torch.einsum("sgd,grd->sgr", o, _w(P, frozen, p + "attn.wo_a.weight", torch.bfloat16))
        y = lin(o.flatten(1), p + "attn.wo_b.weight")
        h = R.hc_post(y, residual, post, comb)
        residual = h
        fpre, fpost, fcomb = _hc(cfg, P, frozen, h, p + "hc_ffn")
        y = R.rmsnorm(R.hc_pre(h, pre), _w(P, frozen, p + "ffn_norm.weight"), cfg.norm_eps)
        out = _moe(cfg, P, frozen, y, k)
        out = out + R.expert_ffn(y, _w(P, frozen, p + "ffn.shared_experts.w1.weight", torch.bfloat16),
                                 _w(P, frozen, p + "ffn.shared_experts.w2.weight", torch.bfloat16),
                                 _w(P, frozen, p + "ffn.shared_experts.w3.weight", torch.bfloat16),
                                 cfg.swiglu_limit).float()
        h = R.hc_post(out.to(torch.bfloat16), residual, fpost, fcomb)
        pre_mix = fpre

    x = R.rmsnorm(R.hc_pre(h, pre_mix), _w(P, frozen, "mtp.2.norm.weight"), cfg.norm_eps)
    logits = F.linear(x, frozen["head.weight"]).float().view(B, D, -1)

    # ---- the Markov chain: step d's logits are biased through the token at step d-1
    emb = _w(P, frozen, "mtp.2.markov_head.embed.weight", torch.bfloat16)
    mh = _w(P, frozen, "mtp.2.markov_head.head.weight", torch.bfloat16)
    prev = b["tok0"]
    outs, ams = [], []
    for d in range(D):
        lg = logits[:, d] + F.linear(emb[prev], mh).float()
        outs.append(lg)
        am = lg.argmax(-1)
        ams.append(am)
        prev = am.detach() if markov_input == "self" else b["tgt_tok"][:, d]
    return torch.stack(outs, dim=1), torch.stack(ams, dim=1)


def _hc(cfg, P, frozen, h, prefix):
    """The Hyper-Connection mixing coefficients of one sublayer, as v41_ref.hc_mixes does them
    (without the MM_TILE row tiling, which exists for chunk invariance at serve time)."""
    import torch
    import torch.nn.functional as F
    import v41_ref as R

    xb = h.flatten(1).float()
    rsqrt = torch.rsqrt(xb.square().mean(-1, keepdim=True) + cfg.norm_eps)
    mixes = F.linear(xb, _w(P, frozen, prefix + "_fn", torch.float32)) * rsqrt
    return R.hc_split_sinkhorn(mixes, _w(P, frozen, prefix + "_scale", torch.float32),
                               _w(P, frozen, prefix + "_base", torch.float32),
                               cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps)


def _moe(cfg, P, frozen, y, k: int):
    """The block's 128-expert MoE, top-3, routed exactly as `_draft` routes it: an fp32 gate,
    softplus-sqrt scores, selection biased by `gate_bias`, the weights renormalized over the three
    and scaled by `route_scale`.

    The experts are frozen, so this loops over the experts the batch actually touched rather than
    gathering their weight tensors: one pick is 35 M parameters, and a batch of 16 x 5 tokens makes
    240 picks."""
    import torch
    import torch.nn.functional as F
    import v41_ref as R

    p = f"mtp.{k}."
    scores = F.softplus(F.linear(y.float(), _w(P, frozen, p + "ffn.gate.weight", torch.float32))).sqrt()
    idx = (scores + _w(P, frozen, p + "ffn.gate.bias", torch.float32)).topk(3, dim=-1)[1]
    wts = scores.gather(1, idx)
    wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20) * cfg.route_scale
    w1, w2, w3 = frozen[p + "experts"]
    out = torch.zeros(y.size(0), cfg.dim, device=y.device, dtype=torch.float32)
    for e in idx.reshape(-1).unique().tolist():
        rows, col = (idx == e).nonzero(as_tuple=True)
        got = R.expert_ffn(y[rows], w1[e], w2[e], w3[e], cfg.swiglu_limit)
        out.index_add_(0, rows, got.float() * wts[rows, col][:, None])
    return out


# =============================================================================
# loss and the acceptance proxy
# =============================================================================

def fastmtp_loss(logits, b, beta: float, ce_weight: float):
    """Exponentially decayed forward-KL + cross-entropy, per draft step.

    The KL is over the target's recorded top-32, renormalized over those 32: the rest of the
    129,280-row softmax is not recorded and not what acceptance turns on. Renormalizing rather than
    treating the tail as zero keeps the two distributions comparable.
    """
    import torch
    import torch.nn.functional as F

    B, D, V = logits.shape
    lp = F.log_softmax(logits.float(), dim=-1)
    tgt_lp = F.log_softmax(b["tgt_vals"], dim=-1)          # [B, D, K], renormalized over the top-32
    got = lp.gather(-1, b["tgt_ids"])                      # the drafter's log-probs on those ids
    kl = (tgt_lp.exp() * (tgt_lp - got)).sum(-1)           # [B, D]
    ce = F.nll_loss(lp.reshape(B * D, V), b["tgt_tok"].reshape(B * D), reduction="none").view(B, D)
    w = torch.tensor([beta ** d for d in range(D)], device=logits.device)
    per_step = kl + ce_weight * ce
    return (per_step * w).sum(-1).mean(), kl.mean(0).detach(), ce.mean(0).detach()


def acceptance_proxy(argmax, tgt_tok):
    """(per-step top-1 agreement [D], mean leading agreement + 1).

    Greedy verification accepts draft d only if every draft before it was accepted, so the quantity
    that matches `accept_len_mean` is the LEADING agreement, not the per-step mean. +1 for the
    token the verify step emits whatever happens, which is how the engine counts it.
    """
    import torch

    agree = (argmax == tgt_tok)
    lead = agree.to(torch.int32).cumprod(dim=1).sum(dim=1).float()
    return agree.float().mean(0), lead.mean() + 1.0


def evaluate(cfg, P, frozen, shards, picks, freqs_w, batch: int, device, markov_input: str):
    import torch

    tot_d, n, lead = None, 0, 0.0
    with torch.no_grad():
        for i in range(0, len(picks), batch):
            chunk = picks[i:i + batch]
            b = shards.batch(chunk, device)
            _, am = draft_forward(cfg, P, frozen, b, freqs_w, markov_input)
            per, ln = acceptance_proxy(am, b["tgt_tok"])
            tot_d = per * len(chunk) if tot_d is None else tot_d + per * len(chunk)
            lead += float(ln) * len(chunk)
            n += len(chunk)
    return (tot_d / max(n, 1)).tolist(), lead / max(n, 1)


# =============================================================================
# main
# =============================================================================

def save_head(path: str, P, model_dir: str, meta: dict, log=print):
    """Write the TRAINED tensors as bf16, under the checkpoint's own names.

    Only the trained ones: the engine's override is per tensor, so everything else keeps the
    checkpoint's own bytes and is never put through a dequantize/requantize round trip it did not
    need. It also makes the file small enough to keep beside a run's results."""
    import torch
    from safetensors.torch import save_file

    out = {name: t.detach().to(torch.bfloat16).cpu().contiguous() for name, t in P.items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(out, path)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump({"format": "dsv41-mtp-finetune", "tensors": sorted(out),
                   "trained": sorted(P), "model_dir": os.path.basename(model_dir.rstrip("/")),
                   **meta}, f, indent=1)
    log(f"wrote {len(out)} tensors ({sum(t.numel() for t in out.values()) * 2 / 1e9:.2f} GB bf16) to {path}")


def quantized_proxy(cfg, P, frozen, shards, picks, freqs_w, batch, device, markov_input, log=print):
    """The acceptance proxy the ENGINE will see: every trained 2-D weight put through the fp8 round
    trip that engine/model.py::_mtp_override does on load. A fine-tune that only survives in bf16 is
    not a fine-tune that ships, and this is the cheapest place to find that out."""
    import torch
    import v41_ref as R

    if R.quantize_to_fp8 is None:
        log("fp8 quantizer unavailable (no Triton); skipping the quantized proxy")
        return None
    saved = {}
    with torch.no_grad():
        for name, t in P.items():
            if t.size(-1) % 32 or t.dim() not in (2, 3) or not name.endswith(".weight"):
                continue   # norms, biases and the hc coefficients are served as they are
            saved[name] = t.detach().clone()
            flat = t.detach().to(torch.bfloat16).reshape(-1, t.size(-1))
            t.copy_(R.quantize_to_fp8(flat).dequant().float().view_as(t))
        res = evaluate(cfg, P, frozen, shards, picks, freqs_w, batch, device, markov_input)
        for name, t in saved.items():
            P[name].copy_(t)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["data/draft"], help="recorder directories")
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "./models/DeepSeek-V4.1-Flash"))
    ap.add_argument("--out", default="weights/mtp_finetuned.safetensors")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5, help="AdamW peak LR (small: this is an adaptation)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.6, help="per-step loss decay (FastMTP)")
    ap.add_argument("--ce-weight", type=float, default=1.0, help="weight of the cross-entropy term next to the KL")
    ap.add_argument("--draft", type=int, default=None,
                    help="draft length D (default: the engine's DSV41_BLOCK, normally 5)")
    ap.add_argument("--window", type=int, default=None, help="drafter attention window (default: the config's 128)")
    ap.add_argument("--train", default=",".join(DEFAULT_GROUPS),
                    help=f"comma-separated groups to train, of {', '.join(sorted(GROUPS))}")
    ap.add_argument("--markov-input", choices=("self", "target"), default="self",
                    help="what the Markov chain is fed at step d: the drafter's own argmax (self, "
                         "what serving does) or the target's token (target)")
    ap.add_argument("--held-out", type=float, default=0.05, help="fraction of SHARDS held out")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plan", action="store_true", help="print sizes and the memory budget, then stop")
    ap.add_argument("--tokens", type=int, default=300000, help="data size the plan quotes")
    a = ap.parse_args()

    groups = [g.strip() for g in a.train.split(",") if g.strip()]
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        ap.error(f"--train: unknown group(s) {unknown}; have {', '.join(sorted(GROUPS))}")

    if a.plan:
        p = plan(a.model_dir, groups, DR.record_bytes(), a.tokens, a.batch)
        print_plan(p, groups, a.tokens)
        return 0

    import torch
    import v41_ref as R

    def log(*m):
        print(time.strftime("%H:%M:%S"), "[train_mtp]", *m, flush=True)

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    # v41_ref's GEMM helpers exist for the engine's chunk invariance and its fp8 activation
    # emulation; neither belongs in a training graph. MM_TILE=0 is the plain F.linear, and the
    # activation fake-quant is replaced by a bf16 cast -- which is exactly what the engine itself
    # does when `act_quant` is off, i.e. always, so this is the serving path's own arithmetic.
    R.MM_TILE = 0
    R.act_qdq_fp8 = lambda x, block=32: x.to(torch.bfloat16)
    cfg = R.Args.from_json(os.path.join(a.model_dir, "inference", "config.json"))
    D = a.draft or int(os.environ.get("DSV41_BLOCK") or cfg.dspark_block_size)
    Wn = a.window or cfg.window_size
    hidden_dim = cfg.dim * len(cfg.dspark_target_layer_ids)

    p = plan(a.model_dir, groups, DR.record_bytes(hidden_dim), a.tokens, a.batch)
    print_plan(p, groups, a.tokens)
    if a.device.startswith("cuda") and torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 1e9
        log(f"free on the device: {free:.0f} GB, plan asks for {p['total_gb']:.0f} GB")
        if free < p["total_gb"]:
            raise SystemExit("not enough free memory. Stop the server (./stop.sh) and try again: "
                             "this trainer and the engine cannot share the box.")

    shards = Shards(a.data, Wn, D, hidden_dim)
    if not len(shards):
        raise SystemExit(f"no usable samples in {a.data}. Record some first: "
                         f"DSV41_RECORD_DRAFT_DATA=<dir> ./start.sh, then tools/draft_data_gen.py")
    train, test = shards.split(a.held_out, a.seed)
    log(f"{len(shards.shards)} shards, {len(shards)} samples "
        f"(D={D}, window={Wn}): {len(train)} train / {len(test)} held out")
    if not test or not train:
        raise SystemExit("the split leaves one side empty: record more than one shard, or lower "
                         "--held-out. The split is by shard on purpose -- two samples three "
                         "positions apart share 125 of their 128 window rows.")

    P, frozen = load_head(a.model_dir, a.device, groups, cfg, log)
    # sized from the data, not from a guess: a recorded position can be anywhere in the context the
    # server was run at, and the table is 8 bytes a position
    max_pos = max(int(s["position"][-1]) for s in shards.shards) + D + 8
    freqs_w = R.precompute_freqs_cis(cfg.rope_head_dim, max_pos, 0, cfg.rope_theta, cfg.rope_factor,
                                     cfg.beta_fast, cfg.beta_slow, a.device)
    log(f"trainable tensors: {len(P)} ({sum(t.numel() for t in P.values()) / 1e6:.0f} M parameters)")

    ev = test[:a.eval_samples]
    before = evaluate(cfg, P, frozen, shards, ev, freqs_w, a.batch, a.device, a.markov_input)
    log(f"before: per-step top-1 {[round(x, 3) for x in before[0]]}  accept proxy {before[1]:.2f}")

    opt = torch.optim.AdamW(list(P.values()), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))
    rng = random.Random(a.seed)
    t0 = time.time()
    hist = []
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:      # linear warmup, then cosine to a tenth of the peak
            if step <= a.warmup:
                g["lr"] = a.lr * step / max(a.warmup, 1)
            else:
                t = (step - a.warmup) / max(a.steps - a.warmup, 1)
                g["lr"] = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))
        b = shards.batch([train[rng.randrange(len(train))] for _ in range(a.batch)], a.device)
        logits, am = draft_forward(cfg, P, frozen, b, freqs_w, a.markov_input)
        loss, kl, ce = fastmtp_loss(logits, b, a.beta, a.ce_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.grad_clip:
            torch.nn.utils.clip_grad_norm_(list(P.values()), a.grad_clip)
        opt.step()
        if step % 25 == 0 or step == 1:
            _, ln = acceptance_proxy(am, b["tgt_tok"])
            log(f"step {step:>5}/{a.steps}  loss {float(loss):.4f}  kl0 {float(kl[0]):.3f}  "
                f"ce0 {float(ce[0]):.3f}  train accept {float(ln):.2f}  "
                f"{(time.time() - t0) / step:.2f} s/step")
        if a.eval_every and step % a.eval_every == 0:
            cur = evaluate(cfg, P, frozen, shards, ev, freqs_w, a.batch, a.device, a.markov_input)
            hist.append({"step": step, "per_step": cur[0], "accept_proxy": cur[1]})
            log(f"eval @{step}: per-step top-1 {[round(x, 3) for x in cur[0]]}  accept proxy {cur[1]:.2f}")

    after = evaluate(cfg, P, frozen, shards, ev, freqs_w, a.batch, a.device, a.markov_input)
    log(f"after:  per-step top-1 {[round(x, 3) for x in after[0]]}  accept proxy {after[1]:.2f}")
    q = quantized_proxy(cfg, P, frozen, shards, ev, freqs_w, a.batch, a.device, a.markov_input, log)
    if q:
        log(f"after the fp8 round trip the engine does on load: per-step top-1 "
            f"{[round(x, 3) for x in q[0]]}  accept proxy {q[1]:.2f}")

    save_head(a.out, P, a.model_dir, {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": [os.path.abspath(d) for d in a.data], "shards": len(shards.shards),
        "samples": len(shards), "train_samples": len(train), "held_out_samples": len(test),
        "steps": a.steps, "batch": a.batch, "lr": a.lr, "beta": a.beta, "ce_weight": a.ce_weight,
        "draft": D, "window": Wn, "groups": groups, "markov_input": a.markov_input,
        "seed": a.seed,
        "accept_proxy_before": before[1], "accept_proxy_after": after[1],
        "accept_proxy_fp8": (q[1] if q else None),
        "per_step_before": before[0], "per_step_after": after[0],
        "per_step_fp8": (q[0] if q else None), "eval_history": hist,
        "trainable_params": sum(t.numel() for t in P.values()),
    }, log)
    log(f"serve it with: DSV41_MTP_WEIGHTS={a.out} ./start.sh   "
        f"then tools/verify_mtp.sh for the real acceptance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
