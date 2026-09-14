"""The FastMTP loop on a toy head: does the loss mean what it says, and does the gradient reach
every tensor the run claims to be training?

Run: python3 tools/test_train_mtp.py    (skips itself without torch; uses CUDA when there is one)

The real head is 631 M trainable parameters against 13.6 G frozen ones and cannot be built anywhere
but the box, so this builds a 32-dimensional one with the same SHAPE -- three blocks, hyper-
connections, a 6-expert top-3 MoE, a window of recorded hidden states, a rank-k Markov chain -- and
asks the three questions that are worth asking before a five-hour data run:

  * **Does anything train?** A tensor in the trainable set with no gradient is not an error, it is a
    parameter that silently never moves. Every one of them is checked.
  * **Does the loss go down?** Twenty AdamW steps on ONE batch have to overfit it. If they do not,
    the graph is disconnected somewhere between the recorded hidden state and the logits.
  * **Does the acceptance proxy mean acceptance?** Greedy verification accepts a LEADING prefix, so
    `[hit, miss, hit]` is one accepted token and not two. That is the difference between a proxy
    that predicts tok/s and one that flatters it.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))

try:
    import torch
except ImportError:  # noqa: BLE001
    print("skip: no torch on this machine (this test belongs on the box)")
    sys.exit(0)

import train_mtp as T  # noqa: E402
import v41_ref as R  # noqa: E402

R.MM_TILE = 0
R.act_qdq_fp8 = lambda x, block=32: x.to(torch.bfloat16)   # the engine's own setting with act_quant off

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}")

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


# =============================================================================
# a toy head with the real shapes
# =============================================================================

V, DIM, NH, HD, RD, HC, G, LR_ = 64, 32, 2, 16, 8, 2, 2, 4
E, INTER, QL, MRANK = 6, 12, 8, 5
HIDDEN = DIM * 3          # dim x len(dspark_target_layer_ids)
B, D, WIN = 3, 4, 6

cfg = R.Args(vocab_size=V, dim=DIM, n_heads=NH, head_dim=HD, rope_head_dim=RD, hc_mult=HC,
             o_groups=G, o_lora_rank=LR_, q_lora_rank=QL, moe_inter_dim=INTER, n_layers=3,
             dspark_noise_token_id=V - 1, dspark_block_size=D, window_size=WIN,
             hc_sinkhorn_iters=3)

torch.manual_seed(0)


def rnd(*shape, dtype=torch.bfloat16, scale=0.05):
    return (torch.randn(*shape, device=dev) * scale).to(dtype)


P, frozen = {}, {}
frozen["embed.weight"] = rnd(V, DIM)
frozen["head.weight"] = rnd(V, DIM)
P["mtp.0.main_proj.weight"] = rnd(DIM, HIDDEN, dtype=torch.float32)
P["mtp.0.main_norm.weight"] = torch.ones(DIM, device=dev).requires_grad_(True)
for k in range(3):
    p = f"mtp.{k}."
    P[p + "attn_norm.weight"] = torch.ones(DIM, device=dev).requires_grad_(True)
    P[p + "ffn_norm.weight"] = torch.ones(DIM, device=dev).requires_grad_(True)
    P[p + "attn.q_norm.weight"] = torch.ones(QL, device=dev).requires_grad_(True)
    P[p + "attn.kv_norm.weight"] = torch.ones(HD, device=dev).requires_grad_(True)
    P[p + "attn.wq_a.weight"] = rnd(QL, DIM, dtype=torch.float32)
    P[p + "attn.wq_b.weight"] = rnd(NH * HD, QL, dtype=torch.float32)
    P[p + "attn.wkv.weight"] = rnd(HD, DIM, dtype=torch.float32)
    P[p + "attn.wo_a.weight"] = rnd(G, LR_, NH * HD // G, dtype=torch.float32)
    P[p + "attn.wo_b.weight"] = rnd(DIM, G * LR_, dtype=torch.float32)
    P[p + "attn.attn_sink"] = torch.zeros(NH, device=dev).requires_grad_(True)
    for s in ("attn", "ffn"):
        P[p + f"hc_{s}_fn"] = rnd((2 + HC) * HC, HC * DIM, dtype=torch.float32)
        P[p + f"hc_{s}_base"] = torch.zeros((2 + HC) * HC, device=dev).requires_grad_(True)
        P[p + f"hc_{s}_scale"] = torch.ones(3, device=dev).requires_grad_(True)
    P[p + "ffn.gate.weight"] = rnd(E, DIM, dtype=torch.float32)
    P[p + "ffn.gate.bias"] = torch.zeros(E, device=dev).requires_grad_(True)
    P[p + "ffn.shared_experts.w1.weight"] = rnd(INTER, DIM, dtype=torch.float32)
    P[p + "ffn.shared_experts.w3.weight"] = rnd(INTER, DIM, dtype=torch.float32)
    P[p + "ffn.shared_experts.w2.weight"] = rnd(DIM, INTER, dtype=torch.float32)
    frozen[p + "experts"] = [rnd(E, INTER, DIM), rnd(E, DIM, INTER), rnd(E, INTER, DIM)]
P["mtp.2.norm.weight"] = torch.ones(DIM, device=dev).requires_grad_(True)
P["mtp.2.markov_head.embed.weight"] = rnd(V, MRANK, dtype=torch.float32)
P["mtp.2.markov_head.head.weight"] = rnd(V, MRANK, dtype=torch.float32)
for t in P.values():
    t.requires_grad_(True)

freqs = R.precompute_freqs_cis(RD, 4096, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast,
                               cfg.beta_slow, dev)

pos0 = torch.full((B,), 100, device=dev, dtype=torch.long)
batch = {
    "hidden": (torch.randn(B, WIN, HIDDEN, device=dev) * 0.5).to(torch.bfloat16),
    "win_pos": pos0[:, None] - torch.arange(WIN, 0, -1, device=dev)[None, :],
    "tok0": torch.randint(0, V - 1, (B,), device=dev),
    "pos0": pos0,
    "tgt_ids": torch.stack([torch.randperm(V, device=dev)[:8] for _ in range(B * D)]).view(B, D, 8),
    "tgt_vals": torch.randn(B, D, 8, device=dev).sort(dim=-1, descending=True).values,
    "tgt_tok": torch.randint(0, V - 1, (B, D), device=dev),
}
batch["tgt_ids"][:, :, 0] = batch["tgt_tok"]      # the settled token is the target's own argmax

# =============================================================================
# the forward
# =============================================================================

logits, am = T.draft_forward(cfg, P, frozen, batch, freqs, "self")
check("the draft produces one distribution per draft step", tuple(logits.shape) == (B, D, V),
      str(tuple(logits.shape)))
check("and one argmax per step", tuple(am.shape) == (B, D))
check("the logits are finite", bool(torch.isfinite(logits).all()))
check("the logits are fp32 whatever the weights are", logits.dtype == torch.float32)

l_t, _ = T.draft_forward(cfg, P, frozen, batch, freqs, "target")
check("the Markov chain's input changes the draft after step 1",
      torch.equal(l_t[:, 0], logits[:, 0]) and not torch.equal(l_t[:, 1:], logits[:, 1:]))

# the window is the drafter's whole view of the target: change it and the draft has to change
b2 = dict(batch)
b2["hidden"] = batch["hidden"] + 1.0
l2, _ = T.draft_forward(cfg, P, frozen, b2, freqs, "self")
check("a different target hidden state is a different draft", not torch.allclose(l2, logits))

# =============================================================================
# the loss
# =============================================================================

loss, kl, ce = T.fastmtp_loss(logits, batch, 0.6, 1.0)
check("the loss is a finite scalar", loss.dim() == 0 and bool(torch.isfinite(loss)))
check("the KL is non-negative at every step", bool((kl >= -1e-5).all()), str([round(float(x), 3) for x in kl]))
check("there is one KL and one CE per draft step", kl.numel() == D and ce.numel() == D)

import torch.nn.functional as F  # noqa: E402

ce_ref = F.cross_entropy(logits.reshape(B * D, V), batch["tgt_tok"].reshape(B * D), reduction="none")
check("the CE term is the cross-entropy on the token the target settled on",
      torch.allclose(ce, ce_ref.view(B, D).mean(0), atol=1e-4))

only_first = T.fastmtp_loss(logits, batch, 0.0, 1.0)[0]
check("beta = 0 weighs the first draft step and nothing else",
      abs(float(only_first) - float(kl[0] + ce[0])) < 1e-4,
      f"{float(only_first):.4f} vs {float(kl[0] + ce[0]):.4f}")

flat = T.fastmtp_loss(logits, batch, 1.0, 1.0)[0]
check("beta = 1 weighs every step the same and is the larger loss", float(flat) > float(loss),
      f"{float(flat):.3f} > {float(loss):.3f}")

# a drafter that already agrees with the target has (almost) no KL left to pay
perfect = torch.full((B, D, V), -30.0, device=dev)
perfect.scatter_(-1, batch["tgt_ids"], batch["tgt_vals"])
kl_perfect = T.fastmtp_loss(perfect, batch, 0.6, 0.0)[1]
check("a draft that reproduces the target's top-32 pays no KL",
      float(kl_perfect.abs().max()) < 0.02, f"max {float(kl_perfect.abs().max()):.4f}")

# =============================================================================
# the acceptance proxy
# =============================================================================

hits = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]], device=dev)
tgt = torch.zeros(3, 4, dtype=torch.long, device=dev)
# the drafter's argmax: the target's token where it hits, something else where it misses
per_step, lead_len = T.acceptance_proxy(torch.where(hits.bool(), tgt, tgt + 1), tgt)
check("per-step agreement is the fraction of samples that hit at that step",
      [round(float(x), 3) for x in per_step] == [round(2 / 3, 3), 1.0, round(2 / 3, 3), 1.0],
      str([round(float(x), 3) for x in per_step]))
check("the accept length is the LEADING agreement plus the always-emitted token",
      abs(float(lead_len) - ((2 + 0 + 4) / 3 + 1)) < 1e-5, f"{float(lead_len):.3f}")
check("a draft that agrees everywhere accepts the whole block",
      abs(float(T.acceptance_proxy(tgt, tgt)[1]) - (4 + 1)) < 1e-5)

# =============================================================================
# does it train?
# =============================================================================

_, lead_before = T.acceptance_proxy(am, batch["tgt_tok"])
loss.backward()
# `ffn.gate.bias` is deliberately not here: it steers a top-k and a top-k has no gradient
# (train_mtp.NO_GRAD_NAMES). Anything else without one is a parameter that would silently not move.
no_grad = [n for n, t in P.items()
           if n.rsplit("mtp.0.", 1)[-1] not in T.NO_GRAD_NAMES
           and (t.grad is None or not torch.isfinite(t.grad).all() or float(t.grad.abs().sum()) == 0.0)]
check("every trainable tensor gets a finite, non-zero gradient",
      not no_grad, f"no gradient: {', '.join(no_grad[:6])}" if no_grad else "")

opt = torch.optim.AdamW(list(P.values()), lr=3e-3)
first, l = None, None
for step in range(20):
    lg, _ = T.draft_forward(cfg, P, frozen, batch, freqs, "self")
    l, _, _ = T.fastmtp_loss(lg, batch, 0.6, 1.0)
    opt.zero_grad(set_to_none=True)
    l.backward()
    opt.step()
    if first is None:
        first = float(l)
check("twenty steps on one batch overfit it", float(l) < first * 0.9, f"{first:.3f} -> {float(l):.3f}")

lg, am2 = T.draft_forward(cfg, P, frozen, batch, freqs, "self")
_, lead_after = T.acceptance_proxy(am2, batch["tgt_tok"])
check("and the acceptance proxy follows the loss down",
      float(lead_after) >= float(lead_before),
      f"accept proxy {float(lead_before):.2f} -> {float(lead_after):.2f}")

print()
print(f"{len(fails)} failed: {', '.join(fails)}" if fails else "all checks passed")
sys.exit(1 if fails else 0)
