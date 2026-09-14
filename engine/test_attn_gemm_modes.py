"""The three math types of prefill's attention GEMMs, against each other, on the real shapes.

Run:  python3 engine/test_attn_gemm_modes.py

It needs a GPU and torch and nothing else -- no checkpoint, no arena, no server -- and it SKIPS
itself (exit 0) where there is no CUDA device, so it is safe to run anywhere. The torch-free half,
which pins the default and the prefill-only guard, is `tools/test_prefill_attn_gemm.py`.

What it measures. `Model._softmax_attn` is reproduced here at the shapes the engine runs it at:
`ATTN_TILE = 64` query rows per tile, `n_heads = 64`, `head_dim = 512`, and a KV width of
`window_size + index_topk = 128 + 512 = 640` (plus the 128-wide variant the two layers with no
compressed stream take). That is one of the 672 score products and one of the 672 PV products a
2,048-token chunk issues per kernel; the profile behind docs/gemm-dispatch.md puts the pair at
855 ms of that chunk, in `cutlass_80_simt_sgemm_128x32_8x5_{tn,nn}_align1` -- SIMT, i.e. not on
the tensor cores at all, which is the whole reason this switch exists.

The bounds. fp32 is the reference. TF32 rounds the GEMM inputs to an 11-bit significand and still
accumulates in fp32, so ~2^-11 = 4.9e-4 per operand and of the order of 1e-3 on the product; bf16
has an 8-bit significand, ~2^-8 = 3.9e-3. The tolerances below are those figures with the margin
a 512- or 640-long reduction earns. They are deliberately checked from BOTH sides: a mode that
came back bit-identical to fp32 would mean the flag never took (TF32 silently ignored, or the
bf16 cast folded away), and that is a failure of the switch just as much as a mode that is wildly
wrong -- it would make a speed run on the box measure fp32 twice and call it a win.

It also prints the median time of each mode on those shapes. That is an indication, not the
measurement: the measurement is `tools/verify_prefill_hc.sh` on the served engine, because a
kernel timed in isolation on a quiet box is not a kernel timed inside a prefill chunk.
"""
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:
    print("SKIP engine/test_attn_gemm_modes.py: no torch")
    sys.exit(0)

if not torch.cuda.is_available():
    print("SKIP engine/test_attn_gemm_modes.py: no CUDA device")
    sys.exit(0)

from engine import prefill_attn_gemm as AG  # noqa: E402

DEV = "cuda"
fails = []

# The engine's own settings, copied rather than imported: engine/model.py imports the checkpoint
# loader chain, and this test has no checkpoint.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
ATTN_TILE = 64
N_HEADS = 64
HEAD_DIM = 512


def softmax_attn_tile(q, kv, mask, sink, mode):
    """One tile of Model._softmax_attn, verbatim apart from taking the mode as an argument."""
    scale = HEAD_DIM ** -0.5
    dt = torch.bfloat16 if mode == "bf16" else torch.float32
    with AG.math_scope(mode):
        qt, kvt = q.to(dt), kv.to(dt)
        scores = torch.einsum("thd,tnd->thn", qt, kvt).float() * scale
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
        p = torch.exp(scores - mx)
        denom = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - mx)
        out = torch.einsum("thn,tnd->thd", (p / denom).to(dt), kvt)
    return out.to(torch.bfloat16)


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / max(float(b.norm()), 1e-30))


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {name}{'   ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def timed(fn, repeats=20):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def case(n_kv, label, lo_tf32, hi_tf32, lo_bf16, hi_bf16):
    torch.manual_seed(0)
    # bf16 in, exactly as the engine has them: q comes out of the fp4 dense kernel and the KV out
    # of the window ring and the compressed cache, both bf16.
    q = (torch.randn(ATTN_TILE, N_HEADS, HEAD_DIM, device=DEV) * 0.5).to(torch.bfloat16)
    kv = (torch.randn(ATTN_TILE, n_kv, HEAD_DIM, device=DEV) * 0.5).to(torch.bfloat16)
    sink = torch.randn(N_HEADS, device=DEV)
    mask = torch.ones(ATTN_TILE, n_kv, dtype=torch.bool, device=DEV)
    # a causal-ish edge and one fully masked row, the two shapes the engine's padding produces
    mask[:, n_kv // 2:] = torch.arange(ATTN_TILE, device=DEV)[:, None] % 3 > 0
    mask[7] = False

    print(f"\n== {label}: q [{ATTN_TILE}, {N_HEADS}, {HEAD_DIM}] bf16, "
          f"kv [{ATTN_TILE}, {n_kv}, {HEAD_DIM}] bf16 ==")
    ref = softmax_attn_tile(q, kv, mask, sink, "fp32")
    out = {m: softmax_attn_tile(q, kv, mask, sink, m) for m in ("tf32", "bf16")}

    check("fp32 is deterministic across calls",
          torch.equal(ref, softmax_attn_tile(q, kv, mask, sink, "fp32")))
    check("the fully masked row is zero, not NaN in any mode",
          all(bool((o[7] == 0).all()) for o in [ref, *out.values()]))
    check("no mode produces a NaN or an inf",
          all(bool(torch.isfinite(o.float()).all()) for o in [ref, *out.values()]))

    for m, (lo, hi) in (("tf32", (lo_tf32, hi_tf32)), ("bf16", (lo_bf16, hi_bf16))):
        r = rel(out[m], ref)
        check(f"{m} vs fp32 is inside [{lo:.0e}, {hi:.0e}]", lo <= r <= hi, f"rel {r:.3e}")

    check("tf32 is closer to fp32 than bf16 is",
          rel(out["tf32"], ref) < rel(out["bf16"], ref))

    ms = {m: timed(lambda m=m: softmax_attn_tile(q, kv, mask, sink, m))
          for m in ("fp32", "tf32", "bf16")}
    print("  time per tile (median of 20, one tile of the 32 a 2,048-token chunk issues per layer):")
    for m, t in ms.items():
        print(f"    {m:5s} {t:7.3f} ms   x{ms['fp32'] / t:5.2f} against fp32")
    print(f"  extrapolated to one 2,048-token chunk (32 tiles x 21 encoder layers = 672 of each "
          f"product): fp32 {ms['fp32'] * 672:.0f} ms, tf32 {ms['tf32'] * 672:.0f} ms, "
          f"bf16 {ms['bf16'] * 672:.0f} ms")


print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
print(f"allow_tf32 on entry: {torch.backends.cuda.matmul.allow_tf32}  "
      f"precision: {torch.get_float32_matmul_precision()}")

# The lower bounds say the mode actually took effect; the upper bounds are the error budget.
case(640, "the 19 layers with a compressed stream (window 128 + index_topk 512)",
     1e-5, 4e-3, 5e-4, 2e-2)
case(128, "the 2 layers with the sliding window only",
     1e-5, 4e-3, 5e-4, 2e-2)

print("\n== the scope leaves nothing armed ==")
check("allow_tf32 is back off after every case", torch.backends.cuda.matmul.allow_tf32 is False)
check("the precision string is back to its default",
      torch.get_float32_matmul_precision() == "highest")

# What a kernel name says. The point of the tf32 mode is that the GEMM stops being SIMT; on a box
# where the profiler is available this is the cheap version of that check.
try:
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(0)
    q = (torch.randn(ATTN_TILE, N_HEADS, HEAD_DIM, device=DEV)).to(torch.bfloat16)
    kv = (torch.randn(ATTN_TILE, 640, HEAD_DIM, device=DEV)).to(torch.bfloat16)
    sink = torch.randn(N_HEADS, device=DEV)
    mask = torch.ones(ATTN_TILE, 640, dtype=torch.bool, device=DEV)
    print("\n== kernel names per mode ==")
    for m in ("fp32", "tf32", "bf16"):
        softmax_attn_tile(q, kv, mask, sink, m)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            softmax_attn_tile(q, kv, mask, sink, m)
            torch.cuda.synchronize()
        names = [str(e.key) for e in prof.key_averages()
                 if str(getattr(e, "device_type", "")).endswith("CUDA")
                 and any(s in str(e.key).lower() for s in ("gemm", "cutlass", "tensorop", "simt"))]
        print(f"  {m:5s} {names if names else '(no gemm-shaped kernel name in the profile)'}")
        if m == "fp32":
            check("fp32 still lands on a SIMT sgemm",
                  any("simt" in n.lower() or "sgemm" in n.lower() for n in names) or not names)
except Exception as e:  # the profiler is optional; the numbers above are the test
    print(f"\n(kernel-name section skipped: {e})")

print(f"\n{len(fails)} failed" if fails else "\nall ok")
sys.exit(1 if fails else 0)
