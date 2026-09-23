"""fp4_expert_cpu.py -- the RAM-resident expert tier: FP4 experts computed on the CPU.

Where this fits
---------------
`tools/fp4_moe.py` runs the routed experts on one GPU out of an arena that is filled from NVMe
on demand. On a 128 GB box that is the only design that fits, and the measurements in the
repo's README are the price of it: 0.92 GB read per generated token, and the SSD is under 8% of
the decode step being compute.

On a 4x V100 host with 256 GB of DDR4 the shape of the problem is different. The routed experts
are 288.8 GB at FP4 (15,360 slots x 18,800,640 B); 128 GB of VRAM plus 256 GB of RAM holds all
of them, so nothing has to be streamed and nothing has to be re-quantised. What is missing is a
way to *compute* the experts that live in RAM, because sglang's offload copies them back to the
GPU first (`OffloaderV1.forward` does `v.to(device)` and computes there), and doing that per
token would put 0.9-4 GB over PCIe 3.0 every step.

So this module gives the RAM tier its own compute: the same bytes the GPU arena holds, decoded
and multiplied by the CPU that is sitting next to the RAM they live in. The activations that
cross the bus are 41 KB per token; the weights never cross it at all.

What it does not do
-------------------
It is not a replacement for the GPU path and it does not try to be: it is the tier that turns
"does not fit" into "there is no TP8 host here", and the honest reading of its throughput is in
`tools/bench_fp4_cpu.py`, measured rather than claimed.

Pieces
------
  CpuKernels        the ctypes handle on tools/_fp4_cpu.so, built by tools/build_fp4_cpu.sh
  fp4_gemv          y = (decode(w) * 2**(s-127)) @ x, fused, one pass over the packed bytes
  ExpertArenaCPU    RAM slots with the same interface the store already fills (engine/experts.py
                    ::ExpertStore calls `arena.slots` and `arena.load_slot`), so the RAM tier is
                    a drop-in for the NVMe-streamed GPU arena
  expert_forward    one expert: gate/up, the checkpoint's clamped SwiGLU, down
  moe_forward_cpu   the routed MoE for a batch, grouped by slot, in the same summation order as
                    `tools/fp4_moe.moe_forward_reference`

Numerics
--------
The reference (`inference/model.py::Expert.forward`, mirrored by `moe_forward_reference`) does
bf16 GEMMs with an fp32 epilogue. On a V100 bf16 has no hardware path at all -- Triton would
emulate it -- so this tier accumulates in fp32 and returns fp16, which is the choice the rest of
the V100 stack makes too (sglang's `SGLANG_SM70_FORCE_FP16`, and 1Cat-vLLM's `dtype="half"`,
"V100 has no BF16 tensor cores"). fp32 accumulate is not a downgrade: it is what the reference
epilogue already does.
"""

from __future__ import annotations

import ctypes
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fp4_decode as D  # noqa: E402

# config.json, same constants as tools/fp4_moe.py (asserted against it in the tests when that
# module is importable, so the two cannot drift).
DIM = 5120
INTER = 2304
GROUP = D.GROUP
KB1, SG1 = DIM // 2, DIM // GROUP  # packed / scale columns of w1 and w3
KB2, SG2 = INTER // 2, INTER // GROUP  # ... of w2

SO_NAME = "_fp4_cpu.so"
SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), SO_NAME)


class CpuKernels:
    """ctypes handle on the compiled kernels. Missing .so is a supported state: `available()`
    returns False and the callers fall back to the torch path, so the tests and the tier planner
    run on a box that has never built anything."""

    def __init__(self, path: str = SO_PATH):
        self.path = path
        self.lib = None
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            lib.fp4_gemv_ref.restype = None
            lib.fp4_gemv_ref.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_long, ctypes.c_long]
            lib.fp4_gemv.restype = None
            lib.fp4_gemv.argtypes = lib.fp4_gemv_ref.argtypes
            lib.fp4_swiglu.restype = None
            lib.fp4_swiglu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_long, ctypes.c_float, ctypes.c_float]
            lib.fp4_cpu_set_threads.restype = None
            lib.fp4_cpu_set_threads.argtypes = [ctypes.c_int]
            lib.fp4_cpu_max_threads.restype = ctypes.c_int
            lib.fp4_cpu_max_threads.argtypes = []
            self.lib = lib

    def available(self) -> bool:
        return self.lib is not None

    @staticmethod
    def build_hint() -> str:
        return "bash tools/build_fp4_cpu.sh"

    def set_threads(self, n: int) -> int:
        """Set the OpenMP thread count the kernels will use, and return what libgomp reports.

        Callers should treat this as mandatory rather than optional. PyTorch calls
        `omp_set_num_threads(1)` during import (its intra-op pool is its own, and it sets the
        OpenMP count to 1 so the two do not fight), and that is process-global: an engine that
        has imported torch gets a single-threaded expert tier unless it overrides the count.
        Measured here, that is 6.0 ms per expert instead of 0.75 ms.
        """
        if self.lib is None:
            return 0
        self.lib.fp4_cpu_set_threads(int(n))
        return self.max_threads()

    def max_threads(self) -> int:
        return self.lib.fp4_cpu_max_threads() if self.lib is not None else 0

    def gemv(self, w: torch.Tensor, s: torch.Tensor, x: torch.Tensor, out: torch.Tensor,
             ref: bool = False) -> None:
        n, k = out.numel(), x.numel()
        fn = self.lib.fp4_gemv_ref if ref else self.lib.fp4_gemv
        fn(w.data_ptr(), s.data_ptr(), x.data_ptr(), out.data_ptr(), n, k)

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor, out: torch.Tensor,
               limit: float, weight: float) -> None:
        self.lib.fp4_swiglu(gate.data_ptr(), up.data_ptr(), out.data_ptr(),
                            out.numel(), float(limit), float(weight))


_KERNELS: CpuKernels | None = None


def kernels(threads: int | None = None) -> CpuKernels:
    """The shared kernel handle, with the thread count set explicitly.

    `threads=None` means "all of them", and it is applied on first use rather than left to the
    process default, because the process default is 1 whenever torch was imported first (see
    `CpuKernels.set_threads`). Override with DSV41_CPU_THREADS to leave headroom for the rest of
    the engine, or pass `threads` explicitly in a benchmark.
    """
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = CpuKernels()
        n = threads if threads is not None else int(os.environ.get("DSV41_CPU_THREADS", 0) or 0)
        if _KERNELS.available():
            _KERNELS.set_threads(n or (os.cpu_count() or 1))
    elif threads is not None and _KERNELS.available():
        _KERNELS.set_threads(threads)
    return _KERNELS


# --------------------------------------------------------------------------- torch fallback
def fp4_gemv_torch(w: torch.Tensor, s: torch.Tensor, x: torch.Tensor,
                   group: int = GROUP) -> torch.Tensor:
    """The same contract in torch: dequantise exactly (fp32, no overflow) and multiply.

    Kept as the correctness oracle for the C path and as the fallback when the .so is missing.
    It materialises the weights, so it is several times the memory traffic of the fused kernel
    -- which is the whole argument for having the kernel.
    """
    wf = D.dequant_fp4(w, s, group=group, dtype=torch.float32)  # [N, K]
    return wf @ x


# --------------------------------------------------------------------------- arena
class ExpertArenaCPU:
    """RAM-resident slots of packed FP4 expert weights, shaped like
    `tools/fp4_moe.ExpertArena` so `engine/experts.py::ExpertStore` can fill it unchanged.

    `pin_memory=True` asks torch for page-locked host memory. It costs nothing to allocate and
    makes the same buffers valid as a copy source for a GPU arena later, but it also cannot be
    swapped, which is worth knowing before pinning 190 GB on a 256 GB box -- `budget.py`'s floor
    arithmetic has to be paid against unpinned memory.
    """

    def __init__(self, slots: int, pin_memory: bool = False):
        self.slots = slots
        self.pin_memory = pin_memory
        kw = dict(dtype=torch.uint8, pin_memory=pin_memory)
        self.w1 = torch.empty((slots, INTER, KB1), **kw)
        self.s1 = torch.empty((slots, INTER, SG1), **kw)
        self.w3 = torch.empty((slots, INTER, KB1), **kw)
        self.s3 = torch.empty((slots, INTER, SG1), **kw)
        self.w2 = torch.empty((slots, DIM, KB2), **kw)
        self.s2 = torch.empty((slots, DIM, SG2), **kw)

    @property
    def bytes_per_slot(self) -> int:
        return sum(t[0].numel() for t in (self.w1, self.s1, self.w3, self.s3, self.w2, self.s2))

    @property
    def res_bytes(self) -> int:
        return self.slots * self.bytes_per_slot

    @staticmethod
    def _u8(t: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        t = t.view(torch.uint8)
        if tuple(t.shape) != shape:
            raise ValueError(f"expected shape {shape}, got {tuple(t.shape)}")
        return t.contiguous()

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False, **kw) -> None:
        """Same signature as the CUDA arena's loader, including the argument order the store uses.
        `non_blocking` and `**kw` are accepted and ignored: there is no stream and no codebook
        packer on this side, and refusing them would break the drop-in."""
        del non_blocking, kw
        self.w1[slot].copy_(self._u8(w1, (INTER, KB1)))
        self.s1[slot].copy_(self._u8(s1, (INTER, SG1)))
        self.w3[slot].copy_(self._u8(w3, (INTER, KB1)))
        self.s3[slot].copy_(self._u8(s3, (INTER, SG1)))
        self.w2[slot].copy_(self._u8(w2, (DIM, KB2)))
        self.s2[slot].copy_(self._u8(s2, (DIM, SG2)))

    def dequant_slot(self, slot: int):
        """(w1, w2, w3) as fp32 [N, K] -- for tests and for the M>1 path."""
        return (D.dequant_fp4(self.w1[slot], self.s1[slot], dtype=torch.float32),
                D.dequant_fp4(self.w2[slot], self.s2[slot], dtype=torch.float32),
                D.dequant_fp4(self.w3[slot], self.s3[slot], dtype=torch.float32))


# --------------------------------------------------------------------------- one expert
def expert_forward(x: torch.Tensor, cache, w1, s1, w2, s2, w3, s3,
                   swiglu_limit: float = 10.0, weights: torch.Tensor | None = None,
                   out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """One expert over a batch of tokens. x fp32 [M, DIM] -> [M, DIM].

    M == 1 is the decode case and takes the fused C GEMV: one pass over 18.8 MB of packed
    weights, no materialised copy. M > 1 materialises the weights once and leans on BLAS, because
    a GEMV per row would re-read every weight byte M times and prefill chunks are wide enough to
    pay for the copy. Both paths are the same arithmetic; `cache` is the CpuKernels handle or
    None to force the torch path.
    """
    if x.dim() != 2 or x.shape[1] != DIM:
        raise ValueError(f"expected x [M, {DIM}], got {tuple(x.shape)}")
    m = x.shape[0]
    if weights is None:
        weights = torch.ones(m, dtype=torch.float32)
    # Accept [M], [M, 1] and a bare scalar and normalise to [M]. Passing [M, 1] straight into
    # `weights[:, None]` below would make it [M, 1, 1] and broadcast the SwiGLU output into
    # [M, M, INTER] -- a wrong answer with a plausible shape, which is the worst kind.
    weights = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
    if weights.numel() != m:
        raise ValueError(f"expected {m} routing weights, got {weights.numel()}")
    x32 = x.to(torch.float32).contiguous()

    if m == 1 and cache is not None and cache.available():
        gate = torch.empty(INTER, dtype=torch.float32)
        up = torch.empty(INTER, dtype=torch.float32)
        h = torch.empty(INTER, dtype=torch.float32)
        cache.gemv(w1, s1, x32[0], gate)
        cache.gemv(w3, s3, x32[0], up)
        cache.swiglu(gate, up, h, swiglu_limit, float(weights[0]))
        y = torch.empty(DIM, dtype=torch.float32)
        cache.gemv(w2, s2, h, y)
        return y.unsqueeze(0).to(out_dtype)

    w1f = D.dequant_fp4(w1, s1, dtype=torch.float32)
    w3f = D.dequant_fp4(w3, s3, dtype=torch.float32)
    w2f = D.dequant_fp4(w2, s2, dtype=torch.float32)
    gate = x32 @ w1f.T
    up = x32 @ w3f.T
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    h = torch.nn.functional.silu(gate) * up * weights[:, None]
    return (h @ w2f.T).to(out_dtype)


def expert_forward_slot(x: torch.Tensor, slot: int, arena: ExpertArenaCPU,
                        swiglu_limit: float = 10.0, weights: torch.Tensor | None = None,
                        out_dtype: torch.dtype = torch.float16, cache: CpuKernels | None = None):
    """`expert_forward` straight out of an arena slot: no copy, the kernel reads the slot's
    buffers where they are. This is the call the RAM tier makes."""
    if cache is None:
        cache = kernels()
    return expert_forward(x, cache, arena.w1[slot], arena.s1[slot], arena.w2[slot], arena.s2[slot],
                          arena.w3[slot], arena.s3[slot], swiglu_limit, weights, out_dtype)


# --------------------------------------------------------------------------- the batch
def moe_forward_cpu(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor,
                    arena: ExpertArenaCPU, swiglu_limit: float = 10.0,
                    out_dtype: torch.dtype = torch.float16,
                    cache: CpuKernels | None = None) -> torch.Tensor:
    """x [T, DIM], slots int32 [T, K], weights [T, K] -> [T, DIM].

    Same routing semantics and summation order as `moe_forward_reference`: for each distinct
    slot, its (token, k) pairs are gathered, one expert call, then scattered back and summed.
    The reference is the contract the GPU path is also held to, so keeping the order identical
    is what makes a CPU-tier token comparable with a GPU-tier token at the same context.

    Duplicate routes are folded before the expert call. Top-k can hand the same expert two of a
    token's six slots; the reference computes the expert twice and adds, but `h` is linear in the
    routing weight (`silu(gate) * up * w`), so summing the two weights and calling once is the
    same number for a fraction of the work. Calling once with only the *first* weight -- which is
    what the first version of this function did -- silently drops the other route, so
    `tools/test_fp4_gemv_cpu.py` pins the behaviour rather than the implementation.
    """
    if cache is None:
        cache = kernels()
    t, k = slots.shape
    y = torch.zeros((t, DIM), dtype=torch.float32)
    for s in torch.unique(slots).tolist():
        t_idx, k_idx = torch.nonzero(slots == s, as_tuple=True)
        wsum = torch.zeros(t, dtype=torch.float32)
        wsum.index_add_(0, t_idx, weights[t_idx, k_idx].to(torch.float32))
        uniq = torch.unique(t_idx)
        xs = x[uniq].to(torch.float32)
        if xs.shape[0] == 1:
            ys = expert_forward_slot(xs, int(s), arena, swiglu_limit, wsum[uniq],
                                     out_dtype=torch.float32, cache=cache)
        else:
            ys = expert_forward(xs, cache, arena.w1[s], arena.s1[s], arena.w2[s], arena.s2[s],
                                arena.w3[s], arena.s3[s], swiglu_limit, wsum[uniq],
                                out_dtype=torch.float32)
        y.index_add_(0, uniq, ys.to(torch.float32))
    return y.to(out_dtype)
