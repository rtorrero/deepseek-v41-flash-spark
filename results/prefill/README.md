# Prefill records

What `tools/verify_prefill_hc.sh` leaves behind. Empty until the script has been run on the box —
this directory ships with the question, not with an answer.

| file | what it is |
|---|---|
| `RECORDS.md` | one section per run: the three modes' prefill tok/s and TTFT on one identical prompt, what the kernel audits said, and which gate (if any) was taken |
| `prompt.txt` | the prompt the run used, written once and reused by all three modes, so "identical prompt" is a property of the bytes |
| `gemm_dispatch-fp32.json`, `gemm_dispatch-tf32.json` | `tools/audit_gemm_dispatch.py --phase prefill --json`, server stopped, one mode each |
| `GATE-frontend-<mode>.md` | `tools/gate_profile.py --profile Frontend --thinking on` under a non-default math type |

## The question these records answer

A 2,048-token prefill chunk's largest single kernel cost is not a quantised kernel. It is the pair
of fp32 GEMMs inside the attention softmax — 672 calls each, 855 ms together, against 893 ms for
both FP4 MoE kernels — and fp32 on this box means the SIMT pipeline rather than the tensor cores.
`DSV41_PREFILL_ATTN_GEMM=fp32|tf32|bf16` is the switch; the arithmetic, the shapes and the error
bounds are in [`../../docs/gemm-dispatch.md`](../../docs/gemm-dispatch.md).

## Reading a record

The speed rows are the cheap half and they settle nothing on their own. Two things decide whether
a non-default mode is worth anything:

* **Did the flag take?** The `fp32` audit must show the two 672-call
  `cutlass_*_simt_sgemm_*_{tn,nn}_align1` rows and the `tf32` audit must not. If both tables look
  the same, cuBLAS never saw the math type and the run measured fp32 three times.
* **What does the model write?** `tf32` is ~1e-3 on the product and `bf16` ~4e-3, against a design
  whose chunk-invariance argument is built on 1e-7. The failure mode is a flipped router top-k
  deep in a long prompt, not a wrong number, so the gate output compared against
  `../keepsets/frontend/GATE.md` is the only evidence that accepts either mode. A record whose
  gate section says none was run accepts nothing.

The gate output lives here and not in `../keepsets/frontend/GATE.md`: those files are the record of
the shipped configuration, and an experimental math type has no business in them.
