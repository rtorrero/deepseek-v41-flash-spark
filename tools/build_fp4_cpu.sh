#!/usr/bin/env bash
# build_fp4_cpu.sh -- build the CPU expert-tier GEMV (tools/fp4_cpu.c) into tools/_fp4_cpu.so
#
# The AVX2/FMA path is selected by the flags and not by the source: if the host does not have
# them, the same file compiles to the scalar reference and the tier still works, just slowly.
# That matters because the V100 hosts this is aimed at are old enough that AVX-512 is the
# exception, and because a build that silently produces the slow path is worse than one that
# says which path it produced.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cc="${CC:-gcc}"
src="$here/fp4_cpu.c"
out="${1:-$here/_fp4_cpu.so}"

flags=(-O3 -fPIC -shared -std=c11 -Wall -Wextra)

if grep -qm1 -E '(^| )(avx2)( |$)' /proc/cpuinfo 2>/dev/null && grep -qm1 -E '(^| )(fma)( |$)' /proc/cpuinfo 2>/dev/null; then
  flags+=(-mavx2 -mfma)
  path="avx2+fma"
else
  path="scalar"
fi

# Threads are what make the RAM tier viable -- one core cannot pull 190 GB of experts out of
# DDR4 on its own -- so OpenMP is not optional when the compiler has it. A compiler without it
# still builds, and the banner says which path was produced rather than leaving it to be
# inferred from a benchmark.
if echo 'int main(){return 0;}' | "$cc" -x c - -fopenmp -o /dev/null 2>/dev/null; then
  flags+=(-fopenmp)
  path="$path+openmp"
fi

echo "[fp4_cpu] $cc ${flags[*]} -> $out  (path: $path)"
"$cc" "${flags[@]}" "$src" -o "$out" -lm
echo "[fp4_cpu] ok: $(ls -l "$out" | awk '{print $5}') bytes"
