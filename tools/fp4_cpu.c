/* fp4_cpu.c -- fused FP4 E2M1 + UE8M0 GEMV for the CPU expert tier.
 *
 * Why a C kernel and not torch
 * ----------------------------
 * The bytes of one expert are 18.8 MB. Any scheme that materialises the weights into fp32 or
 * fp16 before the multiply touches far more than that: 118 MB for fp32, plus a second read by
 * the GEMM. On a box whose whole point is to hold 190 GB of experts in DDR4, that extra traffic
 * is the difference between "the RAM tier is free capacity" and "the RAM tier halves
 * throughput". So the decode happens in the multiply, once per stored byte:
 *
 *     y[n] = sum over K groups of  scale[n,g] * sum_{k in g} x[k] * value(code(w[n,k]))
 *
 * The scale goes on the group partial, not on the weight -- the checkpoint's scale byte is a
 * free exponent and 6.0 * 2**14 does not fit in fp16. That is also what the Triton kernels do
 * (`_chunk_dot` returns `p * _ue8m0(scale)`), so the two paths agree by construction.
 *
 * How the decode is done without a table load
 * -------------------------------------------
 * AVX2 `vpermps` is an 8-entry float lookup in one instruction, and the E2M1 magnitude grid has
 * exactly 8 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}: the sign is a separate bit that XORs into
 * the float's sign bit. So per 8 packed bytes:
 *
 *     bytes  = cvtepu8_epi32(loadl(8 bytes))        ; 8 bytes -> 8 int32 lanes
 *     codes  = {bytes & 15, bytes >> 4 & 15}        ; even and odd K elements
 *     mag    = vpermps(MAG8, codes & 7)             ; one instruction for 8 values
 *     sign   = (codes & 8) << 28                    ; 0x80000000 where negative
 *     value  = xor_ps(mag, sign)                    ; negate by flipping one bit
 *
 * and the activation is deinterleaved to match (even K in one register, odd in the other) rather
 * than the weights being interleaved. `tools/test_fp4_gemv_cpu.py` holds this to the torch
 * reference, and `tools/fp4_decode.py` holds the reference to the checkpoint's own layout.
 *
 * Layout, exactly as stored in the HF safetensors:
 *   w [N, K/2] uint8   two E2M1 codes per byte along K, LOW nibble = even K element
 *   s [N, K/32] uint8  UE8M0, value 2**(b-127)
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#define FP4_GROUP 32

/* ------------------------------------------------------------------ thread control
 *
 * This exists because of a trap that would otherwise cost 8x silently: PyTorch calls
 * `omp_set_num_threads(1)` during its own initialisation (it runs its intra-op parallelism on a
 * pool of its own, and sets the OpenMP count to 1 to stop the two from fighting). That setting is
 * process-global, so a library that just uses `#pragma omp parallel for` inherits one thread
 * whenever torch has been imported -- which, in an engine whose model, tokenizer and attention
 * are all torch, is always.
 *
 * Measured on a 12-core Ryzen 5900X, one expert forward: 6.0 ms with one thread, 0.75 ms with
 * all of them. Nobody would look for that in a profiler labelled "OpenMP" after writing a
 * parallel loop, so the thread count is set explicitly by the caller instead of being left to
 * whatever the process happened to inherit.
 */
#ifdef _OPENMP
#include <omp.h>
void fp4_cpu_set_threads(int n) { if (n > 0) omp_set_num_threads(n); }
int fp4_cpu_max_threads(void) { return omp_get_max_threads(); }
int fp4_cpu_thread_limit(void) { return omp_get_thread_limit(); }
#else
void fp4_cpu_set_threads(int n) { (void)n; }
int fp4_cpu_max_threads(void) { return 1; }
int fp4_cpu_thread_limit(void) { return 1; }
#endif

/* ------------------------------------------------------------------ scalar reference */
/* Kept in the library on purpose: the tests compare the vector path against it, so a broken
 * intrinsic shows up as a mismatch between two functions in the same file rather than as a
 * slightly-wrong model. */
void fp4_gemv_ref(const uint8_t *w, const uint8_t *s, const float *x, float *y, long N, long K) {
    static const float MAG[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const long NG = K / FP4_GROUP;
    for (long n = 0; n < N; ++n) {
        const uint8_t *wp = w + n * (K / 2);
        const uint8_t *sp = s + n * NG;
        float acc = 0.0f;
        for (long g = 0; g < NG; ++g) {
            float gsum = 0.0f;
            for (long j = 0; j < FP4_GROUP; ++j) {
                const uint8_t byte = wp[(g * FP4_GROUP + j) >> 1];
                const uint8_t code = (j & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
                const float mag = MAG[code & 7];
                gsum += x[g * FP4_GROUP + j] * ((code & 8) ? -mag : mag);
            }
            /* 2**(b-127) built from the bit pattern: the same uint32 shift as tools/fp4_decode.py
             * ::ue8m0_bits and as the kernel's _ue8m0. b=0 gives 0.0 here, not 2**-127; that is
             * the documented corner, and it is the behaviour the two other paths share. */
            const uint32_t sb = ((uint32_t)sp[g]) << 23;
            float scale;
            memcpy(&scale, &sb, sizeof(scale));
            acc += gsum * scale;
        }
        y[n] = acc;
    }
}

/* ------------------------------------------------------------------ AVX2 */
#ifdef __AVX2__

/* Even/odd deinterleave of two 8-lane float registers holding 16 consecutive K elements.
 * shufps takes lanes (0,2) of each 128-bit half; the permute then concatenates the two halves
 * in order, so the result is x[0],x[2],...,x[14] (or x[1],x[3],...,x[15]). */
static inline __m256 deint_even(__m256 a, __m256 b) {
    const __m256i idx = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    return _mm256_permutevar8x32_ps(_mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0)), idx);
}
static inline __m256 deint_odd(__m256 a, __m256 b) {
    const __m256i idx = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    return _mm256_permutevar8x32_ps(_mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1)), idx);
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

void fp4_gemv(const uint8_t *w, const uint8_t *s, const float *x, float *y, long N, long K) {
    const __m256 mag8 = _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f);
    const __m256i m7 = _mm256_set1_epi32(7);
    const __m256i m8 = _mm256_set1_epi32(8);
    const __m256i m15 = _mm256_set1_epi32(0x0F);
    const long NG = K / FP4_GROUP;

    /* Rows are independent and a row is one contiguous 2304-5120 byte read, so splitting N over
     * threads splits the bytes with no false sharing and no reduction. This is the whole reason
     * the RAM tier is worth having on a 12-16 core host: one core cannot pull 190 GB of experts
     * out of DDR4 on its own. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long n = 0; n < N; ++n) {
        const uint8_t *wp = w + n * (K / 2);
        const uint8_t *sp = s + n * NG;
        /* Four independent accumulators: the group partial is summed at the end of the group,
         * and independent chains keep the FMA pipeline busy while the next group's weights
         * load. */
        float acc = 0.0f;
        for (long g = 0; g < NG; ++g) {
            const float *xg = x + g * FP4_GROUP;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            for (long j = 0; j < FP4_GROUP; j += 16) {
                /* 8 packed bytes == 16 K elements: byte i carries K elements 2i (low) and 2i+1. */
                __m256i bytes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(wp + (g * FP4_GROUP + j) / 2)));
                __m256i lo = _mm256_and_si256(bytes, m15);              /* even K codes */
                __m256i hi = _mm256_and_si256(_mm256_srli_epi32(bytes, 4), m15); /* odd K codes */
                __m256 vlo = _mm256_xor_ps(
                    _mm256_permutevar8x32_ps(mag8, _mm256_and_si256(lo, m7)),
                    _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(lo, m8), 28)));
                __m256 vhi = _mm256_xor_ps(
                    _mm256_permutevar8x32_ps(mag8, _mm256_and_si256(hi, m7)),
                    _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(hi, m8), 28)));
                __m256 xa = _mm256_loadu_ps(xg + j);
                __m256 xb = _mm256_loadu_ps(xg + j + 8);
                a0 = _mm256_fmadd_ps(deint_even(xa, xb), vlo, a0);
                a1 = _mm256_fmadd_ps(deint_odd(xa, xb), vhi, a1);
            }
            const uint32_t sb = ((uint32_t)sp[g]) << 23;
            float scale;
            memcpy(&scale, &sb, sizeof(scale));
            acc += hsum256(_mm256_add_ps(a0, a1)) * scale;
        }
        y[n] = acc;
    }
}

#else /* portable build: the tests still run, just slowly */

void fp4_gemv(const uint8_t *w, const uint8_t *s, const float *x, float *y, long N, long K) {
    fp4_gemv_ref(w, s, x, y, N, K);
}

#endif /* __AVX2__ */

/* ------------------------------------------------------------------ fused SwiGLU epilogue
 * gate = min(g, limit); up = clamp(u, -limit, limit); h = silu(gate) * up * routing_weight
 * Mirrors engine/model.py::Expert.forward and v41_ref, so the CPU tier produces the same
 * numbers as the GPU tier rather than merely similar ones. */
void fp4_swiglu(const float *gate, const float *up, float *h, long n, float limit, float weight) {
    for (long i = 0; i < n; ++i) {
        float g = gate[i];
        float u = up[i];
        if (limit > 0.0f) {
            if (g > limit) g = limit;
            if (u > limit) u = limit;
            if (u < -limit) u = -limit;
        }
        const float silu = g / (1.0f + expf(-g));
        h[i] = silu * u * weight;
    }
}
