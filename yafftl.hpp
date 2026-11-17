// yafftl.hpp — Yet Another FFT Library
//
// Single-header, power-of-two, mixed-radix (4/8) Cooley-Tukey FFT,
// generalized into a standalone transform you can use for anything
// (spectra, convolution, filtering, your own bignum layer, whatever).
//
//   #include "yafftl.hpp"
//   auto spectrum = yafftl::fft(signal);
//   auto back     = yafftl::ifft(spectrum);
//   auto product  = yafftl::convolve(a, b);       // linear convolution
//
// That's the whole API most people need. Everything below is either the
// engine or opt-in performance knobs (Options::threads).
//
//
// Design notes (for anyone extending this):
//
// Algorithm: recursive mixed-radix Cooley-Tukey, decimation in the "four
// step" form. For N = r*m, viewing the array as an r x m matrix (row = arm):
//   forward:  column pass (radix-r butterfly + fused output twiddle) then
//             recurse into each of the r rows of length m.
//   inverse:  recurse into the rows first, then an inverse column pass
//             (untwiddle inputs by conj(T_k), conjugated radix-r core).
// The internal frequency bins are in mixed-radix digit-reversed order; forward
// and inverse use the same order, so convolution remains valid without a final
// permutation pass. radix(L) alternates
// 8/4 by stage size: radix-8 is retained for smaller cache-resident stages,
// while radix-4 is preferred once stages reach 8192 points to reduce twiddle
// traffic and improve bandwidth (see fft_radix() below). Leaves at L <= 32 are hardcoded
// codelets: an in-register SIMD dft8 where AVX2+FMA or AVX-512 is available
// (build with -mavx2 -mfma, or -mavx512f -mavx512dq), scalar otherwise.
//
// Threading: opt-in only (Options::threads, default 1). Above
// Options::parallel_threshold, a level's r row-recursions are fanned out
// across std::thread instead of run sequentially. Fork/join has real
// overhead (tens of microseconds), which is why it's off by default and
// only kicks in for genuinely large transforms — small ones are faster
// single-threaded.
//
// Precision: this is a double-precision floating-point FFT. For exact
// integer results (e.g. big-integer multiplication via convolution), keep
// per-point magnitudes and transform length small enough that accumulated
// round-off stays under 0.5 — see the comment on MAX_SAFE_INTEGER_FFT_LEN
// below for the empirical bound this was validated against.

#ifndef YAFFTL_HPP
#define YAFFTL_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define YAFFTL_AVX512 1
#else
#define YAFFTL_AVX512 0
#endif
#if YAFFTL_AVX512 || (defined(__AVX2__) && defined(__FMA__))
#include <immintrin.h>
#define YAFFTL_SIMD 1
#else
#define YAFFTL_SIMD 0
#endif

// Tune the transition from radix-8 to radix-4 for a target CPU if desired.
#ifndef YAFFTL_RADIX4_STAGE_THRESHOLD
#define YAFFTL_RADIX4_STAGE_THRESHOLD 8192
#endif

namespace yafftl
{

    // Public options

    struct Options
    {
        // 1 = single-threaded (default). Set higher to let large transforms
        // fan work out across std::thread; ignored below parallel_threshold.
        unsigned threads = 1;
        // Minimum transform length before threading kicks in at all. The
        // default (2^18 = 262144 points) is the point past which fork/join
        // overhead is reliably smaller than the work it parallelizes.
        std::size_t parallel_threshold = 1u << 18;
    };

    // Smallest power of two >= n (n = 0 or 1 both map to 1: a length-<=1
    // transform is a no-op).
    constexpr std::size_t next_pow2(std::size_t n) noexcept
    {
        if (n <= 1)
            return 1;
        --n;
        // Bit-smear to the next all-ones value, then add one. Returns 0 only
        // when the next power of two cannot be represented by size_t.
        for (std::size_t shift = 1; shift < std::numeric_limits<std::size_t>::digits; shift <<= 1)
            n |= n >> shift;
        return n + 1;
    }

    // Empirical bound (validated against an exact reference transform on
    // adversarial random data) past which round-off in a double-precision FFT
    // can corrupt an exact-integer result such as a big-integer multiply via
    // convolution. Not enforced by this library — it's a fact about doubles,
    // not this code — but anyone building exact-integer convolution on top of
    // yafftl should keep n below this and choose per-point magnitude so the
    // pointwise product stays well under 2^53.
    inline constexpr std::size_t kMaxSafeIntegerFFTLen = 1ull << 20;

    namespace detail
    {

        // Twiddle tables
        //
        // Tightly packed per length: wr/wi = [T_1 | T_2 | ... | T_{r-1}],
        // T_k[c] = exp(-2*pi*i*k*c/n), c in [0, n/r). Built once per length and
        // cached in a fixed array indexed by log2(n). Each length is initialised once;
        // steady-state lookup is a call_once fast path without hashing or map nodes.

        template <class T, std::size_t Alignment>
        class AlignedAllocator
        {
        public:
            using value_type = T;
            using is_always_equal = std::true_type;

            AlignedAllocator() noexcept = default;
            template <class U>
            constexpr AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

            [[nodiscard]] T *allocate(std::size_t count)
            {
                if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
                    throw std::bad_array_new_length();
                return static_cast<T *>(::operator new(count * sizeof(T), std::align_val_t(Alignment)));
            }

            void deallocate(T *ptr, std::size_t) noexcept
            {
                ::operator delete(ptr, std::align_val_t(Alignment));
            }

            template <class U>
            struct rebind
            {
                using other = AlignedAllocator<U, Alignment>;
            };
        };

        template <class T, class U, std::size_t A>
        constexpr bool operator==(const AlignedAllocator<T, A> &, const AlignedAllocator<U, A> &) noexcept { return true; }
        template <class T, class U, std::size_t A>
        constexpr bool operator!=(const AlignedAllocator<T, A> &, const AlignedAllocator<U, A> &) noexcept { return false; }

        struct TwiddleSet
        {
            std::size_t n = 0;
            std::size_t radix = 0;
            std::size_t count = 0;
            // One 64-byte-aligned allocation, split into SoA real/imaginary planes.
            // The block stride is a multiple of the active SIMD width in every kernel.
            std::vector<double, AlignedAllocator<double, 64>> storage;

            double *wr() noexcept { return storage.data(); }
            double *wi() noexcept { return storage.data() + count; }
            const double *wr() const noexcept { return storage.data(); }
            const double *wi() const noexcept { return storage.data() + count; }
        };

        constexpr double kTwoPi = 6.283185307179586476925286766559;
        constexpr double kHalfSqrt2 = 0.70710678118654752440084436210484905; // 1/sqrt(2)

        // radix(L): retain the small-stage 8/4 schedule, but favour radix-4 for
        // L >= YAFFTL_RADIX4_STAGE_THRESHOLD where lower twiddle traffic wins.
        // L <= 32 are hardcoded leaves and never consult this.
        constexpr std::size_t fft_radix(std::size_t n) noexcept
        {
            if (n <= 32)
                return 4;
            // Radix-4 streams fewer twiddles per byte moved on large stages.
            if (n >= YAFFTL_RADIX4_STAGE_THRESHOLD)
                return 4;
            // n is a power of two, so its binary logarithm is its trailing-zero count.
            std::size_t log2n = 0;
            while (n > 1)
            {
                n >>= 1;
                ++log2n;
            }
            return (log2n & 3) == 1 || (log2n & 3) == 2 ? 8 : 4;
        }

        constexpr std::size_t table_radix(std::size_t n) noexcept
        {
            return n <= 8 ? 2 : fft_radix(n);
        }

        inline void build_twiddle_set(TwiddleSet &ts, std::size_t n)
        {
            const std::size_t r = table_radix(n);
            const std::size_t m = n / r;
            ts.n = n;
            ts.radix = r;
            ts.count = (r - 1) * m;
            ts.storage.assign(2 * ts.count, 0.0);
            double *const wr = ts.wr();
            double *const wi = ts.wi();

            const double base = -kTwoPi / static_cast<double>(n);

            for (std::size_t k = 1; k < r; ++k)
            {
                double *wr_k = wr + (k - 1) * m;
                double *wi_k = wi + (k - 1) * m;

                // rho_k = W_r^k, a hardcoded r-th root of unity used for the
                // mirror fill T_k[m-c] = rho_k * conj(T_k[c]).
                double rho_r = 0.0, rho_i = 0.0;
                if (r == 8)
                {
                    switch (k)
                    {
                    case 1:
                        rho_r = kHalfSqrt2;
                        rho_i = -kHalfSqrt2;
                        break;
                    case 2:
                        rho_r = 0.0;
                        rho_i = -1.0;
                        break;
                    case 3:
                        rho_r = -kHalfSqrt2;
                        rho_i = -kHalfSqrt2;
                        break;
                    case 4:
                        rho_r = -1.0;
                        rho_i = 0.0;
                        break;
                    case 5:
                        rho_r = -kHalfSqrt2;
                        rho_i = kHalfSqrt2;
                        break;
                    case 6:
                        rho_r = 0.0;
                        rho_i = 1.0;
                        break;
                    case 7:
                        rho_r = kHalfSqrt2;
                        rho_i = kHalfSqrt2;
                        break;
                    }
                }
                else if (r == 4)
                {
                    switch (k)
                    {
                    case 1:
                        rho_r = 0.0;
                        rho_i = -1.0;
                        break;
                    case 2:
                        rho_r = -1.0;
                        rho_i = 0.0;
                        break;
                    case 3:
                        rho_r = 0.0;
                        rho_i = 1.0;
                        break;
                    }
                }
                else
                { // r == 2, k == 1
                    rho_r = -1.0;
                    rho_i = 0.0;
                }

                wr_k[0] = 1.0;
                wi_k[0] = 0.0;

                for (std::size_t c = 1; c <= m / 2; ++c)
                {
                    const double angle = base * static_cast<double>(k * c);
                    const double cr = std::cos(angle);
                    const double ci = std::sin(angle);
                    wr_k[c] = cr;
                    wi_k[c] = ci;
                    const std::size_t mc = m - c;
                    wr_k[mc] = rho_r * cr + rho_i * ci;
                    wi_k[mc] = rho_i * cr - rho_r * ci;
                }
            }
        }

        struct TwiddleCacheEntry
        {
            std::once_flag once;
            TwiddleSet table;
        };

        inline const TwiddleSet &get_twiddles(std::size_t n)
        {
            // All transform lengths are powers of two. Indexing by log2(n) avoids a
            // hash table, rehashing, shared_mutex traffic and per-entry node allocation.
            static std::array<TwiddleCacheEntry, std::numeric_limits<std::size_t>::digits> cache{};
            std::size_t index = 0;
            for (std::size_t v = n; v > 1; v >>= 1)
                ++index;
            assert(index < cache.size());
            TwiddleCacheEntry &entry = cache[index];
            std::call_once(entry.once, [&entry, n]
                           { build_twiddle_set(entry.table, n); });
            return entry.table;
        }

        // SIMD column-butterfly kernels (AVX-512: 8 lanes, AVX2+FMA: 4 lanes), with scalar fallbacks
        //
        //   radix{4,8}_butterfly_{fwd,inv}(xr, xi, m, twr, twi)
        //     arm j lives at (xr + j*m, xi + j*m); columns c in [0, m) — callers
        //     guarantee m is a multiple of 8 (true whenever this is reached: m is
        //     always a power of two >= 8 in this engine's recursion).
        //     forward: out_k = T_k * DFT_radix(arms)[k]     (twiddle the outputs)
        //     inverse: out_r = IDFT_radix(conj(T)*arms)[r]  (untwiddle the inputs,
        //                                                     conjugated core)

#if YAFFTL_SIMD

// One kernel body serves both widths: YAFFTL_V(op) pastes the intrinsic prefix.
#if YAFFTL_AVX512
        using vd = __m512d;
        constexpr std::size_t kLanes = 8;
#define YAFFTL_V(op) _mm512_##op
#define YAFFTL_TW_LOAD_PD(p) _mm512_load_pd(p)
#else
        using vd = __m256d;
        constexpr std::size_t kLanes = 4;
#define YAFFTL_V(op) _mm256_##op
#define YAFFTL_TW_LOAD_PD(p) _mm256_load_pd(p)
#endif

        struct Cx
        {
            vd re, im;
        };

        inline Cx cx_add(Cx a, Cx b) { return {YAFFTL_V(add_pd)(a.re, b.re), YAFFTL_V(add_pd)(a.im, b.im)}; }
        inline Cx cx_sub(Cx a, Cx b) { return {YAFFTL_V(sub_pd)(a.re, b.re), YAFFTL_V(sub_pd)(a.im, b.im)}; }
        inline Cx cx_mul(Cx z, Cx w)
        {
            return {YAFFTL_V(fmsub_pd)(z.re, w.re, YAFFTL_V(mul_pd)(z.im, w.im)),
                    YAFFTL_V(fmadd_pd)(z.im, w.re, YAFFTL_V(mul_pd)(z.re, w.im))};
        }
        inline Cx cx_mul_i(Cx a) { return {YAFFTL_V(xor_pd)(a.im, YAFFTL_V(set1_pd)(-0.0)), a.re}; }
        inline Cx cx_mul_neg_i(Cx a) { return {a.im, YAFFTL_V(xor_pd)(a.re, YAFFTL_V(set1_pd)(-0.0))}; }
        inline Cx cx_2pma(Cx p, Cx a)
        {
            const vd two = YAFFTL_V(set1_pd)(2.0);
            return {YAFFTL_V(fmsub_pd)(two, p.re, a.re), YAFFTL_V(fmsub_pd)(two, p.im, a.im)};
        }
        template <int SIGN>
        inline Cx cx_mul_45(Cx z)
        {
            const vd s = YAFFTL_V(set1_pd)(kHalfSqrt2);
            if constexpr (SIGN < 0)
            {
                return {YAFFTL_V(mul_pd)(YAFFTL_V(add_pd)(z.re, z.im), s),
                        YAFFTL_V(mul_pd)(YAFFTL_V(sub_pd)(z.im, z.re), s)};
            }
            else
            {
                return {YAFFTL_V(mul_pd)(YAFFTL_V(sub_pd)(z.re, z.im), s),
                        YAFFTL_V(mul_pd)(YAFFTL_V(add_pd)(z.re, z.im), s)};
            }
        }
        template <int SIGN>
        inline void cx_dft4(Cx a0, Cx a1, Cx a2, Cx a3, Cx out[4])
        {
            const Cx s0 = cx_add(a0, a2), d0 = cx_sub(a0, a2);
            const Cx s1 = cx_add(a1, a3), d1 = cx_sub(a1, a3);
            out[0] = cx_add(s0, s1);
            out[2] = cx_2pma(s0, out[0]);
            if constexpr (SIGN < 0)
                out[1] = cx_sub(d0, cx_mul_i(d1));
            else
                out[1] = cx_add(d0, cx_mul_i(d1));
            out[3] = cx_2pma(d0, out[1]);
        }

#endif // YAFFTL_SIMD

        inline void radix4_butterfly_fwd(double *xr, double *xi, std::size_t m,
                                         const double *twr, const double *twi)
        {
#if YAFFTL_SIMD
            for (std::size_t c = 0; c < m; c += kLanes)
            {
                Cx A[4];
                for (int j = 0; j < 4; ++j)
                {
                    A[j].re = YAFFTL_V(loadu_pd)(xr + j * m + c);
                    A[j].im = YAFFTL_V(loadu_pd)(xi + j * m + c);
                }
                Cx out[4];
                cx_dft4<-1>(A[0], A[1], A[2], A[3], out);
                A[0] = out[0];
                for (int k = 1; k < 4; ++k)
                {
                    A[k] = cx_mul(out[k], {YAFFTL_TW_LOAD_PD(twr + (k - 1) * m + c),
                                           YAFFTL_TW_LOAD_PD(twi + (k - 1) * m + c)});
                }
                for (int j = 0; j < 4; ++j)
                {
                    YAFFTL_V(storeu_pd)(xr + j * m + c, A[j].re);
                    YAFFTL_V(storeu_pd)(xi + j * m + c, A[j].im);
                }
            }
#else
            for (std::size_t c = 0; c < m; ++c)
            {
                double a0r = xr[0 * m + c], a0i = xi[0 * m + c];
                double a1r = xr[1 * m + c], a1i = xi[1 * m + c];
                double a2r = xr[2 * m + c], a2i = xi[2 * m + c];
                double a3r = xr[3 * m + c], a3i = xi[3 * m + c];
                const double s0r = a0r + a2r, s0i = a0i + a2i, d0r = a0r - a2r, d0i = a0i - a2i;
                const double s1r = a1r + a3r, s1i = a1i + a3i, d1r = a1r - a3r, d1i = a1i - a3i;
                double y0r = s0r + s1r, y0i = s0i + s1i;
                const double y2r = 2.0 * s0r - y0r, y2i = 2.0 * s0i - y0i;
                const double y1r = d0r + d1i, y1i = d0i - d1r;
                const double y3r = 2.0 * d0r - y1r, y3i = 2.0 * d0i - y1i;
                const double t1r = twr[0 * m + c], t1i = twi[0 * m + c];
                const double t2r = twr[1 * m + c], t2i = twi[1 * m + c];
                const double t3r = twr[2 * m + c], t3i = twi[2 * m + c];
                xr[0 * m + c] = y0r;
                xi[0 * m + c] = y0i;
                xr[1 * m + c] = y1r * t1r - y1i * t1i;
                xi[1 * m + c] = y1r * t1i + y1i * t1r;
                xr[2 * m + c] = y2r * t2r - y2i * t2i;
                xi[2 * m + c] = y2r * t2i + y2i * t2r;
                xr[3 * m + c] = y3r * t3r - y3i * t3i;
                xi[3 * m + c] = y3r * t3i + y3i * t3r;
            }
#endif
        }

        inline void radix4_butterfly_inv(double *xr, double *xi, std::size_t m,
                                         const double *twr, const double *twi)
        {
#if YAFFTL_SIMD
            const vd neg = YAFFTL_V(set1_pd)(-0.0);
            for (std::size_t c = 0; c < m; c += kLanes)
            {
                Cx A[4];
                for (int j = 0; j < 4; ++j)
                {
                    A[j].re = YAFFTL_V(loadu_pd)(xr + j * m + c);
                    A[j].im = YAFFTL_V(loadu_pd)(xi + j * m + c);
                }
                for (int k = 1; k < 4; ++k)
                {
                    Cx tk{YAFFTL_TW_LOAD_PD(twr + (k - 1) * m + c),
                          YAFFTL_V(xor_pd)(YAFFTL_TW_LOAD_PD(twi + (k - 1) * m + c), neg)};
                    A[k] = cx_mul(A[k], tk);
                }
                Cx out[4];
                cx_dft4<+1>(A[0], A[1], A[2], A[3], out);
                for (int j = 0; j < 4; ++j)
                {
                    YAFFTL_V(storeu_pd)(xr + j * m + c, out[j].re);
                    YAFFTL_V(storeu_pd)(xi + j * m + c, out[j].im);
                }
            }
#else
            for (std::size_t c = 0; c < m; ++c)
            {
                double a0r = xr[0 * m + c], a0i = xi[0 * m + c];
                double a1r = xr[1 * m + c], a1i = xi[1 * m + c];
                double a2r = xr[2 * m + c], a2i = xi[2 * m + c];
                double a3r = xr[3 * m + c], a3i = xi[3 * m + c];
                {
                    const double t1r = twr[0 * m + c], t1i = -twi[0 * m + c];
                    const double t2r = twr[1 * m + c], t2i = -twi[1 * m + c];
                    const double t3r = twr[2 * m + c], t3i = -twi[2 * m + c];
                    double nr = a1r * t1r - a1i * t1i, ni = a1r * t1i + a1i * t1r;
                    a1r = nr;
                    a1i = ni;
                    nr = a2r * t2r - a2i * t2i;
                    ni = a2r * t2i + a2i * t2r;
                    a2r = nr;
                    a2i = ni;
                    nr = a3r * t3r - a3i * t3i;
                    ni = a3r * t3i + a3i * t3r;
                    a3r = nr;
                    a3i = ni;
                }
                const double s0r = a0r + a2r, s0i = a0i + a2i, d0r = a0r - a2r, d0i = a0i - a2i;
                const double s1r = a1r + a3r, s1i = a1i + a3i, d1r = a1r - a3r, d1i = a1i - a3i;
                double y0r = s0r + s1r, y0i = s0i + s1i;
                const double y2r = 2.0 * s0r - y0r, y2i = 2.0 * s0i - y0i;
                const double y1r = d0r - d1i, y1i = d0i + d1r;
                const double y3r = 2.0 * d0r - y1r, y3i = 2.0 * d0i - y1i;
                xr[0 * m + c] = y0r;
                xi[0 * m + c] = y0i;
                xr[1 * m + c] = y1r;
                xi[1 * m + c] = y1i;
                xr[2 * m + c] = y2r;
                xi[2 * m + c] = y2i;
                xr[3 * m + c] = y3r;
                xi[3 * m + c] = y3i;
            }
#endif
        }

        inline void radix8_butterfly_fwd(double *xr, double *xi, std::size_t m,
                                         const double *twr, const double *twi)
        {
#if YAFFTL_SIMD
            for (std::size_t c = 0; c < m; c += kLanes)
            {
                Cx A[8];
                for (int j = 0; j < 8; ++j)
                {
                    A[j].re = YAFFTL_V(loadu_pd)(xr + j * m + c);
                    A[j].im = YAFFTL_V(loadu_pd)(xi + j * m + c);
                }
                Cx s[4], d[4];
                for (int j = 0; j < 4; ++j)
                {
                    s[j] = cx_add(A[j], A[j + 4]);
                    d[j] = cx_sub(A[j], A[j + 4]);
                }
                Cx e[4];
                e[0] = d[0];
                e[1] = cx_mul_45<-1>(d[1]);
                e[2] = cx_mul_neg_i(d[2]);
                e[3] = cx_mul_neg_i(cx_mul_45<-1>(d[3]));
                Cx E[4], O[4];
                cx_dft4<-1>(s[0], s[1], s[2], s[3], E);
                cx_dft4<-1>(e[0], e[1], e[2], e[3], O);
                Cx raw[8] = {E[0], O[0], E[1], O[1], E[2], O[2], E[3], O[3]};
                A[0] = raw[0];
                for (int k = 1; k < 8; ++k)
                {
                    A[k] = cx_mul(raw[k], {YAFFTL_TW_LOAD_PD(twr + (k - 1) * m + c),
                                           YAFFTL_TW_LOAD_PD(twi + (k - 1) * m + c)});
                }
                for (int j = 0; j < 8; ++j)
                {
                    YAFFTL_V(storeu_pd)(xr + j * m + c, A[j].re);
                    YAFFTL_V(storeu_pd)(xi + j * m + c, A[j].im);
                }
            }
#else
            constexpr double kc = kHalfSqrt2;
            for (std::size_t c = 0; c < m; ++c)
            {
                double ar[8], ai[8];
                for (int j = 0; j < 8; ++j)
                {
                    ar[j] = xr[j * m + c];
                    ai[j] = xi[j * m + c];
                }
                double sr[4], si[4], dr[4], di[4];
                for (int j = 0; j < 4; ++j)
                {
                    sr[j] = ar[j] + ar[j + 4];
                    si[j] = ai[j] + ai[j + 4];
                    dr[j] = ar[j] - ar[j + 4];
                    di[j] = ai[j] - ai[j + 4];
                }
                double er[4], ei[4];
                er[0] = dr[0];
                ei[0] = di[0];
                er[1] = kc * (dr[1] + di[1]);
                ei[1] = kc * (di[1] - dr[1]);
                er[2] = di[2];
                ei[2] = -dr[2];
                er[3] = kc * (di[3] - dr[3]);
                ei[3] = -kc * (dr[3] + di[3]);
                double Er[4], Ei[4], Or[4], Oi[4];
                {
                    const double s0r = sr[0] + sr[2], s0i = si[0] + si[2];
                    const double d0r = sr[0] - sr[2], d0i = si[0] - si[2];
                    const double s1r = sr[1] + sr[3], s1i = si[1] + si[3];
                    const double d1r = sr[1] - sr[3], d1i = si[1] - si[3];
                    Er[0] = s0r + s1r;
                    Ei[0] = s0i + s1i;
                    Er[2] = 2.0 * s0r - Er[0];
                    Ei[2] = 2.0 * s0i - Ei[0];
                    Er[1] = d0r + d1i;
                    Ei[1] = d0i - d1r;
                    Er[3] = 2.0 * d0r - Er[1];
                    Ei[3] = 2.0 * d0i - Ei[1];
                }
                {
                    const double s0r = er[0] + er[2], s0i = ei[0] + ei[2];
                    const double d0r = er[0] - er[2], d0i = ei[0] - ei[2];
                    const double s1r = er[1] + er[3], s1i = ei[1] + ei[3];
                    const double d1r = er[1] - er[3], d1i = ei[1] - ei[3];
                    Or[0] = s0r + s1r;
                    Oi[0] = s0i + s1i;
                    Or[2] = 2.0 * s0r - Or[0];
                    Oi[2] = 2.0 * s0i - Oi[0];
                    Or[1] = d0r + d1i;
                    Oi[1] = d0i - d1r;
                    Or[3] = 2.0 * d0r - Or[1];
                    Oi[3] = 2.0 * d0i - Oi[1];
                }
                for (int t = 0; t < 4; ++t)
                {
                    const int k2 = 2 * t, k3 = 2 * t + 1;
                    if (k2 == 0)
                    {
                        xr[0 * m + c] = Er[0];
                        xi[0 * m + c] = Ei[0];
                    }
                    else
                    {
                        const double tr = twr[(k2 - 1) * m + c], ti = twi[(k2 - 1) * m + c];
                        xr[k2 * m + c] = Er[t] * tr - Ei[t] * ti;
                        xi[k2 * m + c] = Er[t] * ti + Ei[t] * tr;
                    }
                    const double tr = twr[(k3 - 1) * m + c], ti = twi[(k3 - 1) * m + c];
                    xr[k3 * m + c] = Or[t] * tr - Oi[t] * ti;
                    xi[k3 * m + c] = Or[t] * ti + Oi[t] * tr;
                }
            }
#endif
        }

        inline void radix8_butterfly_inv(double *xr, double *xi, std::size_t m,
                                         const double *twr, const double *twi)
        {
#if YAFFTL_SIMD
            const vd neg = YAFFTL_V(set1_pd)(-0.0);
            for (std::size_t c = 0; c < m; c += kLanes)
            {
                Cx A[8];
                for (int j = 0; j < 8; ++j)
                {
                    A[j].re = YAFFTL_V(loadu_pd)(xr + j * m + c);
                    A[j].im = YAFFTL_V(loadu_pd)(xi + j * m + c);
                }
                for (int k = 1; k < 8; ++k)
                {
                    Cx tk{YAFFTL_TW_LOAD_PD(twr + (k - 1) * m + c),
                          YAFFTL_V(xor_pd)(YAFFTL_TW_LOAD_PD(twi + (k - 1) * m + c), neg)};
                    A[k] = cx_mul(A[k], tk);
                }
                Cx s[4], d[4];
                for (int j = 0; j < 4; ++j)
                {
                    s[j] = cx_add(A[j], A[j + 4]);
                    d[j] = cx_sub(A[j], A[j + 4]);
                }
                Cx e[4];
                e[0] = d[0];
                e[1] = cx_mul_45<+1>(d[1]);
                e[2] = cx_mul_i(d[2]);
                e[3] = cx_mul_i(cx_mul_45<+1>(d[3]));
                Cx E[4], O[4];
                cx_dft4<+1>(s[0], s[1], s[2], s[3], E);
                cx_dft4<+1>(e[0], e[1], e[2], e[3], O);
                for (int t = 0; t < 4; ++t)
                {
                    A[2 * t] = E[t];
                    A[2 * t + 1] = O[t];
                }
                for (int j = 0; j < 8; ++j)
                {
                    YAFFTL_V(storeu_pd)(xr + j * m + c, A[j].re);
                    YAFFTL_V(storeu_pd)(xi + j * m + c, A[j].im);
                }
            }
#else
            constexpr double kc = kHalfSqrt2;
            for (std::size_t c = 0; c < m; ++c)
            {
                double ar[8], ai[8];
                for (int j = 0; j < 8; ++j)
                {
                    ar[j] = xr[j * m + c];
                    ai[j] = xi[j * m + c];
                }
                for (int k = 1; k < 8; ++k)
                {
                    const double tr = twr[(k - 1) * m + c], ti = -twi[(k - 1) * m + c];
                    const double nr = ar[k] * tr - ai[k] * ti, ni = ar[k] * ti + ai[k] * tr;
                    ar[k] = nr;
                    ai[k] = ni;
                }
                double sr[4], si[4], dr[4], di[4];
                for (int j = 0; j < 4; ++j)
                {
                    sr[j] = ar[j] + ar[j + 4];
                    si[j] = ai[j] + ai[j + 4];
                    dr[j] = ar[j] - ar[j + 4];
                    di[j] = ai[j] - ai[j + 4];
                }
                double er[4], ei[4];
                er[0] = dr[0];
                ei[0] = di[0];
                er[1] = kc * (dr[1] - di[1]);
                ei[1] = kc * (dr[1] + di[1]);
                er[2] = -di[2];
                ei[2] = dr[2];
                er[3] = -kc * (dr[3] + di[3]);
                ei[3] = kc * (dr[3] - di[3]);
                double Er[4], Ei[4], Or[4], Oi[4];
                {
                    const double s0r = sr[0] + sr[2], s0i = si[0] + si[2];
                    const double d0r = sr[0] - sr[2], d0i = si[0] - si[2];
                    const double s1r = sr[1] + sr[3], s1i = si[1] + si[3];
                    const double d1r = sr[1] - sr[3], d1i = si[1] - si[3];
                    Er[0] = s0r + s1r;
                    Ei[0] = s0i + s1i;
                    Er[2] = 2.0 * s0r - Er[0];
                    Ei[2] = 2.0 * s0i - Ei[0];
                    Er[1] = d0r - d1i;
                    Ei[1] = d0i + d1r;
                    Er[3] = 2.0 * d0r - Er[1];
                    Ei[3] = 2.0 * d0i - Ei[1];
                }
                {
                    const double s0r = er[0] + er[2], s0i = ei[0] + ei[2];
                    const double d0r = er[0] - er[2], d0i = ei[0] - ei[2];
                    const double s1r = er[1] + er[3], s1i = ei[1] + ei[3];
                    const double d1r = er[1] - er[3], d1i = ei[1] - ei[3];
                    Or[0] = s0r + s1r;
                    Oi[0] = s0i + s1i;
                    Or[2] = 2.0 * s0r - Or[0];
                    Oi[2] = 2.0 * s0i - Oi[0];
                    Or[1] = d0r - d1i;
                    Oi[1] = d0i + d1r;
                    Or[3] = 2.0 * d0r - Or[1];
                    Oi[3] = 2.0 * d0i - Oi[1];
                }
                for (int t = 0; t < 4; ++t)
                {
                    xr[2 * t * m + c] = Er[t];
                    xi[2 * t * m + c] = Ei[t];
                    xr[(2 * t + 1) * m + c] = Or[t];
                    xi[(2 * t + 1) * m + c] = Oi[t];
                }
            }
#endif
        }

        // Terminal (leaf) codelets — scalar, constants only, L in {2,4,8,16,32}

        inline void dft2_scalar(double *xr, double *xi)
        {
            const double ar = xr[0], br = xr[1], ai = xi[0], bi = xi[1];
            xr[0] = ar + br;
            xi[0] = ai + bi;
            xr[1] = ar - br;
            xi[1] = ai - bi;
        }

        template <int SIGN>
        inline void dft4_scalar(double *xr, double *xi)
        {
            const double a0r = xr[0], a0i = xi[0], a1r = xr[1], a1i = xi[1];
            const double a2r = xr[2], a2i = xi[2], a3r = xr[3], a3i = xi[3];
            const double s0r = a0r + a2r, s0i = a0i + a2i, d0r = a0r - a2r, d0i = a0i - a2i;
            const double s1r = a1r + a3r, s1i = a1i + a3i, d1r = a1r - a3r, d1i = a1i - a3i;
            xr[0] = s0r + s1r;
            xi[0] = s0i + s1i;
            xr[2] = 2.0 * s0r - xr[0];
            xi[2] = 2.0 * s0i - xi[0];
            if constexpr (SIGN < 0)
            {
                xr[1] = d0r + d1i;
                xi[1] = d0i - d1r;
            }
            else
            {
                xr[1] = d0r - d1i;
                xi[1] = d0i + d1r;
            }
            xr[3] = 2.0 * d0r - xr[1];
            xi[3] = 2.0 * d0i - xi[1];
        }

        template <int SIGN>
        inline void dft8_scalar(double *xr, double *xi)
        {
            constexpr double c = kHalfSqrt2;
            double sr[4], si[4], dr[4], di[4];
            for (int j = 0; j < 4; ++j)
            {
                sr[j] = xr[j] + xr[j + 4];
                si[j] = xi[j] + xi[j + 4];
                dr[j] = xr[j] - xr[j + 4];
                di[j] = xi[j] - xi[j + 4];
            }
            double er[4], ei[4];
            er[0] = dr[0];
            ei[0] = di[0];
            if constexpr (SIGN < 0)
            {
                er[1] = c * (dr[1] + di[1]);
                ei[1] = c * (di[1] - dr[1]);
                er[2] = di[2];
                ei[2] = -dr[2];
                er[3] = c * (di[3] - dr[3]);
                ei[3] = -c * (dr[3] + di[3]);
            }
            else
            {
                er[1] = c * (dr[1] - di[1]);
                ei[1] = c * (dr[1] + di[1]);
                er[2] = -di[2];
                ei[2] = dr[2];
                er[3] = -c * (dr[3] + di[3]);
                ei[3] = c * (dr[3] - di[3]);
            }
            double E[4][2], O[4][2];
            {
                const double s0r = sr[0] + sr[2], s0i = si[0] + si[2];
                const double d0r = sr[0] - sr[2], d0i = si[0] - si[2];
                const double s1r = sr[1] + sr[3], s1i = si[1] + si[3];
                const double d1r = sr[1] - sr[3], d1i = si[1] - si[3];
                E[0][0] = s0r + s1r;
                E[0][1] = s0i + s1i;
                E[2][0] = 2.0 * s0r - E[0][0];
                E[2][1] = 2.0 * s0i - E[0][1];
                if constexpr (SIGN < 0)
                {
                    E[1][0] = d0r + d1i;
                    E[1][1] = d0i - d1r;
                }
                else
                {
                    E[1][0] = d0r - d1i;
                    E[1][1] = d0i + d1r;
                }
                E[3][0] = 2.0 * d0r - E[1][0];
                E[3][1] = 2.0 * d0i - E[1][1];
            }
            {
                const double s0r = er[0] + er[2], s0i = ei[0] + ei[2];
                const double d0r = er[0] - er[2], d0i = ei[0] - ei[2];
                const double s1r = er[1] + er[3], s1i = ei[1] + ei[3];
                const double d1r = er[1] - er[3], d1i = ei[1] - ei[3];
                O[0][0] = s0r + s1r;
                O[0][1] = s0i + s1i;
                O[2][0] = 2.0 * s0r - O[0][0];
                O[2][1] = 2.0 * s0i - O[0][1];
                if constexpr (SIGN < 0)
                {
                    O[1][0] = d0r + d1i;
                    O[1][1] = d0i - d1r;
                }
                else
                {
                    O[1][0] = d0r - d1i;
                    O[1][1] = d0i + d1r;
                }
                O[3][0] = 2.0 * d0r - O[1][0];
                O[3][1] = 2.0 * d0i - O[1][1];
            }
            for (int t = 0; t < 4; ++t)
            {
                xr[2 * t] = E[t][0];
                xi[2 * t] = E[t][1];
                xr[2 * t + 1] = O[t][0];
                xi[2 * t + 1] = O[t][1];
            }
        }

#if YAFFTL_SIMD
        // In-register DFT8 built from radix-2 DIF stages. A stage is x*(+1|-1) + partner: the low lane of
        // each pair gets the sum, the high lane (low - high). Stage twiddles are folded into a constant
        // complex multiply (W8^j) and a cheap +/-i swap (W4). Each variant leaves its lanes bit-reversed and
        // fixes that up with the shuffles that are already needed to store, so output is in natural order.
        template <class P>
        inline Cx dif_stage(Cx z, vd sg, P partner)
        {
            return {YAFFTL_V(fmadd_pd)(z.re, sg, partner(z.re)), YAFFTL_V(fmadd_pd)(z.im, sg, partner(z.im))};
        }

#if YAFFTL_AVX512
        // One point per lane: three stages (partner = lane^4, ^2, ^1), then one permute undoes the bit reversal.
        template <int SIGN>
        inline void dft8_simd(double *xr, double *xi)
        {
            constexpr double c = kHalfSqrt2;
            const __m512d zero = _mm512_setzero_pd();
            Cx z{_mm512_loadu_pd(xr), _mm512_loadu_pd(xi)};
            z = dif_stage(z, _mm512_setr_pd(1, 1, 1, 1, -1, -1, -1, -1), [](__m512d v)
                          { return _mm512_shuffle_f64x2(v, v, 0x4E); });
            z = cx_mul(z, {_mm512_setr_pd(1, 1, 1, 1, 1, c, 0, -c), _mm512_setr_pd(0, 0, 0, 0, 0, SIGN * c, SIGN * 1.0, SIGN * c)});
            z = dif_stage(z, _mm512_setr_pd(1, 1, -1, -1, 1, 1, -1, -1), [](__m512d v)
                          { return _mm512_permutex_pd(v, 0x4E); });
            if constexpr (SIGN < 0)
                z = {_mm512_mask_mov_pd(z.re, 0x88, z.im), _mm512_mask_sub_pd(z.im, 0x88, zero, z.re)};
            else
                z = {_mm512_mask_sub_pd(z.re, 0x88, zero, z.im), _mm512_mask_mov_pd(z.im, 0x88, z.re)};
            z = dif_stage(z, _mm512_setr_pd(1, -1, 1, -1, 1, -1, 1, -1), [](__m512d v)
                          { return _mm512_permute_pd(v, 0x55); });
            const __m512i bitrev = _mm512_setr_epi64(0, 4, 2, 6, 1, 5, 3, 7);
            _mm512_storeu_pd(xr, _mm512_permutexvar_pd(bitrev, z.re));
            _mm512_storeu_pd(xi, _mm512_permutexvar_pd(bitrev, z.im));
        }
#else
        // Two registers per component (points 0-3 and 4-7): stage 1 is a plain add/sub across them, the other
        // two stages are a 4-point DIF inside each register, and unpacklo/hi interleaves evens with odds.
        template <int SIGN>
        inline void dft8_simd(double *xr, double *xi)
        {
            constexpr double c = kHalfSqrt2;
            const __m256d zero = _mm256_setzero_pd();
            const Cx lo{_mm256_loadu_pd(xr), _mm256_loadu_pd(xi)}, hi{_mm256_loadu_pd(xr + 4), _mm256_loadu_pd(xi + 4)};
            auto dft4 = [&](Cx z)
            {
                z = dif_stage(z, _mm256_setr_pd(1, 1, -1, -1), [](__m256d v)
                              { return _mm256_permute2f128_pd(v, v, 0x01); });
                if constexpr (SIGN < 0)
                    z = {_mm256_blend_pd(z.re, z.im, 0x8), _mm256_blend_pd(z.im, _mm256_sub_pd(zero, z.re), 0x8)};
                else
                    z = {_mm256_blend_pd(z.re, _mm256_sub_pd(zero, z.im), 0x8), _mm256_blend_pd(z.im, z.re, 0x8)};
                return dif_stage(z, _mm256_setr_pd(1, -1, 1, -1), [](__m256d v)
                                 { return _mm256_permute_pd(v, 0x5); });
            };
            const Cx e = dft4(cx_add(lo, hi));                                                                                           // lanes: X0, X4, X2, X6
            const Cx o = dft4(cx_mul(cx_sub(lo, hi), {_mm256_setr_pd(1, c, 0, -c), _mm256_setr_pd(0, SIGN * c, SIGN * 1.0, SIGN * c)})); // X1, X5, X3, X7
            _mm256_storeu_pd(xr, _mm256_unpacklo_pd(e.re, o.re));
            _mm256_storeu_pd(xr + 4, _mm256_unpackhi_pd(e.re, o.re));
            _mm256_storeu_pd(xi, _mm256_unpacklo_pd(e.im, o.im));
            _mm256_storeu_pd(xi + 4, _mm256_unpackhi_pd(e.im, o.im));
        }
#endif
#endif // YAFFTL_SIMD

        template <int SIGN>
        inline void dft8(double *xr, double *xi)
        {
#if YAFFTL_SIMD
            dft8_simd<SIGN>(xr, xi);
#else
            dft8_scalar<SIGN>(xr, xi);
#endif
        }

        // Scalar radix-4 column pass over m columns (used only by the L=16 leaf,
        // where m=4 is too small for the AVX-512 kernel's 8-wide groups).
        inline void radix4_column_scalar(double *xr, double *xi, std::size_t m,
                                         const double *twr, const double *twi, int sign)
        {
            const double sgn = sign < 0 ? 1.0 : -1.0;
            for (std::size_t c = 0; c < m; ++c)
            {
                double a0r = xr[0 * m + c], a0i = xi[0 * m + c];
                double a1r = xr[1 * m + c], a1i = xi[1 * m + c];
                double a2r = xr[2 * m + c], a2i = xi[2 * m + c];
                double a3r = xr[3 * m + c], a3i = xi[3 * m + c];
                if (sign > 0)
                {
                    const double t1r = twr[0 * m + c], t1i = sgn * twi[0 * m + c];
                    const double t2r = twr[1 * m + c], t2i = sgn * twi[1 * m + c];
                    const double t3r = twr[2 * m + c], t3i = sgn * twi[2 * m + c];
                    double nr = a1r * t1r - a1i * t1i, ni = a1r * t1i + a1i * t1r;
                    a1r = nr;
                    a1i = ni;
                    nr = a2r * t2r - a2i * t2i;
                    ni = a2r * t2i + a2i * t2r;
                    a2r = nr;
                    a2i = ni;
                    nr = a3r * t3r - a3i * t3i;
                    ni = a3r * t3i + a3i * t3r;
                    a3r = nr;
                    a3i = ni;
                }
                const double s0r = a0r + a2r, s0i = a0i + a2i;
                const double d0r = a0r - a2r, d0i = a0i - a2i;
                const double s1r = a1r + a3r, s1i = a1i + a3i;
                const double d1r = a1r - a3r, d1i = a1i - a3i;
                double y0r = s0r + s1r, y0i = s0i + s1i;
                double y2r = 2.0 * s0r - y0r, y2i = 2.0 * s0i - y0i;
                double y1r, y1i;
                if (sign < 0)
                {
                    y1r = d0r + d1i;
                    y1i = d0i - d1r;
                }
                else
                {
                    y1r = d0r - d1i;
                    y1i = d0i + d1r;
                }
                const double y3r = 2.0 * d0r - y1r, y3i = 2.0 * d0i - y1i;
                if (sign < 0)
                {
                    const double t1r = twr[0 * m + c], t1i = sgn * twi[0 * m + c];
                    const double t2r = twr[1 * m + c], t2i = sgn * twi[1 * m + c];
                    const double t3r = twr[2 * m + c], t3i = sgn * twi[2 * m + c];
                    xr[1 * m + c] = y1r * t1r - y1i * t1i;
                    xi[1 * m + c] = y1r * t1i + y1i * t1r;
                    xr[2 * m + c] = y2r * t2r - y2i * t2i;
                    xi[2 * m + c] = y2r * t2i + y2i * t2r;
                    xr[3 * m + c] = y3r * t3r - y3i * t3i;
                    xi[3 * m + c] = y3r * t3i + y3i * t3r;
                }
                else
                {
                    xr[1 * m + c] = y1r;
                    xi[1 * m + c] = y1i;
                    xr[2 * m + c] = y2r;
                    xi[2 * m + c] = y2i;
                    xr[3 * m + c] = y3r;
                    xi[3 * m + c] = y3i;
                }
                xr[0 * m + c] = y0r;
                xi[0 * m + c] = y0i;
            }
        }

        inline void base_transform(double *xr, double *xi, std::size_t n, int sign, const TwiddleSet *tw)
        {
            switch (n)
            {
            case 1:
                return;
            case 2:
                dft2_scalar(xr, xi);
                return;
            case 4:
                sign < 0 ? dft4_scalar<-1>(xr, xi) : dft4_scalar<+1>(xr, xi);
                return;
            case 8:
                sign < 0 ? dft8<-1>(xr, xi) : dft8<+1>(xr, xi);
                return;
            case 16:
                assert(tw && tw->n == 16);
                if (sign < 0)
                {
                    radix4_column_scalar(xr, xi, 4, tw->wr(), tw->wi(), sign);
                    for (int k = 0; k < 4; ++k)
                        dft4_scalar<-1>(xr + 4 * k, xi + 4 * k);
                }
                else
                {
                    for (int k = 0; k < 4; ++k)
                        dft4_scalar<+1>(xr + 4 * k, xi + 4 * k);
                    radix4_column_scalar(xr, xi, 4, tw->wr(), tw->wi(), sign);
                }
                return;
            case 32:
                assert(tw && tw->n == 32);
                if (sign < 0)
                {
                    radix4_butterfly_fwd(xr, xi, 8, tw->wr(), tw->wi());
                    for (int k = 0; k < 4; ++k)
                        dft8<-1>(xr + 8 * k, xi + 8 * k);
                }
                else
                {
                    for (int k = 0; k < 4; ++k)
                        dft8<+1>(xr + 8 * k, xi + 8 * k);
                    radix4_butterfly_inv(xr, xi, 8, tw->wr(), tw->wi());
                }
                return;
            default:
                assert(false && "base_transform: unsupported terminal length");
            }
        }

        // Recursive driver, with opt-in fork/join threading

        constexpr std::size_t kMaxLevels = 64; // generous bound on recursion depth

        inline void fft_recursive(double *xr, double *xi, std::size_t L, int sign,
                                  const TwiddleSet **tabs, const Options &opts, unsigned threads)
        {
            if (L <= 32)
            {
                base_transform(xr, xi, L, sign, tabs[0]);
                return;
            }

            const std::size_t r = fft_radix(L);
            const std::size_t m = L / r;
            const TwiddleSet &tw = *tabs[0];

            auto column_pass = [&]
            {
                if (r == 4)
                {
                    if (sign < 0)
                        radix4_butterfly_fwd(xr, xi, m, tw.wr(), tw.wi());
                    else
                        radix4_butterfly_inv(xr, xi, m, tw.wr(), tw.wi());
                }
                else
                {
                    if (sign < 0)
                        radix8_butterfly_fwd(xr, xi, m, tw.wr(), tw.wi());
                    else
                        radix8_butterfly_inv(xr, xi, m, tw.wr(), tw.wi());
                }
            };

            auto recurse_rows = [&]
            {
                if (threads > 1 && L >= opts.parallel_threshold)
                {
                    // Treat threads as a hard worker budget. The previous fan-out
                    // spawned r-1 threads even when only two workers were requested.
                    // A fixed stack array also removes the per-node vector allocation.
                    const unsigned workers = std::min<unsigned>(threads, static_cast<unsigned>(r));
                    const unsigned base_budget = threads / workers;
                    const unsigned extra_budget = threads % workers;
                    std::array<std::thread, 7> pool;

                    auto run_worker = [&](unsigned worker)
                    {
                        const unsigned child_budget = base_budget + (worker < extra_budget ? 1u : 0u);
                        for (std::size_t k = worker; k < r; k += workers)
                        {
                            fft_recursive(xr + k * m, xi + k * m, m, sign, tabs + 1,
                                          opts, child_budget);
                        }
                    };

                    for (unsigned worker = 1; worker < workers; ++worker)
                        pool[worker - 1] = std::thread(run_worker, worker);
                    run_worker(0);
                    for (unsigned worker = 1; worker < workers; ++worker)
                        pool[worker - 1].join();
                }
                else
                {
                    for (std::size_t k = 0; k < r; ++k)
                        fft_recursive(xr + k * m, xi + k * m, m, sign, tabs + 1, opts, 1);
                }
            };

            if (sign < 0)
            {
                column_pass();
                recurse_rows();
            }
            else
            {
                recurse_rows();
                column_pass();
            }
        }

        inline bool is_pow2(std::size_t n) noexcept { return n != 0 && (n & (n - 1)) == 0; }

    } // namespace detail

    // Low-level in-place API — split real/imaginary arrays, power-of-two only.
    // This is the fast path: no allocation beyond first-use twiddle tables. The
    // forward output is in mixed-radix digit-reversed bin order; the inverse
    // consumes the same order. Callers requiring conventional natural-frequency
    // ordering must explicitly reorder the bins.

    inline void transform(double *re, double *im, std::size_t n, bool inverse, Options opts = {})
    {
        if (!detail::is_pow2(n))
        {
            throw std::invalid_argument("yafftl::transform: n must be a power of two");
        }
        if (n <= 1)
            return;

        const detail::TwiddleSet *tabs[detail::kMaxLevels]{};
        std::size_t cnt = 0;
        std::size_t L = n;
        if (n > 8)
        {
            tabs[cnt++] = &detail::get_twiddles(n);
            while (L > 32)
            {
                L /= detail::fft_radix(L);
                if (L > 8)
                    tabs[cnt++] = &detail::get_twiddles(L);
            }
        }
        assert(cnt <= detail::kMaxLevels);

        detail::fft_recursive(re, im, n, inverse ? +1 : -1, tabs, opts, opts.threads);

        if (inverse)
        {
            const double inv_n = 1.0 / static_cast<double>(n);
            std::size_t i = 0;
#if YAFFTL_SIMD
            const detail::vd vinv = YAFFTL_V(set1_pd)(inv_n);
            for (; i + detail::kLanes <= n; i += detail::kLanes)
            {
                YAFFTL_V(storeu_pd)(re + i, YAFFTL_V(mul_pd)(YAFFTL_V(loadu_pd)(re + i), vinv));
                YAFFTL_V(storeu_pd)(im + i, YAFFTL_V(mul_pd)(YAFFTL_V(loadu_pd)(im + i), vinv));
            }
#endif
            for (; i < n; ++i)
            {
                re[i] *= inv_n;
                im[i] *= inv_n;
            }
        }
    }

    // Easy high-level API — the front door for people who just want a transform.

    // Zero-pads to the next power of two and returns the padded-length spectrum
    // in the transform's mixed-radix digit-reversed bin order. Padded length is
    // next_pow2(data.size()).
    inline std::vector<std::complex<double>> fft(std::vector<std::complex<double>> data, Options opts = {})
    {
        const std::size_t n = next_pow2(data.size());
        if (n == 0)
            throw std::length_error("yafftl::fft: input is too large");
        // Zero-padding is done in the contiguous workspace; avoid resizing the
        // by-value input vector, which may otherwise trigger another allocation.
        std::vector<double> work(2 * n, 0.0);
        double *const re = work.data();
        double *const im = re + n;
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            re[i] = data[i].real();
            im[i] = data[i].imag();
        }
        transform(re, im, n, false, opts);

        std::vector<std::complex<double>> out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = {re[i], im[i]};
        return out;
    }

    inline std::vector<std::complex<double>> fft(const std::vector<double> &data, Options opts = {})
    {
        const std::size_t n = next_pow2(data.size());
        if (n == 0)
            throw std::length_error("yafftl::fft: input is too large");
        std::vector<double> work(2 * n, 0.0);
        double *const re = work.data();
        double *const im = re + n;
        std::copy(data.begin(), data.end(), re);
        transform(re, im, n, false, opts);

        std::vector<std::complex<double>> out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = {re[i], im[i]};
        return out;
    }

    // Inverse of fft(): input length is zero-padded to a power of two too, so
    // ifft(fft(x)) round-trips for any x (up to the padded length).
    inline std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> data, Options opts = {})
    {
        const std::size_t n = next_pow2(data.size());
        if (n == 0)
            throw std::length_error("yafftl::ifft: input is too large");
        std::vector<double> work(2 * n, 0.0);
        double *const re = work.data();
        double *const im = re + n;
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            re[i] = data[i].real();
            im[i] = data[i].imag();
        }
        transform(re, im, n, true, opts);

        std::vector<std::complex<double>> out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = {re[i], im[i]};
        return out;
    }

    // Linear (non-circular) convolution of two real signals via zero-padded FFT
    // pointwise-multiply-then-inverse. Output length is a.size() + b.size() - 1
    // — the library handles the padding/truncation for you.
    inline std::vector<double> convolve(const std::vector<double> &a, const std::vector<double> &b,
                                        Options opts = {})
    {
        if (a.empty() || b.empty())
            return {};
        if (a.size() > std::numeric_limits<std::size_t>::max() - (b.size() - 1))
            throw std::length_error("yafftl::convolve: output is too large");
        const std::size_t out_len = a.size() + b.size() - 1;
        const std::size_t n = next_pow2(out_len);
        if (n == 0 || n > std::numeric_limits<std::size_t>::max() / 4)
            throw std::length_error("yafftl::convolve: output is too large");

        // One contiguous workspace replaces four separately allocated arrays.
        std::vector<double> work(4 * n, 0.0);
        double *const ar = work.data();
        double *const ai = ar + n;
        double *const br = ai + n;
        double *const bi = br + n;
        std::copy(a.begin(), a.end(), ar);
        std::copy(b.begin(), b.end(), br);

        transform(ar, ai, n, false, opts);
        transform(br, bi, n, false, opts);

#if YAFFTL_SIMD
        std::size_t i = 0;
        for (; i + detail::kLanes <= n; i += detail::kLanes)
        {
            const detail::vd avr = YAFFTL_V(loadu_pd)(ar + i);
            const detail::vd avi = YAFFTL_V(loadu_pd)(ai + i);
            const detail::vd bvr = YAFFTL_V(loadu_pd)(br + i);
            const detail::vd bvi = YAFFTL_V(loadu_pd)(bi + i);
            const detail::vd rr = YAFFTL_V(fmsub_pd)(avr, bvr, YAFFTL_V(mul_pd)(avi, bvi));
            const detail::vd ri = YAFFTL_V(fmadd_pd)(avr, bvi, YAFFTL_V(mul_pd)(avi, bvr));
            YAFFTL_V(storeu_pd)(ar + i, rr);
            YAFFTL_V(storeu_pd)(ai + i, ri);
        }
        for (; i < n; ++i)
        {
            const double rr = ar[i] * br[i] - ai[i] * bi[i];
            const double ri = ar[i] * bi[i] + ai[i] * br[i];
            ar[i] = rr;
            ai[i] = ri;
        }
#else
        for (std::size_t i = 0; i < n; ++i)
        {
            const double rr = ar[i] * br[i] - ai[i] * bi[i];
            const double ri = ar[i] * bi[i] + ai[i] * br[i];
            ar[i] = rr;
            ai[i] = ri;
        }
#endif

        transform(ar, ai, n, true, opts);
        return std::vector<double>(ar, ar + out_len);
    }

} // namespace yafftl

#undef YAFFTL_V
#undef YAFFTL_TW_LOAD_PD

#endif // YAFFTL_HPP