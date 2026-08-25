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
// This never needs a bit-reversal or permutation pass. radix(L) alternates
// 8/4 by (log2 L) mod 4, matching the schedule that was measured fastest for
// the AVX-512 kernels (8-wide butterflies amortize better when they show up
// most levels; see fft_radix() below). Leaves at L <= 32 are hardcoded
// scalar codelets.
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
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
  #include <immintrin.h>
  #define YAFFTL_AVX512 1
#else
  #define YAFFTL_AVX512 0
#endif

namespace yafftl {


// Public options

struct Options {
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
constexpr std::size_t next_pow2(std::size_t n) noexcept {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Empirical bound (validated against an exact reference transform on
// adversarial random data) past which round-off in a double-precision FFT
// can corrupt an exact-integer result such as a big-integer multiply via
// convolution. Not enforced by this library — it's a fact about doubles,
// not this code — but anyone building exact-integer convolution on top of
// yafftl should keep n below this and choose per-point magnitude so the
// pointwise product stays well under 2^53.
inline constexpr std::size_t kMaxSafeIntegerFFTLen = 1ull << 20;

namespace detail {


// Twiddle tables
//
// Tightly packed per length: wr/wi = [T_1 | T_2 | ... | T_{r-1}],
// T_k[c] = exp(-2*pi*i*k*c/n), c in [0, n/r). Built once per length and
// cached; look-ups after that are just an unordered_map hit under a shared
// lock (uncontended once every length in use has been built).

struct TwiddleSet {
    std::size_t n = 0;
    std::size_t radix = 0;
    std::vector<double> wr, wi;
};

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kHalfSqrt2 = 0.70710678118654752440084436210484905;

// radix(L): 8 when L >= 64 and (log2 L) mod 4 in {1,2}, else 4. L <= 32 are
// hardcoded leaves and never consult this.
constexpr std::size_t fft_radix(std::size_t n) noexcept {
    if (n <= 32) return 4;
    std::size_t a = 0, v = n;
    while ((v & 1) == 0) { v >>= 1; ++a; }
    return (a & 3) == 1 || (a & 3) == 2 ? 8 : 4;
}

constexpr std::size_t table_radix(std::size_t n) noexcept {
    return n <= 8 ? 2 : fft_radix(n);
}

inline void build_twiddle_set(TwiddleSet& ts, std::size_t n) {
    const std::size_t r = table_radix(n);
    const std::size_t m = n / r;
    ts.n = n;
    ts.radix = r;
    ts.wr.assign((r - 1) * m, 0.0);
    ts.wi.assign((r - 1) * m, 0.0);

    const double base = -kTwoPi / static_cast<double>(n);

    for (std::size_t k = 1; k < r; ++k) {
        double* wr_k = ts.wr.data() + (k - 1) * m;
        double* wi_k = ts.wi.data() + (k - 1) * m;

        // rho_k = W_r^k, a hardcoded r-th root of unity used for the
        // mirror fill T_k[m-c] = rho_k * conj(T_k[c]).
        double rho_r = 0.0, rho_i = 0.0;
        if (r == 8) {
            switch (k) {
                case 1: rho_r = kHalfSqrt2;  rho_i = -kHalfSqrt2; break;
                case 2: rho_r = 0.0;         rho_i = -1.0;        break;
                case 3: rho_r = -kHalfSqrt2; rho_i = -kHalfSqrt2; break;
                case 4: rho_r = -1.0;        rho_i = 0.0;         break;
                case 5: rho_r = -kHalfSqrt2; rho_i = kHalfSqrt2;  break;
                case 6: rho_r = 0.0;         rho_i = 1.0;         break;
                case 7: rho_r = kHalfSqrt2;  rho_i = kHalfSqrt2;  break;
            }
        } else if (r == 4) {
            switch (k) {
                case 1: rho_r = 0.0;  rho_i = -1.0; break;
                case 2: rho_r = -1.0; rho_i = 0.0;  break;
                case 3: rho_r = 0.0;  rho_i = 1.0;  break;
            }
        } else {  // r == 2, k == 1
            rho_r = -1.0; rho_i = 0.0;
        }

        wr_k[0] = 1.0;
        wi_k[0] = 0.0;

        for (std::size_t c = 1; c <= m / 2; ++c) {
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

inline const TwiddleSet& get_twiddles(std::size_t n) {
    static std::unordered_map<std::size_t, TwiddleSet> cache;
    static std::shared_mutex mtx;

    {
        std::shared_lock lk(mtx);
        auto it = cache.find(n);
        if (it != cache.end()) return it->second;
    }

    TwiddleSet ts;
    build_twiddle_set(ts, n);

    std::unique_lock lk(mtx);
    return cache.try_emplace(n, std::move(ts)).first->second;
}


// AVX-512 column-butterfly kernels, with scalar fallbacks
//
//   radix{4,8}_butterfly_{fwd,inv}(xr, xi, m, twr, twi)
//     arm j lives at (xr + j*m, xi + j*m); columns c in [0, m) — callers
//     guarantee m is a multiple of 8 (true whenever this is reached: m is
//     always a power of two >= 8 in this engine's recursion).
//     forward: out_k = T_k * DFT_radix(arms)[k]     (twiddle the outputs)
//     inverse: out_r = IDFT_radix(conj(T)*arms)[r]  (untwiddle the inputs,
//                                                     conjugated core)

#if YAFFTL_AVX512

struct Cx8 { __m512d re, im; };

inline Cx8 cx8_add(Cx8 a, Cx8 b) { return {_mm512_add_pd(a.re, b.re), _mm512_add_pd(a.im, b.im)}; }
inline Cx8 cx8_sub(Cx8 a, Cx8 b) { return {_mm512_sub_pd(a.re, b.re), _mm512_sub_pd(a.im, b.im)}; }
inline Cx8 cx8_mul(Cx8 z, Cx8 w) {
    return {_mm512_fmsub_pd(z.re, w.re, _mm512_mul_pd(z.im, w.im)),
            _mm512_fmadd_pd(z.im, w.re, _mm512_mul_pd(z.re, w.im))};
}
inline Cx8 cx8_mul_i(Cx8 a) { return {_mm512_xor_pd(a.im, _mm512_set1_pd(-0.0)), a.re}; }
inline Cx8 cx8_mul_neg_i(Cx8 a) { return {a.im, _mm512_xor_pd(a.re, _mm512_set1_pd(-0.0))}; }
inline Cx8 cx8_2pma(Cx8 p, Cx8 a) {
    const __m512d two = _mm512_set1_pd(2.0);
    return {_mm512_fmsub_pd(two, p.re, a.re), _mm512_fmsub_pd(two, p.im, a.im)};
}
template <int SIGN>
inline Cx8 cx8_mul_45(Cx8 z) {
    const __m512d s = _mm512_set1_pd(kHalfSqrt2);
    if constexpr (SIGN < 0) {
        return {_mm512_mul_pd(_mm512_add_pd(z.re, z.im), s),
                _mm512_mul_pd(_mm512_sub_pd(z.im, z.re), s)};
    } else {
        return {_mm512_mul_pd(_mm512_sub_pd(z.re, z.im), s),
                _mm512_mul_pd(_mm512_add_pd(z.re, z.im), s)};
    }
}
template <int SIGN>
inline void cx8_dft4(Cx8 a0, Cx8 a1, Cx8 a2, Cx8 a3, Cx8 out[4]) {
    const Cx8 s0 = cx8_add(a0, a2), d0 = cx8_sub(a0, a2);
    const Cx8 s1 = cx8_add(a1, a3), d1 = cx8_sub(a1, a3);
    out[0] = cx8_add(s0, s1);
    out[2] = cx8_2pma(s0, out[0]);
    if constexpr (SIGN < 0) out[1] = cx8_sub(d0, cx8_mul_i(d1));
    else                    out[1] = cx8_add(d0, cx8_mul_i(d1));
    out[3] = cx8_2pma(d0, out[1]);
}

#endif  // YAFFTL_AVX512

inline void radix4_butterfly_fwd(double* xr, double* xi, std::size_t m,
                                  const double* twr, const double* twi) {
#if YAFFTL_AVX512
    for (std::size_t c = 0; c < m; c += 8) {
        Cx8 A[4];
        for (int j = 0; j < 4; ++j) {
            A[j].re = _mm512_loadu_pd(xr + j * m + c);
            A[j].im = _mm512_loadu_pd(xi + j * m + c);
        }
        Cx8 out[4];
        cx8_dft4<-1>(A[0], A[1], A[2], A[3], out);
        A[0] = out[0];
        for (int k = 1; k < 4; ++k) {
            A[k] = cx8_mul(out[k], {_mm512_loadu_pd(twr + (k - 1) * m + c),
                                     _mm512_loadu_pd(twi + (k - 1) * m + c)});
        }
        for (int j = 0; j < 4; ++j) {
            _mm512_storeu_pd(xr + j * m + c, A[j].re);
            _mm512_storeu_pd(xi + j * m + c, A[j].im);
        }
    }
#else
    for (std::size_t c = 0; c < m; ++c) {
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
        xr[0 * m + c] = y0r;                    xi[0 * m + c] = y0i;
        xr[1 * m + c] = y1r * t1r - y1i * t1i;  xi[1 * m + c] = y1r * t1i + y1i * t1r;
        xr[2 * m + c] = y2r * t2r - y2i * t2i;  xi[2 * m + c] = y2r * t2i + y2i * t2r;
        xr[3 * m + c] = y3r * t3r - y3i * t3i;  xi[3 * m + c] = y3r * t3i + y3i * t3r;
    }
#endif
}

inline void radix4_butterfly_inv(double* xr, double* xi, std::size_t m,
                                  const double* twr, const double* twi) {
#if YAFFTL_AVX512
    const __m512d neg = _mm512_set1_pd(-0.0);
    for (std::size_t c = 0; c < m; c += 8) {
        Cx8 A[4];
        for (int j = 0; j < 4; ++j) {
            A[j].re = _mm512_loadu_pd(xr + j * m + c);
            A[j].im = _mm512_loadu_pd(xi + j * m + c);
        }
        for (int k = 1; k < 4; ++k) {
            Cx8 tk{_mm512_loadu_pd(twr + (k - 1) * m + c),
                   _mm512_xor_pd(_mm512_loadu_pd(twi + (k - 1) * m + c), neg)};
            A[k] = cx8_mul(A[k], tk);
        }
        Cx8 out[4];
        cx8_dft4<+1>(A[0], A[1], A[2], A[3], out);
        for (int j = 0; j < 4; ++j) {
            _mm512_storeu_pd(xr + j * m + c, out[j].re);
            _mm512_storeu_pd(xi + j * m + c, out[j].im);
        }
    }
#else
    for (std::size_t c = 0; c < m; ++c) {
        double a0r = xr[0 * m + c], a0i = xi[0 * m + c];
        double a1r = xr[1 * m + c], a1i = xi[1 * m + c];
        double a2r = xr[2 * m + c], a2i = xi[2 * m + c];
        double a3r = xr[3 * m + c], a3i = xi[3 * m + c];
        {
            const double t1r = twr[0 * m + c], t1i = -twi[0 * m + c];
            const double t2r = twr[1 * m + c], t2i = -twi[1 * m + c];
            const double t3r = twr[2 * m + c], t3i = -twi[2 * m + c];
            double nr = a1r * t1r - a1i * t1i, ni = a1r * t1i + a1i * t1r;
            a1r = nr; a1i = ni;
            nr = a2r * t2r - a2i * t2i; ni = a2r * t2i + a2i * t2r;
            a2r = nr; a2i = ni;
            nr = a3r * t3r - a3i * t3i; ni = a3r * t3i + a3i * t3r;
            a3r = nr; a3i = ni;
        }
        const double s0r = a0r + a2r, s0i = a0i + a2i, d0r = a0r - a2r, d0i = a0i - a2i;
        const double s1r = a1r + a3r, s1i = a1i + a3i, d1r = a1r - a3r, d1i = a1i - a3i;
        double y0r = s0r + s1r, y0i = s0i + s1i;
        const double y2r = 2.0 * s0r - y0r, y2i = 2.0 * s0i - y0i;
        const double y1r = d0r - d1i, y1i = d0i + d1r;
        const double y3r = 2.0 * d0r - y1r, y3i = 2.0 * d0i - y1i;
        xr[0 * m + c] = y0r; xi[0 * m + c] = y0i;
        xr[1 * m + c] = y1r; xi[1 * m + c] = y1i;
        xr[2 * m + c] = y2r; xi[2 * m + c] = y2i;
        xr[3 * m + c] = y3r; xi[3 * m + c] = y3i;
    }
#endif
}

inline void radix8_butterfly_fwd(double* xr, double* xi, std::size_t m,
                                  const double* twr, const double* twi) {
#if YAFFTL_AVX512
    for (std::size_t c = 0; c < m; c += 8) {
        Cx8 A[8];
        for (int j = 0; j < 8; ++j) {
            A[j].re = _mm512_loadu_pd(xr + j * m + c);
            A[j].im = _mm512_loadu_pd(xi + j * m + c);
        }
        Cx8 s[4], d[4];
        for (int j = 0; j < 4; ++j) {
            s[j] = cx8_add(A[j], A[j + 4]);
            d[j] = cx8_sub(A[j], A[j + 4]);
        }
        Cx8 e[4];
        e[0] = d[0];
        e[1] = cx8_mul_45<-1>(d[1]);
        e[2] = cx8_mul_neg_i(d[2]);
        e[3] = cx8_mul_neg_i(cx8_mul_45<-1>(d[3]));
        Cx8 E[4], O[4];
        cx8_dft4<-1>(s[0], s[1], s[2], s[3], E);
        cx8_dft4<-1>(e[0], e[1], e[2], e[3], O);
        Cx8 raw[8] = {E[0], O[0], E[1], O[1], E[2], O[2], E[3], O[3]};
        A[0] = raw[0];
        for (int k = 1; k < 8; ++k) {
            A[k] = cx8_mul(raw[k], {_mm512_loadu_pd(twr + (k - 1) * m + c),
                                     _mm512_loadu_pd(twi + (k - 1) * m + c)});
        }
        for (int j = 0; j < 8; ++j) {
            _mm512_storeu_pd(xr + j * m + c, A[j].re);
            _mm512_storeu_pd(xi + j * m + c, A[j].im);
        }
    }
#else
    constexpr double kc = kHalfSqrt2;
    for (std::size_t c = 0; c < m; ++c) {
        double ar[8], ai[8];
        for (int j = 0; j < 8; ++j) { ar[j] = xr[j * m + c]; ai[j] = xi[j * m + c]; }
        double sr[4], si[4], dr[4], di[4];
        for (int j = 0; j < 4; ++j) {
            sr[j] = ar[j] + ar[j + 4]; si[j] = ai[j] + ai[j + 4];
            dr[j] = ar[j] - ar[j + 4]; di[j] = ai[j] - ai[j + 4];
        }
        double er[4], ei[4];
        er[0] = dr[0]; ei[0] = di[0];
        er[1] = kc * (dr[1] + di[1]);  ei[1] = kc * (di[1] - dr[1]);
        er[2] = di[2];                 ei[2] = -dr[2];
        er[3] = kc * (di[3] - dr[3]);  ei[3] = -kc * (dr[3] + di[3]);
        double Er[4], Ei[4], Or[4], Oi[4];
        {
            const double s0r = sr[0] + sr[2], s0i = si[0] + si[2];
            const double d0r = sr[0] - sr[2], d0i = si[0] - si[2];
            const double s1r = sr[1] + sr[3], s1i = si[1] + si[3];
            const double d1r = sr[1] - sr[3], d1i = si[1] - si[3];
            Er[0] = s0r + s1r; Ei[0] = s0i + s1i;
            Er[2] = 2.0 * s0r - Er[0]; Ei[2] = 2.0 * s0i - Ei[0];
            Er[1] = d0r + d1i; Ei[1] = d0i - d1r;
            Er[3] = 2.0 * d0r - Er[1]; Ei[3] = 2.0 * d0i - Ei[1];
        }
        {
            const double s0r = er[0] + er[2], s0i = ei[0] + ei[2];
            const double d0r = er[0] - er[2], d0i = ei[0] - ei[2];
            const double s1r = er[1] + er[3], s1i = ei[1] + ei[3];
            const double d1r = er[1] - er[3], d1i = ei[1] - ei[3];
            Or[0] = s0r + s1r; Oi[0] = s0i + s1i;
            Or[2] = 2.0 * s0r - Or[0]; Oi[2] = 2.0 * s0i - Oi[0];
            Or[1] = d0r + d1i; Oi[1] = d0i - d1r;
            Or[3] = 2.0 * d0r - Or[1]; Oi[3] = 2.0 * d0i - Oi[1];
        }
        for (int t = 0; t < 4; ++t) {
            const int k2 = 2 * t, k3 = 2 * t + 1;
            if (k2 == 0) {
                xr[0 * m + c] = Er[0]; xi[0 * m + c] = Ei[0];
            } else {
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

inline void radix8_butterfly_inv(double* xr, double* xi, std::size_t m,
                                  const double* twr, const double* twi) {
#if YAFFTL_AVX512
    const __m512d neg = _mm512_set1_pd(-0.0);
    for (std::size_t c = 0; c < m; c += 8) {
        Cx8 A[8];
        for (int j = 0; j < 8; ++j) {
            A[j].re = _mm512_loadu_pd(xr + j * m + c);
            A[j].im = _mm512_loadu_pd(xi + j * m + c);
        }
        for (int k = 1; k < 8; ++k) {
            Cx8 tk{_mm512_loadu_pd(twr + (k - 1) * m + c),
                   _mm512_xor_pd(_mm512_loadu_pd(twi + (k - 1) * m + c), neg)};
            A[k] = cx8_mul(A[k], tk);
        }
        Cx8 s[4], d[4];
        for (int j = 0; j < 4; ++j) {
            s[j] = cx8_add(A[j], A[j + 4]);
            d[j] = cx8_sub(A[j], A[j + 4]);
        }
        Cx8 e[4];
        e[0] = d[0];
        e[1] = cx8_mul_45<+1>(d[1]);
        e[2] = cx8_mul_i(d[2]);
        e[3] = cx8_mul_i(cx8_mul_45<+1>(d[3]));
        Cx8 E[4], O[4];
        cx8_dft4<+1>(s[0], s[1], s[2], s[3], E);
        cx8_dft4<+1>(e[0], e[1], e[2], e[3], O);
        for (int t = 0; t < 4; ++t) { A[2 * t] = E[t]; A[2 * t + 1] = O[t]; }
        for (int j = 0; j < 8; ++j) {
            _mm512_storeu_pd(xr + j * m + c, A[j].re);
            _mm512_storeu_pd(xi + j * m + c, A[j].im);
        }
    }
#else
    constexpr double kc = kHalfSqrt2;
    for (std::size_t c = 0; c < m; ++c) {
        double ar[8], ai[8];
        for (int j = 0; j < 8; ++j) { ar[j] = xr[j * m + c]; ai[j] = xi[j * m + c]; }
        for (int k = 1; k < 8; ++k) {
            const double tr = twr[(k - 1) * m + c], ti = -twi[(k - 1) * m + c];
            const double nr = ar[k] * tr - ai[k] * ti, ni = ar[k] * ti + ai[k] * tr;
            ar[k] = nr; ai[k] = ni;
        }
        double sr[4], si[4], dr[4], di[4];
        for (int j = 0; j < 4; ++j) {
            sr[j] = ar[j] + ar[j + 4]; si[j] = ai[j] + ai[j + 4];
            dr[j] = ar[j] - ar[j + 4]; di[j] = ai[j] - ai[j + 4];
        }
        double er[4], ei[4];
        er[0] = dr[0]; ei[0] = di[0];
        er[1] = kc * (dr[1] - di[1]);  ei[1] = kc * (dr[1] + di[1]);
        er[2] = -di[2];                ei[2] = dr[2];
        er[3] = -kc * (dr[3] + di[3]); ei[3] = kc * (dr[3] - di[3]);
        double Er[4], Ei[4], Or[4], Oi[4];
        {
            const double s0r = sr[0] + sr[2], s0i = si[0] + si[2];
            const double d0r = sr[0] - sr[2], d0i = si[0] - si[2];
            const double s1r = sr[1] + sr[3], s1i = si[1] + si[3];
            const double d1r = sr[1] - sr[3], d1i = si[1] - si[3];
            Er[0] = s0r + s1r; Ei[0] = s0i + s1i;
            Er[2] = 2.0 * s0r - Er[0]; Ei[2] = 2.0 * s0i - Ei[0];
            Er[1] = d0r - d1i; Ei[1] = d0i + d1r;
            Er[3] = 2.0 * d0r - Er[1]; Ei[3] = 2.0 * d0i - Ei[1];
        }
        {
            const double s0r = er[0] + er[2], s0i = ei[0] + ei[2];
            const double d0r = er[0] - er[2], d0i = ei[0] - ei[2];
            const double s1r = er[1] + er[3], s1i = ei[1] + ei[3];
            const double d1r = er[1] - er[3], d1i = ei[1] - ei[3];
            Or[0] = s0r + s1r; Oi[0] = s0i + s1i;
            Or[2] = 2.0 * s0r - Or[0]; Oi[2] = 2.0 * s0i - Oi[0];
            Or[1] = d0r - d1i; Oi[1] = d0i + d1r;
            Or[3] = 2.0 * d0r - Or[1]; Oi[3] = 2.0 * d0i - Oi[1];
        }
        for (int t = 0; t < 4; ++t) {
            xr[2 * t * m + c] = Er[t]; xi[2 * t * m + c] = Ei[t];
            xr[(2 * t + 1) * m + c] = Or[t]; xi[(2 * t + 1) * m + c] = Oi[t];
        }
    }
#endif
}


// Terminal (leaf) codelets — scalar, constants only, L in {2,4,8,16,32}

inline void dft2_scalar(double* xr, double* xi) {
    const double ar = xr[0], br = xr[1], ai = xi[0], bi = xi[1];
    xr[0] = ar + br; xi[0] = ai + bi;
    xr[1] = ar - br; xi[1] = ai - bi;
}

template <int SIGN>
inline void dft4_scalar(double* xr, double* xi) {
    const double a0r = xr[0], a0i = xi[0], a1r = xr[1], a1i = xi[1];
    const double a2r = xr[2], a2i = xi[2], a3r = xr[3], a3i = xi[3];
    const double s0r = a0r + a2r, s0i = a0i + a2i, d0r = a0r - a2r, d0i = a0i - a2i;
    const double s1r = a1r + a3r, s1i = a1i + a3i, d1r = a1r - a3r, d1i = a1i - a3i;
    xr[0] = s0r + s1r; xi[0] = s0i + s1i;
    xr[2] = 2.0 * s0r - xr[0]; xi[2] = 2.0 * s0i - xi[0];
    if constexpr (SIGN < 0) { xr[1] = d0r + d1i; xi[1] = d0i - d1r; }
    else                    { xr[1] = d0r - d1i; xi[1] = d0i + d1r; }
    xr[3] = 2.0 * d0r - xr[1]; xi[3] = 2.0 * d0i - xi[1];
}

template <int SIGN>
inline void dft8_scalar(double* xr, double* xi) {
    constexpr double c = kHalfSqrt2;
    double sr[4], si[4], dr[4], di[4];
    for (int j = 0; j < 4; ++j) {
        sr[j] = xr[j] + xr[j + 4]; si[j] = xi[j] + xi[j + 4];
        dr[j] = xr[j] - xr[j + 4]; di[j] = xi[j] - xi[j + 4];
    }
    double er[4], ei[4];
    er[0] = dr[0]; ei[0] = di[0];
    if constexpr (SIGN < 0) {
        er[1] = c * (dr[1] + di[1]);  ei[1] = c * (di[1] - dr[1]);
        er[2] = di[2];                ei[2] = -dr[2];
        er[3] = c * (di[3] - dr[3]);  ei[3] = -c * (dr[3] + di[3]);
    } else {
        er[1] = c * (dr[1] - di[1]);  ei[1] = c * (dr[1] + di[1]);
        er[2] = -di[2];               ei[2] = dr[2];
        er[3] = -c * (dr[3] + di[3]); ei[3] = c * (dr[3] - di[3]);
    }
    double E[4][2], O[4][2];
    {
        const double s0r = sr[0] + sr[2], s0i = si[0] + si[2];
        const double d0r = sr[0] - sr[2], d0i = si[0] - si[2];
        const double s1r = sr[1] + sr[3], s1i = si[1] + si[3];
        const double d1r = sr[1] - sr[3], d1i = si[1] - si[3];
        E[0][0] = s0r + s1r; E[0][1] = s0i + s1i;
        E[2][0] = 2.0 * s0r - E[0][0]; E[2][1] = 2.0 * s0i - E[0][1];
        if constexpr (SIGN < 0) { E[1][0] = d0r + d1i; E[1][1] = d0i - d1r; }
        else                    { E[1][0] = d0r - d1i; E[1][1] = d0i + d1r; }
        E[3][0] = 2.0 * d0r - E[1][0]; E[3][1] = 2.0 * d0i - E[1][1];
    }
    {
        const double s0r = er[0] + er[2], s0i = ei[0] + ei[2];
        const double d0r = er[0] - er[2], d0i = ei[0] - ei[2];
        const double s1r = er[1] + er[3], s1i = ei[1] + ei[3];
        const double d1r = er[1] - er[3], d1i = ei[1] - ei[3];
        O[0][0] = s0r + s1r; O[0][1] = s0i + s1i;
        O[2][0] = 2.0 * s0r - O[0][0]; O[2][1] = 2.0 * s0i - O[0][1];
        if constexpr (SIGN < 0) { O[1][0] = d0r + d1i; O[1][1] = d0i - d1r; }
        else                    { O[1][0] = d0r - d1i; O[1][1] = d0i + d1r; }
        O[3][0] = 2.0 * d0r - O[1][0]; O[3][1] = 2.0 * d0i - O[1][1];
    }
    for (int t = 0; t < 4; ++t) {
        xr[2 * t]     = E[t][0]; xi[2 * t]     = E[t][1];
        xr[2 * t + 1] = O[t][0]; xi[2 * t + 1] = O[t][1];
    }
}

// Scalar radix-4 column pass over m columns (used only by the L=16 leaf,
// where m=4 is too small for the AVX-512 kernel's 8-wide groups).
inline void radix4_column_scalar(double* xr, double* xi, std::size_t m,
                                  const double* twr, const double* twi, int sign) {
    const double sgn = sign < 0 ? 1.0 : -1.0;
    for (std::size_t c = 0; c < m; ++c) {
        double a0r = xr[0 * m + c], a0i = xi[0 * m + c];
        double a1r = xr[1 * m + c], a1i = xi[1 * m + c];
        double a2r = xr[2 * m + c], a2i = xi[2 * m + c];
        double a3r = xr[3 * m + c], a3i = xi[3 * m + c];
        if (sign > 0) {
            const double t1r = twr[0 * m + c], t1i = sgn * twi[0 * m + c];
            const double t2r = twr[1 * m + c], t2i = sgn * twi[1 * m + c];
            const double t3r = twr[2 * m + c], t3i = sgn * twi[2 * m + c];
            double nr = a1r * t1r - a1i * t1i, ni = a1r * t1i + a1i * t1r;
            a1r = nr; a1i = ni;
            nr = a2r * t2r - a2i * t2i; ni = a2r * t2i + a2i * t2r;
            a2r = nr; a2i = ni;
            nr = a3r * t3r - a3i * t3i; ni = a3r * t3i + a3i * t3r;
            a3r = nr; a3i = ni;
        }
        const double s0r = a0r + a2r, s0i = a0i + a2i;
        const double d0r = a0r - a2r, d0i = a0i - a2i;
        const double s1r = a1r + a3r, s1i = a1i + a3i;
        const double d1r = a1r - a3r, d1i = a1i - a3i;
        double y0r = s0r + s1r, y0i = s0i + s1i;
        double y2r = 2.0 * s0r - y0r, y2i = 2.0 * s0i - y0i;
        double y1r, y1i;
        if (sign < 0) { y1r = d0r + d1i; y1i = d0i - d1r; }
        else          { y1r = d0r - d1i; y1i = d0i + d1r; }
        const double y3r = 2.0 * d0r - y1r, y3i = 2.0 * d0i - y1i;
        if (sign < 0) {
            const double t1r = twr[0 * m + c], t1i = sgn * twi[0 * m + c];
            const double t2r = twr[1 * m + c], t2i = sgn * twi[1 * m + c];
            const double t3r = twr[2 * m + c], t3i = sgn * twi[2 * m + c];
            xr[1 * m + c] = y1r * t1r - y1i * t1i; xi[1 * m + c] = y1r * t1i + y1i * t1r;
            xr[2 * m + c] = y2r * t2r - y2i * t2i; xi[2 * m + c] = y2r * t2i + y2i * t2r;
            xr[3 * m + c] = y3r * t3r - y3i * t3i; xi[3 * m + c] = y3r * t3i + y3i * t3r;
        } else {
            xr[1 * m + c] = y1r; xi[1 * m + c] = y1i;
            xr[2 * m + c] = y2r; xi[2 * m + c] = y2i;
            xr[3 * m + c] = y3r; xi[3 * m + c] = y3i;
        }
        xr[0 * m + c] = y0r; xi[0 * m + c] = y0i;
    }
}

inline void base_transform(double* xr, double* xi, std::size_t n, int sign, const TwiddleSet* tw) {
    switch (n) {
        case 1: return;
        case 2: dft2_scalar(xr, xi); return;
        case 4: sign < 0 ? dft4_scalar<-1>(xr, xi) : dft4_scalar<+1>(xr, xi); return;
        case 8: sign < 0 ? dft8_scalar<-1>(xr, xi) : dft8_scalar<+1>(xr, xi); return;
        case 16:
            assert(tw && tw->n == 16);
            if (sign < 0) {
                radix4_column_scalar(xr, xi, 4, tw->wr.data(), tw->wi.data(), sign);
                for (int k = 0; k < 4; ++k) dft4_scalar<-1>(xr + 4 * k, xi + 4 * k);
            } else {
                for (int k = 0; k < 4; ++k) dft4_scalar<+1>(xr + 4 * k, xi + 4 * k);
                radix4_column_scalar(xr, xi, 4, tw->wr.data(), tw->wi.data(), sign);
            }
            return;
        case 32:
            assert(tw && tw->n == 32);
            if (sign < 0) {
                radix4_butterfly_fwd(xr, xi, 8, tw->wr.data(), tw->wi.data());
                for (int k = 0; k < 4; ++k) dft8_scalar<-1>(xr + 8 * k, xi + 8 * k);
            } else {
                for (int k = 0; k < 4; ++k) dft8_scalar<+1>(xr + 8 * k, xi + 8 * k);
                radix4_butterfly_inv(xr, xi, 8, tw->wr.data(), tw->wi.data());
            }
            return;
        default:
            assert(false && "base_transform: unsupported terminal length");
    }
}


// Recursive driver, with opt-in fork/join threading

constexpr std::size_t kMaxLevels = 64;  // generous bound on recursion depth

inline void fft_recursive(double* xr, double* xi, std::size_t L, int sign,
                           const TwiddleSet** tabs, const Options& opts, unsigned threads) {
    if (L <= 32) { base_transform(xr, xi, L, sign, tabs[0]); return; }

    const std::size_t r = fft_radix(L);
    const std::size_t m = L / r;
    const TwiddleSet& tw = *tabs[0];

    auto column_pass = [&] {
        if (r == 4) {
            (sign < 0 ? radix4_butterfly_fwd : radix4_butterfly_inv)(xr, xi, m, tw.wr.data(), tw.wi.data());
        } else {
            (sign < 0 ? radix8_butterfly_fwd : radix8_butterfly_inv)(xr, xi, m, tw.wr.data(), tw.wi.data());
        }
    };

    auto recurse_rows = [&] {
        if (threads > 1 && L >= opts.parallel_threshold) {
            const unsigned per = std::max(1u, threads / static_cast<unsigned>(r));
            std::vector<std::thread> pool;
            pool.reserve(r - 1);
            for (std::size_t k = 1; k < r; ++k) {
                pool.emplace_back(fft_recursive, xr + k * m, xi + k * m, m, sign, tabs + 1,
                                   std::cref(opts), per);
            }
            fft_recursive(xr, xi, m, sign, tabs + 1, opts, per);
            for (auto& t : pool) t.join();
        } else {
            for (std::size_t k = 0; k < r; ++k) {
                fft_recursive(xr + k * m, xi + k * m, m, sign, tabs + 1, opts, 1);
            }
        }
    };

    if (sign < 0) { column_pass(); recurse_rows(); }
    else          { recurse_rows(); column_pass(); }
}

inline bool is_pow2(std::size_t n) noexcept { return n != 0 && (n & (n - 1)) == 0; }

}  // namespace detail


// Low-level in-place API — split real/imaginary arrays, power-of-two only.
// This is the fast path: no allocation beyond the twiddle-table cache, and
// what the bignum-multiplication use case (or anything else chasing every
// last cycle) should call directly.

inline void transform(double* re, double* im, std::size_t n, bool inverse, Options opts = {}) {
    if (!detail::is_pow2(n)) {
        throw std::invalid_argument("yafftl::transform: n must be a power of two");
    }
    if (n <= 1) return;

    const detail::TwiddleSet& tw = detail::get_twiddles(n);

    const detail::TwiddleSet* tabs[detail::kMaxLevels];
    std::size_t cnt = 0;
    std::size_t L = n;
    tabs[cnt++] = &tw;
    while (L > 32) {
        L /= detail::fft_radix(L);
        tabs[cnt++] = &detail::get_twiddles(L);
    }
    assert(cnt <= detail::kMaxLevels);

    detail::fft_recursive(re, im, n, inverse ? +1 : -1, tabs, opts, opts.threads);

    if (inverse) {
        const double inv_n = 1.0 / static_cast<double>(n);
        std::size_t i = 0;
#if YAFFTL_AVX512
        const __m512d vinv = _mm512_set1_pd(inv_n);
        for (; i + 8 <= n; i += 8) {
            _mm512_storeu_pd(re + i, _mm512_mul_pd(_mm512_loadu_pd(re + i), vinv));
            _mm512_storeu_pd(im + i, _mm512_mul_pd(_mm512_loadu_pd(im + i), vinv));
        }
#endif
        for (; i < n; ++i) { re[i] *= inv_n; im[i] *= inv_n; }
    }
}


// Easy high-level API — the front door for people who just want a transform.

// Zero-pads to the next power of two, transforms in place, returns the
// (padded-length) spectrum. Padded length = next_pow2(data.size()).
inline std::vector<std::complex<double>> fft(std::vector<std::complex<double>> data, Options opts = {}) {
    const std::size_t n = next_pow2(data.size());
    data.resize(n, {0.0, 0.0});
    std::vector<double> re(n), im(n);
    for (std::size_t i = 0; i < n; ++i) { re[i] = data[i].real(); im[i] = data[i].imag(); }
    transform(re.data(), im.data(), n, false, opts);
    std::vector<std::complex<double>> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = {re[i], im[i]};
    return out;
}

inline std::vector<std::complex<double>> fft(const std::vector<double>& data, Options opts = {}) {
    std::vector<std::complex<double>> in(data.begin(), data.end());
    return fft(std::move(in), opts);
}

// Inverse of fft(): input length is zero-padded to a power of two too, so
// ifft(fft(x)) round-trips for any x (up to the padded length).
inline std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> data, Options opts = {}) {
    const std::size_t n = next_pow2(data.size());
    data.resize(n, {0.0, 0.0});
    std::vector<double> re(n), im(n);
    for (std::size_t i = 0; i < n; ++i) { re[i] = data[i].real(); im[i] = data[i].imag(); }
    transform(re.data(), im.data(), n, true, opts);
    std::vector<std::complex<double>> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = {re[i], im[i]};
    return out;
}

// Linear (non-circular) convolution of two real signals via zero-padded FFT
// pointwise-multiply-then-inverse. Output length is a.size() + b.size() - 1
// — the library handles the padding/truncation for you.
inline std::vector<double> convolve(const std::vector<double>& a, const std::vector<double>& b,
                                     Options opts = {}) {
    if (a.empty() || b.empty()) return {};
    const std::size_t out_len = a.size() + b.size() - 1;
    const std::size_t n = next_pow2(out_len);

    std::vector<double> ar(n, 0.0), ai(n, 0.0), br(n, 0.0), bi(n, 0.0);
    std::copy(a.begin(), a.end(), ar.begin());
    std::copy(b.begin(), b.end(), br.begin());

    transform(ar.data(), ai.data(), n, false, opts);
    transform(br.data(), bi.data(), n, false, opts);

    for (std::size_t i = 0; i < n; ++i) {
        const double rr = ar[i] * br[i] - ai[i] * bi[i];
        const double ri = ar[i] * bi[i] + ai[i] * br[i];
        ar[i] = rr; ai[i] = ri;
    }

    transform(ar.data(), ai.data(), n, true, opts);
    ar.resize(out_len);
    return ar;
}

}  // namespace yafftl

#endif  // YAFFTL_HPP
