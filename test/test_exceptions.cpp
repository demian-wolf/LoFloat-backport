// Tests IEEE-754 exception-flag signaling (no trapping).
//
// Build with -DENABLE_EXCEPT (the Makefile target does this). The flags are only
// raised for formats whose NA_behavior == _754; a parallel _3109 format is used
// to confirm the gate stays silent.
#include <cmath>
#include <cstdint>
#include <iostream>
#include "lo_float.h"
#include "lo_float_sci.hpp"

using namespace lo_float;
namespace lfi = lo_float::lo_float_internal;

// ---- P3109 inf/NaN checkers (k = 8, signed) -------------------------------

using P3109Inf_Ext = lfi::P_3109_InfChecker<8, Signedness::Signed, Inf_Behaviors::Extended>;
using P3109Inf_Sat = lfi::P_3109_InfChecker<8, Signedness::Signed, Inf_Behaviors::Saturating>;
using P3109NaN     = lfi::P_3109_NaNChecker<8, Signedness::Signed>;

constexpr FloatingPointParams param_754(
    8, 2, 15, Inf_Behaviors::Extended, NaN_Behaviors::_754,
    Signedness::Signed, P3109Inf_Ext(), P3109NaN());
constexpr FloatingPointParams param_3109(
    8, 2, 15, Inf_Behaviors::Extended, NaN_Behaviors::_3109,
    Signedness::Signed, P3109Inf_Ext(), P3109NaN());

// Saturating variant (Inf_Behaviors::Saturating): the target has no infinities,
// so inf inputs must clamp to +/-max rather than propagate.
constexpr FloatingPointParams param_sat(
    8, 2, 15, Inf_Behaviors::Saturating, NaN_Behaviors::_3109,
    Signedness::Signed, P3109Inf_Sat(), P3109NaN());

using f754  = Templated_Float<param_754>;
using f3109 = Templated_Float<param_3109>;
using fsat  = Templated_Float<param_sat>;

static int failures = 0;

static bool has(uint8_t f, lfi::LF_exception_flags bit) {
    return (f & static_cast<uint8_t>(bit)) != 0;
}

// Run `body`, then assert the named flag is (raised==true) raised afterwards.
template <typename F>
static void expect(const char* name, lfi::LF_exception_flags bit, bool raised, F&& body) {
    lfi::f_env.reset_exception_flags();
    body();
    const uint8_t f = lfi::f_env.get_exception_flags();
    const bool ok = has(f, bit) == raised;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name
              << " (flags=0x" << std::hex << (int)f << std::dec << ")\n";
    if (!ok) ++failures;
}

// Assert NO flags at all were raised.
template <typename F>
static void expect_none(const char* name, F&& body) {
    lfi::f_env.reset_exception_flags();
    body();
    const uint8_t f = lfi::f_env.get_exception_flags();
    const bool ok = (f == 0);
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name
              << " (flags=0x" << std::hex << (int)f << std::dec << ")\n";
    if (!ok) ++failures;
}

// Enum -> name, reusing the same X-macro lists the enums are generated from.
static const char* rm_name(Rounding_Mode m) {
    switch (m) {
        #define X(n) case Rounding_Mode::n: return #n;
        rounding_modes
        #undef X
        default: return "?";
    }
}
static const char* sm_name(Saturation_Mode m) {
    switch (m) {
        #define X(n) case Saturation_Mode::n: return #n;
        saturation_modes
        #undef X
        default: return "?";
    }
}

// Two values agree if equal, or both NaN.
static bool same(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return a == b;
}

// The deterministic rounding modes: every path computes the SAME result for
// these, so they can be cross-checked. The stochastic modes are excluded here
// (scalar and SIMD draw independent samples, so they legitimately diverge).
static constexpr Rounding_Mode kDetModes[] = {
    Rounding_Mode::RoundToNearestEven, Rounding_Mode::RoundToNearestOdd,
    Rounding_Mode::RoundTowardsZero,   Rounding_Mode::RoundAwayFromZero,
    Rounding_Mode::RoundUp,            Rounding_Mode::RoundDown,
    Rounding_Mode::RoundTiesToAway,    Rounding_Mode::RoundTiesTowardsZero,
    Rounding_Mode::RoundToOdd,
};
static constexpr Saturation_Mode kSatModes[] = {
    Saturation_Mode::OvfInf, Saturation_Mode::SatFinite, Saturation_Mode::SatPropagate,
};

// For one typed format T (with matching FloatingPointParams `param`) and one
// ProjSpec, confirm the three OTHER rounding paths agree with the scalar
// reference Round<T>(x, ps) on every probed element:
//   * SIMD real round    -- Project<T> over the whole array
//   * scalar virtual_round(x, param, ps)
//   * vector virtual_round(array, param, ps)
// Returns the number of mismatches; prints a line per mismatch.
template <typename T, typename Param>
static int check_paths(const char* fmt, float* src, int kProbed,
                       const Param& param, ProjSpec ps) {
    constexpr int N = 64;
    alignas(64) T     dst[N];
    alignas(64) float vdst[N];
    lo_float::Project<T>(src, dst, N, ps);              // SIMD real round
    lo_float::virtual_round(src, vdst, N, param, ps);   // vector virtual_round

    int bad = 0;
    auto report = [&](const char* path, int i, double got, double ref) {
        std::cout << "[FAIL] " << fmt << " " << path
                  << " rm=" << rm_name(ps.rounding_mode)
                  << " sm=" << sm_name(ps.saturation_mode)
                  << " src[" << i << "]=" << (double)src[i]
                  << " got=" << got << " ref=" << ref << "\n";
        ++bad;
    };
    for (int i = 0; i < kProbed; ++i) {
        const double ref = (double)lo_float::Round<T>(src[i], ps);
        if (!same((double)dst[i], ref))                             report("simd-real",   i, (double)dst[i], ref);
        if (!same((double)lo_float::virtual_round(src[i], param, ps), ref))
                                                                    report("scalar-virt", i, (double)lo_float::virtual_round(src[i], param, ps), ref);
        if (!same((double)vdst[i], ref))                            report("vector-virt", i, (double)vdst[i], ref);
    }
    return bad;
}

// Oracle for Round<T>'s saturation handling: an overflowing finite (1e30) and a
// true infinity, both signs, across the three saturation modes. Mirrors the
// semantics validated in test_sat_mode:
//   SatFinite  -> always the finite max.
//   OvfInf     -> inf when the format has infinities, else max.
//   SatPropagate -> a true inf propagates (if the format has inf); a finite
//                   overflow still clamps to max.
template <typename T>
static int round_sat_oracle(const char* fmt) {
    const bool   has_inf = std::numeric_limits<T>::has_infinity;
    const double maxf    = (double)std::numeric_limits<T>::max();
    const double inf     = (double)std::numeric_limits<T>::infinity();
    const float  finf    = std::numeric_limits<float>::infinity();
    int bad = 0;
    auto one = [&](const char* what, float x, Saturation_Mode sm, double want) {
        const double got = (double)lo_float::Round<T>(x, ProjSpec(Rounding_Mode::RoundToNearestEven, sm));
        if (!same(got, want)) {
            std::cout << "[FAIL] Round<" << fmt << "> " << what
                      << " sm=" << sm_name(sm) << " x=" << (double)x
                      << " got=" << got << " want=" << want << "\n";
            ++bad;
        }
    };
    for (double s : {+1.0, -1.0}) {
        const float  xovf  = (float)(s * 1.0e30);
        const float  xinf  = (float)(s * (double)finf);
        const double wmax  = s * maxf;
        const double winf  = s * inf;
        // Overflowing finite input.
        one("ovf", xovf, Saturation_Mode::SatFinite,    wmax);
        one("ovf", xovf, Saturation_Mode::OvfInf,       has_inf ? winf : wmax);
        one("ovf", xovf, Saturation_Mode::SatPropagate, wmax);   // finite overflow clamps
        // Infinite input.
        one("inf", xinf, Saturation_Mode::SatFinite,    wmax);
        one("inf", xinf, Saturation_Mode::OvfInf,       has_inf ? winf : wmax);
        one("inf", xinf, Saturation_Mode::SatPropagate, has_inf ? winf : wmax);
    }
    return bad;
}

int main() {
    using IF = lfi::LF_exception_flags;
    volatile double sink = 0.0;  // keep results live

    const f754 one   = f754(1.0);
    const f754 zero  = f754(0.0);
    const f754 inf_  = std::numeric_limits<f754>::infinity();

    // ---- division ---------------------------------------------------------
    expect("finite/0 -> DivisionByZero", IF::DivisionByZero, true,
           [&]{ f754 r = one / zero; sink = (double)r; });
    expect("finite/0 NOT InvalidOperation", IF::InvalidOperation, false,
           [&]{ f754 r = one / zero; sink = (double)r; });
    expect("0/0 -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ f754 r = zero / zero; sink = (double)r; });
    expect("0/0 NOT DivisionByZero", IF::DivisionByZero, false,
           [&]{ f754 r = zero / zero; sink = (double)r; });

    // ---- invalid arithmetic ----------------------------------------------
    expect("inf - inf -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ f754 r = inf_ - inf_; sink = (double)r; });
    expect("0 * inf -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ f754 r = zero * inf_; sink = (double)r; });

    // ---- overflow / underflow on conversion ------------------------------
    expect("overflow on Project(1e30) -> Overflow", IF::Overflow, true,
           [&]{ f754 r = Project<f754>(1.0e30); sink = (double)r; });
    expect("tiny value -> Underflow", IF::Underflow, true,
           [&]{ f754 r = Project<f754>(std::ldexp(1.0, -15)); sink = (double)r; });

    // ---- float -> int ----------------------------------------------------
    expect("inf -> int -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ int v = (int)inf_; sink = v; });

    // ---- transcendentals (lo_float_sci.hpp) ------------------------------
    expect("sqrt(-1) -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ auto r = sqrt(f754(-1.0)); sink = (double)r; });
    expect("acosh(0) -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ auto r = acosh(f754(0.0)); sink = (double)r; });
    expect("log(-1) -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ auto r = log(f754(-1.0)); sink = (double)r; });
    expect("log2(0) -> DivisionByZero", IF::DivisionByZero, true,
           [&]{ auto r = log2(zero); sink = (double)r; });
    expect("atanh(1) -> DivisionByZero", IF::DivisionByZero, true,
           [&]{ auto r = atanh(one); sink = (double)r; });
    expect("rSqrt(0) -> DivisionByZero", IF::DivisionByZero, true,
           [&]{ auto r = rSqrt(zero); sink = (double)r; });

    // ---- fma -------------------------------------------------------------
    expect("fma(0, inf, 1) -> InvalidOperation", IF::InvalidOperation, true,
           [&]{ auto r = fma<param_754, param_754, param_754, param_754>(zero, inf_, one);
                sink = (double)r; });
    expect_none("fma(2, 3, 1) raises nothing",
           [&]{ auto r = fma<param_754, param_754, param_754, param_754>(f754(2.0), f754(3.0), one);
                sink = (double)r; });

    // ---- value sanity for a few new transcendentals ----------------------
    {
        const double got = (double)log10(f754(100.0));
        const bool ok = std::abs(got - 2.0) < 0.2;   // coarse: e5m2 is low precision
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << "log10(100) ~= 2 (got " << got << ")\n";
        if (!ok) ++failures;
    }

    // ---- gate: a _3109 format must raise NOTHING -------------------------
    const f3109 q_one  = f3109(1.0);
    const f3109 q_zero = f3109(0.0);
    expect_none("_3109 finite/0 raises nothing",
           [&]{ f3109 r = q_one / q_zero; sink = (double)r; });
    expect_none("_3109 sqrt(-1) raises nothing",
           [&]{ auto r = sqrt(f3109(-1.0)); sink = (double)r; });
    expect_none("_3109 overflow raises nothing",
           [&]{ f3109 r = Project<f3109>(1.0e30); sink = (double)r; });

    // ======================================================================
    //  Cross-path agreement, swept over the whole ProjSpec.
    //
    //  The scalar real round Round<T>(x, ps) is the reference. For EVERY
    //  deterministic rounding mode x EVERY saturation mode (kDetModes x
    //  kSatModes) and ALL THREE target formats -- Extended _754, Extended _3109
    //  and Saturating -- we confirm the other three paths (SIMD real Project,
    //  scalar virtual_round, vector virtual_round) reproduce it on +/-inf, NaN
    //  and a spread of finite normals (both signs).
    //
    //  Previously this only ran the default ProjSpec and skipped param_3109
    //  entirely, so the saturation/rounding-mode plumbing of the vector paths
    //  went untested.
    // ======================================================================
    {
        constexpr int N = 64;                          // multiple of every SIMD block
        const float finf = std::numeric_limits<float>::infinity();
        const float fnan = std::numeric_limits<float>::quiet_NaN();

        alignas(64) float src[N];
        for (int i = 0; i < N; ++i) src[i] = 1.0f;     // benign finite filler
        src[0] = finf; src[1] = -finf; src[2] = fnan;
        src[3] = 3.0f; src[4] = -3.0f; src[5] = 0.5f; src[6] = 2.0f;
        src[7] = 1.0e30f; src[8] = -1.0e30f;           // overflow (exercise saturation)
        const int kProbed = 9;

        // Silence: the three non-reference paths must never raise a flag, for
        // any ProjSpec (they have no ENABLE_EXCEPT hooks). Round<T> is NOT run
        // here -- for a _754 format it legitimately signals Overflow/Invalid.
        expect_none("non-reference paths stay silent across all ProjSpecs", [&]{
            alignas(64) f754  d754[N];
            alignas(64) f3109 d3109[N];
            alignas(64) fsat  dsat[N];
            alignas(64) float v[N];
            for (Rounding_Mode rm : kDetModes)
                for (Saturation_Mode sm : kSatModes) {
                    const ProjSpec ps(rm, sm);
                    Project<f754>(src, d754, N, ps);
                    Project<f3109>(src, d3109, N, ps);
                    Project<fsat>(src, dsat, N, ps);
                    lo_float::virtual_round(src, v, N, param_754,  ps);
                    lo_float::virtual_round(src, v, N, param_3109, ps);
                    lo_float::virtual_round(src, v, N, param_sat,  ps);
                    sink = (double)lo_float::virtual_round(finf, param_754, ps);
                    sink = (double)lo_float::virtual_round(fnan, param_sat, ps);
                }
        });

        int bad = 0, combos = 0;
        for (Rounding_Mode rm : kDetModes)
            for (Saturation_Mode sm : kSatModes) {
                const ProjSpec ps(rm, sm);
                bad += check_paths<f754>("754",  src, kProbed, param_754,  ps);
                bad += check_paths<f3109>("3109", src, kProbed, param_3109, ps);
                bad += check_paths<fsat>("sat",  src, kProbed, param_sat,  ps);
                ++combos;
            }
        std::cout << (bad == 0 ? "[PASS] " : "[FAIL] ")
                  << "all paths agree with Round<T> over " << combos
                  << " ProjSpecs x 3 formats (" << bad << " mismatches)\n";
        failures += bad;
    }

    // ======================================================================
    //  Direct oracle for the real scalar round Round<T>(x, ps) itself.
    //
    //  The block above uses Round<T> as its reference; this block instead pins
    //  Round<T> to independently-known answers while sweeping the saturation
    //  side of the ProjSpec: an overflowing finite and a true infinity resolve
    //  per the saturation mode and whether the format has infinities.
    //  (Rounding-mode / tie-breaking coverage lives in test_rounding_modes.)
    // ======================================================================
    {
        int bad = 0;
        bad += round_sat_oracle<f754>("754");
        bad += round_sat_oracle<f3109>("3109");
        bad += round_sat_oracle<fsat>("sat");
        std::cout << (bad == 0 ? "[PASS] " : "[FAIL] ")
                  << "Round<T> honours every saturation mode (oracle, "
                  << bad << " mismatches)\n";
        failures += bad;
    }

    (void)sink;
    if (failures == 0) {
        std::cout << "All tests passed\n";
        return 0;
    }
    std::cout << failures << " test(s) FAILED\n";
    return 1;
}
