// -----------------------------------------------------------------------------
// test_edge_cases.cpp — IEEE P3109 special-value / edge-case behaviour.
//
// Companion to test_exceptions.cpp (which covers the IEEE-754 exception-flag
// side).  This file exercises the P3109 *value* semantics of the functions in
// lo_float.h / lo_float_sci.hpp — classification, extrema, ordering, next-value,
// format queries, Recip / CopySign / Softplus / fma, and the transcendental
// NaN/pole edge cases — against an INDEPENDENT double-precision oracle derived
// from loop/edge_cases_3109.md.
//
// Harness style mirrors test_rounding_modes.cpp / test_sat_mode.cpp: every P3109
// format is instantiated through the same <l,p,Signedness,Inf_Behaviors> template
// machinery, folded over l = 3..8, p = 2..l-1, for both Signed domains
// (Saturating = Finite datum set, Extended = ±inf in the datum set).  A test
// returns an error count; 0 = pass, nonzero prints the failing case and fails.
//
// Scope notes (per the backlog item):
//   * Block operations and FAA are intentionally NOT covered.
//   * Only *Signed* P3109 formats are exercised — the existing suite
//     (test_rounding_modes, test_sat_mode) is Signed-only, and Unsigned P3109 is
//     an unsupported corner (numeric_limits<>::max() returns NaN there; see
//     loop/NOTES.md).  The 754 edge cases live in test_exceptions.cpp.
//   * The core arithmetic operators (+,-,*,/) are IEEE-on-double by design (see
//     operator/ in lo_float.h: finite/0 -> ±inf, not P3109 NaN), so their
//     special-value behaviour is verified where it is well-defined; the P3109
//     divide-by-zero-> NaN semantics live in the dedicated Recip function.
// -----------------------------------------------------------------------------
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>
#include <string>
#include <algorithm>
#include <utility>

#include "lo_float.h"
#include "lo_float_sci.hpp"

using namespace lo_float;

// A running failure count with a compact reporter.
static int g_fail = 0;
static void fail(const std::string& msg) {
    std::printf("[FAIL] %s\n", msg.c_str());
    ++g_fail;
}

// Two doubles agree if equal, or both NaN (special values compare by class).
static bool same(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return a == b;
}

// ============================================================================
//  Per-format edge-case suite.
//
//  T   = P_3109_float<l,p,Signed,D>          (the device-under-test type)
//  Fp  = its FloatingPointParams NTTP        (for the *<Fp>(...) call form)
// ============================================================================
template <int l, int p, Inf_Behaviors D>
int test_edge_3109() {
    using T = P_3109_float<l, p, Signedness::Signed, D>;
    constexpr auto Fp =
        lo_float_internal::param_float_p_3109<l, p, Signedness::Signed, D>;
    using Lim = std::numeric_limits<T>;
    using Rep = decltype(std::declval<T>().rep());

    constexpr bool ext = (D == Inf_Behaviors::Extended);
    const int    reps  = 1 << l;                 // l <= 8  ->  <= 256 code points
    char tag[64];
    std::snprintf(tag, sizeof(tag), "P3109<%d,%d,%s>", l, p, ext ? "Ext" : "Sat");

    const int  mant   = p - 1;
    const double dmin = (double)Lim::denorm_min();
    const double mnrm = (double)Lim::min();       // smallest normal
    const double dmax = (double)Lim::max();
    // True most-negative finite of a SIGNED format is -max, computed
    // independently of numeric_limits::lowest() (which is itself under test —
    // see the §4.14 MinFiniteOf check and the BACKLOG "lowest() Saturating" bug).
    const double dlow = -dmax;
    const T      qNaN = Lim::quiet_NaN();
    const T      zero = T(0.0f);
    const T      one  = T(1.0f);
    int e = 0;

    // Handy: (double) of a candidate value, and whether the format can hold it.
    auto RT = [&](double x) { return (double)T(static_cast<float>(x)); };

    // ------------------------------------------------------------------------
    // §4.13 / §4.13.1 — classification predicates + Class, EXHAUSTIVE over reps.
    //
    // Oracle is the decoded double value + the format's normal threshold, all
    // independent of the predicate implementations.
    // ------------------------------------------------------------------------
    for (int r = 0; r < reps; ++r) {
        T x = T::FromRep(static_cast<Rep>(r));
        const double v = (double)x;

        const bool o_nan = std::isnan(v);
        const bool o_inf = std::isinf(v);
        const bool o_fin = std::isfinite(v);
        const bool o_zero = (!o_nan && v == 0.0);
        const bool o_one  = (!o_nan && !o_inf && v == 1.0);
        const bool o_neg  = (!o_nan && v < 0.0);                 // -0 does not exist
        const bool o_nrm  = (o_fin && !o_zero && std::fabs(v) >= mnrm);
        const bool o_sub  = (o_fin && !o_zero && std::fabs(v) <  mnrm);

        auto chk = [&](const char* nm, bool got, bool want) {
            if (got != want) {
                char b[128];
                std::snprintf(b, sizeof(b), "%s %s rep=0x%x v=%g got=%d want=%d",
                              tag, nm, r, v, got, want);
                fail(b); ++e;
            }
        };
        chk("IsNaN",      IsNaN<Fp>(x),      o_nan);
        chk("IsInfinite", IsInfinite<Fp>(x), o_inf);
        chk("IsFinite",   IsFinite<Fp>(x),   o_fin);
        chk("IsZero",     IsZero<Fp>(x),     o_zero);
        chk("IsOne",      IsOne<Fp>(x),      o_one);
        chk("IsSignMinus",IsSignMinus<Fp>(x),o_neg);
        chk("IsNormal",   IsNormal<Fp>(x),   o_nrm);
        chk("IsSubnormal",IsSubnormal<Fp>(x),o_sub);

        // Class must map to exactly the one class implied by the oracle.
        Class_Enum want_cls =
            o_nan  ? ClsNaN :
            o_inf  ? (o_neg ? ClsNegativeInfinity : ClsPositiveInfinity) :
            o_zero ? ClsZero :
            o_neg  ? (o_sub ? ClsNegativeSubnormal : ClsNegativeNormal) :
                     (o_sub ? ClsPositiveSubnormal : ClsPositiveNormal);
        if (Class<Fp>(x) != want_cls) {
            char b[128];
            std::snprintf(b, sizeof(b), "%s Class rep=0x%x v=%g got=%d want=%d",
                          tag, r, v, (int)Class<Fp>(x), (int)want_cls);
            fail(b); ++e;
        }
    }

    // ------------------------------------------------------------------------
    // Build the "interesting operands" set for pairwise tests.  ±inf only when
    // the format actually has them (Extended).
    // ------------------------------------------------------------------------
    std::vector<T> ops = {
        qNaN, zero, one, T(-1.0f), T(2.0f), T(-2.0f),
        T(static_cast<float>(dmax)),  T(static_cast<float>(dlow)),
        T(static_cast<float>(mnrm)),  T(static_cast<float>(-mnrm)),
        T(static_cast<float>(dmin)),  T(static_cast<float>(-dmin)),
    };
    if constexpr (ext) {
        ops.push_back(Lim::infinity());
        ops.push_back(-Lim::infinity());
    }

    // ------------------------------------------------------------------------
    // §4.11 — extrema family, and §4.11.4 Clamp.  Oracle applies the doc's
    // NaN / ±inf / tie rules directly in double.
    // ------------------------------------------------------------------------
    {
        auto dbl = [](T v) { return (double)v; };
        for (T xt : ops) for (T yt : ops) {
            const double x = dbl(xt), y = dbl(yt);
            const bool nx = std::isnan(x), ny = std::isnan(y);

            // Minimum / Maximum: NaN-propagating; tie -> y (min) / x (max).
            double o_min = (nx || ny) ? std::nan("") : (x < y ? x : y);
            double o_max = (nx || ny) ? std::nan("") : (x < y ? y : x);
            // *Number: a lone NaN is ignored; (NaN,NaN) -> NaN.
            double o_minN = nx ? (ny ? std::nan("") : y) : (ny ? x : (x < y ? x : y));
            double o_maxN = nx ? (ny ? std::nan("") : y) : (ny ? x : (x < y ? y : x));
            // Magnitude variants: compare |x|,|y|; tie -> lesser (min) / greater (max).
            double o_minM, o_maxM;
            if (nx || ny) { o_minM = o_maxM = std::nan(""); }
            else {
                const double ax = std::fabs(x), ay = std::fabs(y);
                o_minM = (ax < ay) ? x : (ay < ax) ? y : (x < y ? x : y);
                o_maxM = (ax > ay) ? x : (ay > ax) ? y : (x < y ? y : x);
            }
            double o_minMN = nx ? (ny ? std::nan("") : y) : (ny ? x : o_minM);
            double o_maxMN = nx ? (ny ? std::nan("") : y) : (ny ? x : o_maxM);

            auto ck = [&](const char* nm, double got, double want) {
                if (!same(got, want)) {
                    char b[160];
                    std::snprintf(b, sizeof(b), "%s %s x=%g y=%g got=%g want=%g",
                                  tag, nm, x, y, got, want);
                    fail(b); ++e;
                }
            };
            ck("Minimum",               dbl(Minimum<Fp>(xt, yt)),               o_min);
            ck("Maximum",               dbl(Maximum<Fp>(xt, yt)),               o_max);
            ck("MinimumNumber",         dbl(MinimumNumber<Fp>(xt, yt)),         o_minN);
            ck("MaximumNumber",         dbl(MaximumNumber<Fp>(xt, yt)),         o_maxN);
            ck("MinimumMagnitude",      dbl(MinimumMagnitude<Fp>(xt, yt)),      o_minM);
            ck("MaximumMagnitude",      dbl(MaximumMagnitude<Fp>(xt, yt)),      o_maxM);
            ck("MinimumMagnitudeNumber",dbl(MinimumMagnitudeNumber<Fp>(xt, yt)),o_minMN);
            ck("MaximumMagnitudeNumber",dbl(MaximumMagnitudeNumber<Fp>(xt, yt)),o_maxMN);
            // MinimumFinite / MaximumFinite are the NaN-ignoring (*Number) forms.
            ck("MinimumFinite",         dbl(MinimumFinite<Fp>(xt, yt)),         o_minN);
            ck("MaximumFinite",         dbl(MaximumFinite<Fp>(xt, yt)),         o_maxN);
        }

        // Clamp over a smaller triple set, including inverted bounds and NaN.
        std::vector<T> cs = { qNaN, T(-2.0f), zero, one, T(2.0f),
                              T(static_cast<float>(dmax)), T(static_cast<float>(dlow)) };
        for (T xt : cs) for (T lot : cs) for (T hit : cs) {
            const double x = (double)xt, lo = (double)lot, hi = (double)hit;
            double want;
            if (std::isnan(x) || std::isnan(lo) || std::isnan(hi)) want = std::nan("");
            else if (lo > hi) want = std::nan("");                 // inverted bounds
            else want = (x < lo) ? lo : (x > hi ? hi : x);
            const double got = (double)Clamp<Fp>(xt, lot, hit);
            if (!same(got, want)) {
                char b[176];
                std::snprintf(b, sizeof(b), "%s Clamp x=%g lo=%g hi=%g got=%g want=%g",
                              tag, x, lo, hi, got, want);
                fail(b); ++e;
            }
        }
    }

    // ------------------------------------------------------------------------
    // §4.12 — ordering: TotalOrder and the comparison operators.  NaN is
    // unordered (all comparisons false, != true); TotalOrder sinks NaN below all.
    // ------------------------------------------------------------------------
    for (T xt : ops) for (T yt : ops) {
        const double x = (double)xt, y = (double)yt;
        const bool nx = std::isnan(x), ny = std::isnan(y);
        const bool anynan = nx || ny;

        auto ckb = [&](const char* nm, bool got, bool want) {
            if (got != want) {
                char b[160];
                std::snprintf(b, sizeof(b), "%s %s x=%g y=%g got=%d want=%d",
                              tag, nm, x, y, got, want);
                fail(b); ++e;
            }
        };
        ckb("op<",  (xt <  yt), anynan ? false : (x <  y));
        ckb("op<=", (xt <= yt), anynan ? false : (x <= y));
        ckb("op>",  (xt >  yt), anynan ? false : (x >  y));
        ckb("op>=", (xt >= yt), anynan ? false : (x >= y));
        ckb("op==", (xt == yt), anynan ? false : (x == y));
        ckb("op!=", (xt != yt), anynan ? true  : (x != y));

        // TotalOrder: NaN(x)->true; NaN(y) (x not NaN)->false; else x<=y.
        bool o_to = nx ? true : (ny ? false : (x <= y));
        ckb("TotalOrder", TotalOrder<Fp>(xt, yt), o_to);
    }

    // ------------------------------------------------------------------------
    // §4.16 — NextGreaterThan / NextLessThan, EXHAUSTIVE.
    //
    // Independent oracle: build the totally-ordered set of decoded values
    // (all finite code points, plus ±inf for Extended), deduped.  The next /
    // previous element in that ordered set is the answer; the top/bottom element
    // has no successor / predecessor and yields NaN.  NaN input -> NaN.
    // ------------------------------------------------------------------------
    {
        std::vector<double> vals;
        for (int r = 0; r < reps; ++r) {
            double v = (double)T::FromRep(static_cast<Rep>(r));
            if (std::isnan(v)) continue;      // NaN is outside the order
            vals.push_back(v);
        }
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());  // single 0

        auto next_of = [&](double v, bool up) -> double {
            // locate v in the ordered set
            for (size_t i = 0; i < vals.size(); ++i) {
                if (vals[i] == v) {
                    if (up)  return (i + 1 < vals.size()) ? vals[i + 1] : std::nan("");
                    else     return (i > 0)               ? vals[i - 1] : std::nan("");
                }
            }
            return std::nan("");   // unreachable for finite code points
        };

        for (int r = 0; r < reps; ++r) {
            T x = T::FromRep(static_cast<Rep>(r));
            const double v = (double)x;
            const bool nan = std::isnan(v);

            const double o_ngt = nan ? std::nan("") : next_of(v, true);
            const double o_nlt = nan ? std::nan("") : next_of(v, false);
            const double g_ngt = (double)NextGreaterThan<Fp>(x);
            const double g_nlt = (double)NextLessThan<Fp>(x);
            if (!same(g_ngt, o_ngt)) {
                char b[144];
                std::snprintf(b, sizeof(b), "%s NextGreaterThan rep=0x%x v=%g got=%g want=%g",
                              tag, r, v, g_ngt, o_ngt);
                fail(b); ++e;
            }
            if (!same(g_nlt, o_nlt)) {
                char b[144];
                std::snprintf(b, sizeof(b), "%s NextLessThan rep=0x%x v=%g got=%g want=%g",
                              tag, r, v, g_nlt, o_nlt);
                fail(b); ++e;
            }
        }
    }

    // ------------------------------------------------------------------------
    // §4.14 — format-level queries.  Oracle = the (l,p,D) template params.
    // ------------------------------------------------------------------------
    {
        auto cki = [&](const char* nm, long got, long want) {
            if (got != want) {
                char b[128];
                std::snprintf(b, sizeof(b), "%s %s got=%ld want=%ld", tag, nm, got, want);
                fail(b); ++e;
            }
        };
        const long bias = 1L << (l - p - 1);
        cki("BitwidthOf",                   BitwidthOf<Fp>(),                   l);
        cki("PrecisionOf",                  PrecisionOf<Fp>(),                  p);
        cki("TrailingSignificandBitwidthOf",TrailingSignificandBitwidthOf<Fp>(),mant);
        cki("ExponentBitwidthOf",           ExponentBitwidthOf<Fp>(),           l - mant - 1);
        cki("ExponentBiasOf",               ExponentBiasOf<Fp>(),               bias);
        cki("SignednessOf", (long)SignednessOf<Fp>(), (long)Signedness::Signed);
        cki("DomainOf",     (long)DomainOf<Fp>(),     (long)D);
        auto ckv = [&](const char* nm, double got, double want) {
            if (!same(got, want)) {
                char b[128];
                std::snprintf(b, sizeof(b), "%s %s got=%g want=%g", tag, nm, got, want);
                fail(b); ++e;
            }
        };
        ckv("MaxFiniteOf",  (double)MaxFiniteOf<Fp>(),  dmax);
        ckv("MinFiniteOf",  (double)MinFiniteOf<Fp>(),  dlow);
        ckv("MinPositiveOf",(double)MinPositiveOf<Fp>(),dmin);
        ckv("MinNormalOf",  (double)MinNormalOf<Fp>(),  mnrm);
        // MaxSubnormalOf: NaN when the format has no trailing significand bits.
        const double o_msub = (mant == 0) ? std::nan("")
                            : (double)T::FromRep(static_cast<Rep>((Rep{1} << mant) - 1));
        ckv("MaxSubnormalOf", (double)MaxSubnormalOf<Fp>(), o_msub);
    }

    // ------------------------------------------------------------------------
    // §4.10 — Recip / CopySign / Softplus / fma special values, swept across the
    // three saturation modes (satisfies "include the different Saturation modes").
    // ------------------------------------------------------------------------
    {
        const Saturation_Mode sms[3] = { Saturation_Mode::OvfInf,
                                         Saturation_Mode::SatFinite,
                                         Saturation_Mode::SatPropagate };
        for (Saturation_Mode sm : sms) {
            const ProjSpec ps(Rounding_Mode::RoundToNearestEven, sm);

            // How a real result R projects into T under (sm, domain).  finite
            // in-range -> round; overflow / inf resolve per test_sat_mode rules.
            auto project_oracle = [&](double R) -> double {
                if (std::isnan(R)) return std::nan("");
                const bool r_inf = std::isinf(R);
                if (!r_inf && R <= dmax && R >= dlow) return RT(R);        // in range
                const double sgnmax = (R < 0.0) ? dlow : dmax;
                const double dinf = std::numeric_limits<double>::infinity();
                if (!r_inf) {  // finite overflow
                    if (sm == Saturation_Mode::OvfInf)
                        return ext ? (R < 0 ? -dinf : dinf) : sgnmax;
                    return sgnmax;                                          // SatFinite/SatPropagate clamp
                }
                // infinite R
                if (sm == Saturation_Mode::SatFinite) return sgnmax;
                return ext ? R : sgnmax;                                    // OvfInf/SatProp keep inf if Ext
            };

            // Recip: NaN->NaN, 0->NaN, ±inf->0, else project(1/x).
            for (T xt : ops) {
                const double x = (double)xt;
                double want;
                if (std::isnan(x))      want = std::nan("");
                else if (x == 0.0)      want = std::nan("");
                else if (std::isinf(x)) want = 0.0;
                else                    want = project_oracle(1.0 / x);
                const double got = (double)Recip<Fp>(xt, ps);
                if (!same(got, want)) {
                    char b[160];
                    std::snprintf(b, sizeof(b), "%s Recip sm=%d x=%g got=%g want=%g",
                                  tag, (int)sm, x, got, want);
                    fail(b); ++e;
                }
            }

            // CopySign: NaN in either -> NaN; else |x| with sign of y (y==0 -> +).
            for (T xt : ops) for (T yt : ops) {
                const double x = (double)xt, y = (double)yt;
                double want;
                if (std::isnan(x) || std::isnan(y)) want = std::nan("");
                else {
                    const double mag = std::fabs(x);            // |±inf| = +inf
                    want = project_oracle((y < 0.0) ? -mag : mag);
                }
                const double got = (double)CopySign<Fp>(xt, yt, ps);
                if (!same(got, want)) {
                    char b[176];
                    std::snprintf(b, sizeof(b), "%s CopySign sm=%d x=%g y=%g got=%g want=%g",
                                  tag, (int)sm, x, y, got, want);
                    fail(b); ++e;
                }
            }

            // Softplus special values: NaN->NaN, +inf->+inf, -inf->0.
            {
                if (!IsNaN<Fp>(Softplus<Fp>(qNaN, ps))) { fail(std::string(tag) + " Softplus(NaN)!=NaN"); ++e; }
                if constexpr (ext) {
                    const double sp_pinf = (double)Softplus<Fp>(Lim::infinity(), ps);
                    const double w_pinf  = project_oracle(std::numeric_limits<double>::infinity());
                    if (!same(sp_pinf, w_pinf)) { fail(std::string(tag) + " Softplus(+inf)"); ++e; }
                    if (!same((double)Softplus<Fp>(-Lim::infinity(), ps), 0.0)) { fail(std::string(tag) + " Softplus(-inf)!=0"); ++e; }
                }
            }
        }

        // fma special values (default ProjSpec).  0*inf -> NaN, inf-inf -> NaN,
        // NaN in any slot -> NaN, and a finite case rounds a*b+c.
        {
            auto is_nan_fma = [&](T a, T b, T c) {
                return IsNaN<Fp>(fma<Fp, Fp, Fp, Fp>(a, b, c));
            };
            if (!is_nan_fma(qNaN, one, one))  { fail(std::string(tag) + " fma(NaN,1,1)!=NaN"); ++e; }
            if (!is_nan_fma(one, qNaN, one))  { fail(std::string(tag) + " fma(1,NaN,1)!=NaN"); ++e; }
            if (!is_nan_fma(one, one, qNaN))  { fail(std::string(tag) + " fma(1,1,NaN)!=NaN"); ++e; }
            if constexpr (ext) {
                if (!is_nan_fma(zero, Lim::infinity(), one)) { fail(std::string(tag) + " fma(0,inf,1)!=NaN"); ++e; }
                if (!is_nan_fma(Lim::infinity(), zero, one)) { fail(std::string(tag) + " fma(inf,0,1)!=NaN"); ++e; }
                if (!is_nan_fma(Lim::infinity(), one, -Lim::infinity())) { fail(std::string(tag) + " fma(inf,1,-inf)!=NaN"); ++e; }
            }
            // finite: a*b+c with ONE rounding.  Derive the oracle from the
            // actually-rounded operand values (not the literal 2,3,1) — on a
            // 0-mantissa format 3 rounds to 2, so the true product differs.
            const double a = (double)T(2.0f), b = (double)T(3.0f), c = (double)one;
            const double fin  = (double)fma<Fp, Fp, Fp, Fp>(T(2.0f), T(3.0f), one);
            const double wfin = (double)Round<T>(a * b + c, ProjSpec{});
            if (!same(fin, wfin)) {
                char bb[128];
                std::snprintf(bb, sizeof(bb), "%s fma(2,3,1) finite got=%g want=%g", tag, fin, wfin);
                fail(bb); ++e;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Transcendentals (lo_float_sci.hpp): NaN-in -> NaN-out is universal; a few
    // finite poles / out-of-domain inputs that std:: and P3109 agree on.
    // These wrap std:: then Project, so ±inf poles land per the domain.
    // ------------------------------------------------------------------------
    {
        auto nanprop = [&](const char* nm, T r) {
            if (!IsNaN<Fp>(r)) { fail(std::string(tag) + " " + nm + "(NaN)!=NaN"); ++e; }
        };
        nanprop("sqrt",  sqrt(qNaN));
        nanprop("rSqrt", rSqrt(qNaN));
        nanprop("log",   log(qNaN));
        nanprop("log2",  log2(qNaN));
        nanprop("exp",   exp(qNaN));
        nanprop("exp2",  exp2(qNaN));
        nanprop("sin",   sin(qNaN));
        nanprop("cos",   cos(qNaN));
        nanprop("tan",   tan(qNaN));
        nanprop("sinh",  sinh(qNaN));
        nanprop("cosh",  cosh(qNaN));
        nanprop("tanh",  tanh(qNaN));
        nanprop("asin",  asin(qNaN));
        nanprop("acos",  acos(qNaN));
        nanprop("atan",  atan(qNaN));
        nanprop("asinh", asinh(qNaN));
        nanprop("acosh", acosh(qNaN));
        nanprop("atanh", atanh(qNaN));

        // Out-of-domain -> NaN (matches P3109 §4.10 domains).
        if (!IsNaN<Fp>(sqrt(T(-1.0f)))) { fail(std::string(tag) + " sqrt(-1)!=NaN"); ++e; }
        if (!IsNaN<Fp>(log (T(-1.0f)))) { fail(std::string(tag) + " log(-1)!=NaN");  ++e; }
        if (!IsNaN<Fp>(asin(T(2.0f))))  { fail(std::string(tag) + " asin(2)!=NaN");  ++e; }
        if (!IsNaN<Fp>(acos(T(2.0f))))  { fail(std::string(tag) + " acos(2)!=NaN");  ++e; }
        if (!IsNaN<Fp>(acosh(zero)))    { fail(std::string(tag) + " acosh(0)!=NaN"); ++e; }

        // log(0) is the -inf pole: Extended -> -inf, Saturating -> -max.
        const double lg0 = (double)log(zero);
        const double w_lg0 = ext ? -std::numeric_limits<double>::infinity() : dlow;
        if (!same(lg0, w_lg0)) { fail(std::string(tag) + " log(0) pole"); ++e; }
    }

    if (e == 0) std::printf("%s : pass\n", tag);
    else        std::printf("%s : FAIL (%d)\n", tag, e);
    return e;
}

// ---- instantiation machinery (mirrors test_rounding_modes.cpp) --------------
static int g_total = 0;

template <int l, int... Ps>
void edge_for_l(std::integer_sequence<int, Ps...>) {
    // p ranges 2 .. l-1  (Ps is 0-based; +2 -> start at p=2)
    ((g_total += test_edge_3109<l, Ps + 2, Inf_Behaviors::Saturating>()), ...);
    ((g_total += test_edge_3109<l, Ps + 2, Inf_Behaviors::Extended>()),   ...);
}
template <int... Ls>
void edge_all_l(std::integer_sequence<int, Ls...>) {
    // l ranges 3 .. 8 ; for each l, p in 2..l-1  (that is l-2 values)
    (edge_for_l<Ls>(std::make_integer_sequence<int, Ls - 2>{}), ...);
}
template <int Offset, int... Is>
constexpr auto offset_sequence(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, (Is + Offset)...>{};
}

int main() {
    // l = 3..8, p = 2..l-1, both Signed domains.
    edge_all_l(offset_sequence<3>(std::make_integer_sequence<int, 6>{}));

    // Explicitly cover a mantissa-0 format (p=1): no subnormals, and
    // MaxSubnormalOf must return NaN.  Both domains.
    g_total += test_edge_3109<4, 1, Inf_Behaviors::Saturating>();
    g_total += test_edge_3109<4, 1, Inf_Behaviors::Extended>();
    g_total += test_edge_3109<8, 1, Inf_Behaviors::Saturating>();
    g_total += test_edge_3109<8, 1, Inf_Behaviors::Extended>();

    if (g_total == 0) { std::printf("All edge-case tests passed\n"); return 0; }
    std::printf("%d edge-case test(s) FAILED\n", g_total);
    return 1;
}
