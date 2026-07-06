// test_block_ops.cpp -- CPU tests for src/block_ops.hpp (IEEE P3109 §5 block operations on the
// MicroScaled objects of Vector.h / Matrix.h).
//
// Ownership: this file is OWNED by the test author. The lead implements src/block_ops.hpp
// concurrently against the FROZEN contract in loop/block_ops_api.md; if a signature must change
// the lead messages the author (see the report). The header may not exist yet -- that is expected;
// this test is written to PASS once the header matches the frozen contract and to have TEETH
// (fail on a regression).
//
// Design (mirrors test_matrix_vector.cpp / test_rounding_modes.cpp per loop/NUMERICAL_TESTING.md):
//   * Oracle is double (rule 1): every expected value is computed straight from the ω-op tables of
//     loop/block_ops_3109.md §1/§2/§5/§6 and edge_cases_3109.md §4.10/§4.11 -- never from the DUT.
//   * The per-element projection primitive is, by the frozen contract, block_project_elem<Out>:
//       if isnan(S)||isnan(X) -> Round<Out>(NaN)
//       else if S==0          -> Round<Out>(0)
//       else if isinf(S)      -> Round<Out>((X==0)?0 : copysign(1,X)*copysign(1,S))
//       else                  -> Round<Out>(X/S)
//     reproduced by oproject_real() below (guard order matters -- first-match-wins).
//   * ωBlockDecode of one element is plain elem*scale in double (∞×0/0×∞ -> NaN come free from IEEE
//     double) -- odecode() below.
//   * Dyadic data (pow2 tile maxima, ratios exact in the element format) makes quantize/dequantize
//     round-trips BIT-EXACT (rule 2). A non-pow2 scale format (UScal, unsigned P3109 with 2 mantissa
//     bits) drives a lossy path with a 0.5·ulp(ratio)·scale bound and a "teeth" element that
//     genuinely re-rounds (rule 3).
//   * For the special-value / semantics sections the element AND scale formats are `double`, so the
//     projection is IDENTITY (Round<double> is IdentityConversion) and the block ops must reproduce
//     the pure double oracle exactly -- these carry the real teeth (no reliance on Round rounding a
//     narrow format). For SatMode / narrow-target sections the expected value is
//     Round<Tout>(independently_computed_real, ps): Round is a separate, independently-tested
//     primitive (test_sat_mode / test_rounding_modes) and is the contract's definition of the
//     projection step, so this checks that the block op computed the right REAL value with the right
//     guard order and threaded the right ps.
//   * All workspace buffers are poisoned before every op (rule 12).
//
// Coordinator-pinned oracle points (message 2026-07-06), all consistent with the frozen contract:
//   (1) convert_to_block_max_abs_finite computes the element UNIFORMLY for every branch as
//       block_project_elem<f_r>(double(stored_scale), x, ps) using the actually-stored (rounded)
//       scale. ±1 for an ∞ element only arises when f_s is Extended (can store ∞); with a Saturating
//       f_s the scale rounds to MaxFinite(f_s) and ∞/MaxFinite saturates instead. This test observes
//       ∞⇒±1 by using an Extended scale format (double).
//   (2) Reductions/block_dot with the DEFAULT Out=double return the RAW double accumulator (no
//       projection, ps ignored) == the double-fold oracle. With an explicit narrow Out they return
//       Round<Out>(acc, ps). Both are tested.
//   (3) Matrix block_decode(A, double* Z) fills Z in ROW-MAJOR LOGICAL order Z[r*A.cols()+c],
//       independent of Layout L.
//
// TEETH (which mutation each section catches):
//   * Dropping the S==0 guard in block_project_elem -> the S=0 special-value rows fail (S=0 would
//     divide and give NaN/∞ instead of 0).
//   * Letting ∞ set the scale in convert_to_block_max_abs_finite (the MX_real_quantize bug) -> the
//     "some ∞, finite max unaffected" and ps_scale sections fail (scale would be ∞ not the finite
//     max, and finite mates would collapse to 0/±1).
//   * Mapping the S=∞ element to ±MaxFinite instead of ±1 -> the narrow-target S=∞ rows fail.
//   * block_div not special-casing decoded Y==0 -> the div-by-zero rows fail (IEEE double gives ±∞,
//     spec wants NaN).
//   * Keying a matrix op off physical ld instead of the logical tile -> the ld-independence section
//     fails (bit mismatch between ld==m and ld==m+3).
//   * all-NaN block giving scale 0 / elements 0 (the MX_real_quantize behavior) instead of NaN/NaN
//     -> the NOTE-1 rows fail.

#include "block_ops.hpp"
#include "Vector.h"
#include "Matrix.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cstring>
#include <vector>
#include <limits>

using namespace lo_float;

static int g_errors = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << "  [" << __FILE__ << ":" << __LINE__ << "]\n"; \
            g_errors++; \
        } \
    } while (0)

// ---- formats ----------------------------------------------------------------------------------
using Elem  = ocp_e4m3;   // element format: 8-bit, 3 mantissa bits, Saturating (no Inf), has NaN
using Scal  = ocp_e8m0;   // MX shared-scale: MUST be Unsigned, power-of-two-only (lossless pow2 max)
// unsigned P3109 scale WITH mantissa bits (2 mantissa) -> a NON power-of-two scale grid (lossy path)
using UScal = P_3109_float<8, 3, lo_float::Signedness::Unsigned>;
// Saturating (no-∞) vs Extended (has-∞) signed element formats -- to exercise SatMode divergence.
using SatElem = P_3109_float<8, 4, lo_float::Signedness::Signed, lo_float::Inf_Behaviors::Saturating>;
using ExtElem = P_3109_float<8, 4, lo_float::Signedness::Signed, lo_float::Inf_Behaviors::Extended>;
// tiny format for exhaustive enumeration (Binary4p2sf): 16 code points.
using F4 = P_3109_float<4, 2, lo_float::Signedness::Signed, lo_float::Inf_Behaviors::Saturating>;

static const double DINF = std::numeric_limits<double>::infinity();
static const double DNAN = std::numeric_limits<double>::quiet_NaN();

// ---- oracle helpers ---------------------------------------------------------------------------
// Compare two doubles treating NaN==NaN as equal; ±Inf must match sign.
static bool eqd(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return a == b;
}

// ωBlockDecode of one element (§1): plain product in double (∞×0 -> NaN comes free).
static double odecode(double elem, double scale) { return elem * scale; }

// ωBlockProject of one real X against scale S (§2/§6.2), the REAL value pre-round. Guard order is
// load-bearing (first-match-wins): NaN -> zero-scale -> inf-scale -> divide.
static double oproject_real(double S, double X) {
    if (std::isnan(S) || std::isnan(X)) return DNAN;
    if (S == 0.0)                        return 0.0;
    if (std::isinf(S))                   return (X == 0.0) ? 0.0
                                                 : std::copysign(1.0, X) * std::copysign(1.0, S);
    return X / S;                        // finite/finite; X/±∞ -> 0 handled by IEEE double
}

// e4m3 ULP for the lossy-bound checks (mirrors test_matrix_vector.cpp).
static double e4m3_ulp(double ratio) {
    double a = std::fabs(ratio);
    return (a >= std::exp2(-6)) ? std::exp2(std::floor(std::log2(a)) - 3) : std::exp2(-9);
}

// =====================================================================================
// 1. Vector block_decode: dyadic bit-exact + special values + exhaustive tiny format
// =====================================================================================
static void test_vec_block_decode() {
    // (a) dyadic, narrow storage (e4m3 elem / e8m0 pow2 scale): decode is exact in double.
    {
        const int r = 4, M = 9;
        float src[M] = {4.0f, -2.0f, 1.0f, 0.0f, 1.0f, 0.5f, -0.25f, 0.0f, 2.0f};
        Vector<float, int> a(src, M);
        Elem priv[M]; Scal exps[3];
        MX_Vector<Elem, Scal, int> b(priv, exps, M, 3, 1, r);
        MX_real_quantize(a, b);

        double Z[M];
        for (double& z : Z) z = -777.0;                 // poison
        block_decode(b, Z);
        for (int i = 0; i < M; i++) {
            double want = odecode((double)b[i], (double)b.get_exp(i));
            CHECK(eqd(Z[i], want), "vec block_decode dyadic @" << i << " got=" << Z[i] << " want=" << want);
            CHECK(Z[i] == (double)src[i], "vec block_decode dyadic reproduces source @" << i);
        }
    }

    // (b) special-value cross product on the DECODE table (§6.2 decode column, §6.5 ∞×0):
    // double element / double scale, so decode is exactly the IEEE double product.
    {
        const double scales[]  = {0.0, DINF, -DINF, DNAN, 3.0};
        const double elems[]   = {0.0, DINF, -DINF, DNAN, 2.5, -2.5};
        const int NS = 5, NE = 6;
        double dat[NE]; double exp1[1];
        MX_Vector<double, double, int> b(dat, exp1, NE, 1, 1, NE);  // one block of NE, one scale
        for (int s = 0; s < NS; s++) {
            b.set_exp(0, scales[s]);
            for (int e = 0; e < NE; e++) b[e] = elems[e];
            double Z[NE];
            for (double& z : Z) z = -777.0;                          // poison
            block_decode(b, Z);
            for (int e = 0; e < NE; e++) {
                double want = odecode(elems[e], scales[s]);
                CHECK(eqd(Z[e], want),
                      "vec block_decode special S=" << scales[s] << " X=" << elems[e]
                      << " got=" << Z[e] << " want=" << want);
            }
        }
    }

    // (c) exhaustive tiny element format (rule 7): all 16 F4 code points × several scales, vs the
    // ωDecode×ωMultiply oracle (F4's own decode-to-double is the trusted ωDecode).
    {
        const int K = 16;
        const double scales[] = {0.5, 1.0, 2.0, 4.0, 0.0};   // includes 0 (0×NaN -> NaN teeth)
        F4 dat[K]; double exp1[1];
        MX_Vector<F4, double, int> b(dat, exp1, K, 1, 1, K);
        for (double sc : scales) {
            b.set_exp(0, sc);
            for (uint32_t rep = 0; rep < (uint32_t)K; rep++) b[rep] = F4::FromRep(rep);
            double Z[K];
            for (double& z : Z) z = -777.0;                  // poison
            block_decode(b, Z);
            for (uint32_t rep = 0; rep < (uint32_t)K; rep++) {
                double want = odecode((double)b[rep], sc);
                CHECK(eqd(Z[rep], want),
                      "vec block_decode exhaustive F4 rep=" << rep << " scale=" << sc
                      << " got=" << Z[rep] << " want=" << want);
            }
        }
    }
}

// =====================================================================================
// 2. Vector convert_from_block: b[i] = Round<Tb>(decode_i, ps)
// =====================================================================================
static void test_vec_convert_from_block() {
    // (a) dyadic into a double destination -> identity round == exact decode.
    {
        const int r = 4, M = 9;
        float src[M] = {4.0f, -2.0f, 1.0f, 0.0f, 1.0f, 0.5f, -0.25f, 0.0f, 2.0f};
        Vector<float, int> a(src, M);
        Elem priv[M]; Scal exps[3];
        MX_Vector<Elem, Scal, int> b(priv, exps, M, 3, 1, r);
        MX_real_quantize(a, b);

        double out[M];
        for (double& x : out) x = -777.0;                     // poison
        Vector<double, int> vb(out, M);
        convert_from_block(b, vb);
        for (int i = 0; i < M; i++)
            CHECK(out[i] == (double)src[i], "vec convert_from_block dyadic @" << i);
    }

    // (b) special values into a double destination (decode then identity project).
    {
        const int NE = 5;
        double dat[NE]; double exp1[1];
        MX_Vector<double, double, int> b(dat, exp1, NE, 1, 1, NE);
        b.set_exp(0, DINF);
        double xs[NE] = {0.0, 2.0, -2.0, DNAN, DINF};
        for (int i = 0; i < NE; i++) b[i] = xs[i];
        double out[NE];
        for (double& x : out) x = -777.0;                     // poison
        Vector<double, int> vb(out, NE);
        convert_from_block(b, vb);
        for (int i = 0; i < NE; i++) {
            double want = odecode(xs[i], DINF);               // ∞×0 -> NaN, ∞×finite -> ±∞
            CHECK(eqd(out[i], want), "vec convert_from_block special @" << i
                  << " got=" << out[i] << " want=" << want);
        }
    }

    // (c) rounding into a narrow destination (teeth: at least one element re-rounds). Non-dyadic
    // decoded values projected into e4m3 must equal Round<Elem>(decode, ps).
    {
        const int M = 4;
        double dat[M]; double exp1[1];
        MX_Vector<double, double, int> b(dat, exp1, M, 1, 1, M);
        b.set_exp(0, 1.0);
        double xs[M] = {1.7, 0.3, 100.25, -0.13};             // not all representable in e4m3
        for (int i = 0; i < M; i++) b[i] = xs[i];
        Elem out[M];
        for (Elem& x : out) x = static_cast<Elem>(-3.0);      // poison
        Vector<Elem, int> vb(out, M);
        ProjSpec ps{Rounding_Mode::RoundToNearestEven};
        convert_from_block(b, vb, ps);
        int rerounded = 0;
        for (int i = 0; i < M; i++) {
            double want = (double)Round<Elem>(odecode(xs[i], 1.0), ps);
            CHECK((double)out[i] == want, "vec convert_from_block narrow round @" << i
                  << " got=" << (double)out[i] << " want=" << want);
            if ((double)out[i] != xs[i]) rerounded++;
        }
        CHECK(rerounded > 0, "vec convert_from_block narrow path actually re-rounds (teeth)");
    }
}

// =====================================================================================
// 3. Vector convert_to_block (caller scale s): b[i] = block_project_elem<Tb>(s, a[i], ps)
// =====================================================================================
static void test_vec_convert_to_block() {
    // (a) dyadic, narrow storage, exact scale: round-trip bit-exact.
    {
        const int r = 4, M = 8;
        double a_[M] = {4.0, -2.0, 1.0, 0.0, 8.0, 4.0, -1.0, 0.0};
        Vector<double, int> a(a_, M);
        Elem priv[M]; Scal exps[2];
        for (Elem& e : priv) e = static_cast<Elem>(-3.0);         // poison
        for (Scal& e : exps) e = static_cast<Scal>(64.0);         // poison
        MX_Vector<Elem, Scal, int> b(priv, exps, M, 2, 1, r);
        // caller scale 4.0 (block0) applies to all blocks here (single s). ratios all exact in e4m3.
        Scal s = static_cast<Scal>(4.0);
        convert_to_block(a, s, b);
        for (int blk = 0; blk < 2; blk++)
            CHECK((double)b.get_exp(blk * r) == 4.0, "vec convert_to_block scale written @" << blk);
        for (int i = 0; i < M; i++) {
            double want = oproject_real(4.0, a_[i]);              // exact ratio, e4m3 lossless
            CHECK((double)b[i] == want, "vec convert_to_block dyadic elem @" << i
                  << " got=" << (double)b[i] << " want=" << want);
        }
    }

    // (b) special-value cross product S × X into a DOUBLE target (identity project -> exact oracle).
    // Asserts §6.2: S=0 -> 0; S=±∞,X≠0 -> ±1 (sign=sgnX·sgnS); S=±∞,X=0 -> 0; S/X NaN -> NaN;
    // X/0 (finite S=0) never divides. This is the strongest special-value teeth (no Round reliance).
    {
        const double scales[] = {0.0, DINF, -DINF, DNAN, 3.0};
        const double elems[]  = {0.0, DINF, -DINF, DNAN, 2.5, -7.5};
        const int NS = 5, NE = 6;
        for (int s = 0; s < NS; s++) {
            double a_[NE]; for (int e = 0; e < NE; e++) a_[e] = elems[e];
            Vector<double, int> a(a_, NE);
            double dat[NE]; double exp1[1];
            for (double& d : dat) d = -777.0;                     // poison
            exp1[0] = -777.0;
            MX_Vector<double, double, int> b(dat, exp1, NE, 1, 1, NE);
            convert_to_block(a, scales[s], b);
            CHECK(eqd((double)b.get_exp(0), scales[s]),
                  "vec convert_to_block special scale written S=" << scales[s]);
            for (int e = 0; e < NE; e++) {
                double want = oproject_real(scales[s], elems[e]);
                CHECK(eqd((double)b[e], want),
                      "vec convert_to_block special S=" << scales[s] << " X=" << elems[e]
                      << " got=" << (double)b[e] << " want=" << want);
            }
        }
    }

    // (c) S=±∞ into a NARROW target must produce EXACTLY ±1 (never ±MaxFinite). Pinned §6.1.
    {
        const int NE = 4;
        double a_[NE] = {2.5, -2.5, 0.0, 100.0};
        Vector<double, int> a(a_, NE);
        Elem priv[NE]; double exp1[1];
        for (Elem& e : priv) e = static_cast<Elem>(-3.0);         // poison
        MX_Vector<Elem, double, int> b(priv, exp1, NE, 1, 1, NE);
        convert_to_block(a, DINF, b);
        CHECK((double)b[0] == 1.0,  "vec convert_to_block S=+inf X>0 -> +1 (not MaxFinite)");
        CHECK((double)b[1] == -1.0, "vec convert_to_block S=+inf X<0 -> -1 (not MaxFinite)");
        CHECK((double)b[2] == 0.0,  "vec convert_to_block S=+inf X=0 -> 0");
        CHECK((double)b[3] == 1.0,  "vec convert_to_block S=+inf X>0(large) -> +1 (not MaxFinite)");
        // S=-inf: sign flips
        convert_to_block(a, -DINF, b);
        CHECK((double)b[0] == -1.0, "vec convert_to_block S=-inf X>0 -> -1");
        CHECK((double)b[1] == 1.0,  "vec convert_to_block S=-inf X<0 -> +1");
        CHECK((double)b[2] == 0.0,  "vec convert_to_block S=-inf X=0 -> 0");
    }
}

// =====================================================================================
// 4. Vector convert_to_block_max_abs_finite (computed scale): §5 NOTE 1-4 + ps_scale + lossy teeth
// =====================================================================================

// Oracle: classify block per the coordinator-pinned rule, return the STORED (rounded) scale as
// double for scale-format Ts, computed via Round<Ts>.
template<class Ts>
static double oracle_scale(const double* X, int n, ProjSpec ps_scale) {
    double Sf = 0.0; bool has_fin_nz = false, has_inf = false, has_nan = false;
    for (int i = 0; i < n; i++) {
        if (std::isnan(X[i])) { has_nan = true; continue; }
        if (std::isinf(X[i])) { has_inf = true; continue; }
        double a = std::fabs(X[i]);
        if (a > Sf) Sf = a;
        if (a > 0.0) has_fin_nz = true;
    }
    if (has_fin_nz)      return (double)Round<Ts>(Sf,   ps_scale);
    else if (has_inf)    return (double)Round<Ts>(DINF, ps_scale);
    else if (has_nan)    return (double)Round<Ts>(DNAN, ps_scale);
    else                 return 0.0;
}

static void test_vec_convert_to_block_max_abs_finite() {
    ProjSpec ps{};        // element projection: RNE / OvfInf
    ProjSpec ps_scale{};  // scale projection: RNE / OvfInf

    // (a) dyadic, narrow storage, pow2 maxima: computed scale exact, elements exact.
    {
        const int r = 4, M = 9;
        double a_[M] = {4.0, -2.0, 1.0, 0.0, 1.0, 0.5, -0.25, 0.0, 2.0};
        Vector<double, int> a(a_, M);
        Elem priv[M]; Scal exps[3];
        for (Elem& e : priv) e = static_cast<Elem>(-3.0);
        for (Scal& e : exps) e = static_cast<Scal>(64.0);
        MX_Vector<Elem, Scal, int> b(priv, exps, M, 3, 1, r);
        convert_to_block_max_abs_finite(a, b, ps_scale, ps);
        double sc[3] = {4.0, 1.0, 2.0};
        for (int blk = 0; blk < 3; blk++)
            CHECK((double)b.get_exp(blk * r) == sc[blk], "vec c2b_maf dyadic scale @" << blk);
        for (int i = 0; i < M; i++) {
            double scv = (double)b.get_exp(i);
            double want = oproject_real(scv, a_[i]);
            CHECK((double)b[i] == want, "vec c2b_maf dyadic elem @" << i);
        }
    }

    // (b) NOTE 1 -- all-NaN block: scale NaN, ALL elements NaN. DIFFERS from MX_real_quantize
    // (which would give scale 0 / elements 0). Double storage -> exact.
    {
        const int M = 4;
        double a_[M] = {DNAN, DNAN, DNAN, DNAN};
        Vector<double, int> a(a_, M);
        double dat[M]; double exp1[1];
        for (double& d : dat) d = -777.0; exp1[0] = -777.0;
        MX_Vector<double, double, int> b(dat, exp1, M, 1, 1, M);
        convert_to_block_max_abs_finite(a, b, ps_scale, ps);
        CHECK(std::isnan((double)b.get_exp(0)), "vec c2b_maf NOTE1 all-NaN scale is NaN");
        for (int i = 0; i < M; i++)
            CHECK(std::isnan((double)b[i]), "vec c2b_maf NOTE1 all-NaN elem NaN @" << i);
    }

    // NOTE 1 corner: a block with finite ZEROS and a NaN (no nonzero finite) -> NaN branch, ALL
    // elements (incl. the zeros) become NaN. Teeth for the has_fin_nz=(Sf>0) guard.
    {
        const int M = 3;
        double a_[M] = {0.0, DNAN, 0.0};
        Vector<double, int> a(a_, M);
        double dat[M]; double exp1[1];
        for (double& d : dat) d = -777.0; exp1[0] = -777.0;
        MX_Vector<double, double, int> b(dat, exp1, M, 1, 1, M);
        convert_to_block_max_abs_finite(a, b, ps_scale, ps);
        CHECK(std::isnan((double)b.get_exp(0)), "vec c2b_maf {0,NaN,0} scale is NaN");
        for (int i = 0; i < M; i++)
            CHECK(std::isnan((double)b[i]), "vec c2b_maf {0,NaN,0} elem NaN @" << i);
    }

    // (c) NOTE 2 / pinned §6.1 -- block with ∞ and NO nonzero finite (incl {0,∞}): with an
    // EXTENDED scale format (double) the scale rounds to +∞ and ∞ elements project to ±1, zeros->0,
    // NaN->NaN.
    {
        const int M = 4;
        double a_[M] = {DINF, 0.0, -DINF, 0.0};              // {0,∞} style
        Vector<double, int> a(a_, M);
        double dat[M]; double exp1[1];
        for (double& d : dat) d = -777.0; exp1[0] = -777.0;
        MX_Vector<double, double, int> b(dat, exp1, M, 1, 1, M);
        convert_to_block_max_abs_finite(a, b, ps_scale, ps);
        CHECK((double)b.get_exp(0) == DINF, "vec c2b_maf NOTE2 scale=+inf (Extended f_s)");
        CHECK((double)b[0] == 1.0,  "vec c2b_maf NOTE2 +inf elem -> +1");
        CHECK((double)b[1] == 0.0,  "vec c2b_maf NOTE2 zero elem -> 0");
        CHECK((double)b[2] == -1.0, "vec c2b_maf NOTE2 -inf elem -> -1");
        CHECK((double)b[3] == 0.0,  "vec c2b_maf NOTE2 zero elem -> 0");
    }

    // (d) NOTE 3 / pinned §6.2 -- SOME ∞, SOME finite: the finite max sets the scale (∞ ignored in
    // the max), only NaN elements are NaN, ∞ elements saturate/preserve per ps. Teeth for the
    // MX_real_quantize "∞ sets the scale" bug (that would give scale ∞ and finite mates -> ±1/0).
    {
        const int M = 5;
        double a_[M] = {2.0, DINF, -4.0, DNAN, 1.0};         // finite max = 4.0
        Vector<double, int> a(a_, M);
        double dat[M]; double exp1[1];
        for (double& d : dat) d = -777.0; exp1[0] = -777.0;
        MX_Vector<double, double, int> b(dat, exp1, M, 1, 1, M);
        convert_to_block_max_abs_finite(a, b, ps_scale, ps);
        CHECK((double)b.get_exp(0) == 4.0, "vec c2b_maf NOTE3 finite max sets scale (∞ ignored)");
        CHECK((double)b[0] == 2.0 / 4.0,  "vec c2b_maf NOTE3 finite elem 2/4");
        CHECK((double)b[1] == DINF,       "vec c2b_maf NOTE3 +inf/finite -> +inf (preserved)");
        CHECK((double)b[2] == -4.0 / 4.0, "vec c2b_maf NOTE3 finite elem -4/4");
        CHECK(std::isnan((double)b[3]),   "vec c2b_maf NOTE3 NaN elem stays NaN");
        CHECK((double)b[4] == 1.0 / 4.0,  "vec c2b_maf NOTE3 finite elem 1/4");
    }

    // (e) NOTE 4 -- all-zero block: scale 0, elements 0 (no divide-by-zero).
    {
        const int M = 4;
        double a_[M] = {0.0, 0.0, 0.0, 0.0};
        Vector<double, int> a(a_, M);
        double dat[M]; double exp1[1];
        for (double& d : dat) d = -777.0; exp1[0] = -777.0;
        MX_Vector<double, double, int> b(dat, exp1, M, 1, 1, M);
        convert_to_block_max_abs_finite(a, b, ps_scale, ps);
        CHECK((double)b.get_exp(0) == 0.0, "vec c2b_maf NOTE4 all-zero scale 0");
        for (int i = 0; i < M; i++) CHECK((double)b[i] == 0.0, "vec c2b_maf NOTE4 elem 0 @" << i);
    }

    // (f) pinned §6.4 -- ps_scale direction: computed scale == Round<UScal>(true_finite_max,ps_scale)
    // for BOTH default (RNE) and round-up. true_finite_max = 1.6, strictly between the UScal grid
    // points 1.5 and 1.75 (midpoint 1.625) and below it, so RNE rounds DOWN to 1.5 while RoundUp
    // rounds UP to 1.75 -- the two directions genuinely differ (teeth). ∞ and NaN present must be
    // ignored in the max.
    {
        const int M = 5;
        double a_[M] = {1.6, -0.8, DINF, DNAN, 0.4};          // finite max = 1.6
        Vector<double, int> a(a_, M);
        ProjSpec ps_rne{Rounding_Mode::RoundToNearestEven};
        ProjSpec ps_up {Rounding_Mode::RoundUp};

        Elem priv[M]; UScal exps[1];
        for (Elem& e : priv) e = static_cast<Elem>(-3.0);

        MX_Vector<Elem, UScal, int> b_rne(priv, exps, M, 1, 1, M);
        convert_to_block_max_abs_finite(a, b_rne, ps_rne, ps);
        double want_rne = oracle_scale<UScal>(a_, M, ps_rne);
        CHECK((double)b_rne.get_exp(0) == want_rne,
              "vec c2b_maf ps_scale RNE scale got=" << (double)b_rne.get_exp(0) << " want=" << want_rne);

        MX_Vector<Elem, UScal, int> b_up(priv, exps, M, 1, 1, M);
        convert_to_block_max_abs_finite(a, b_up, ps_up, ps);
        double want_up = oracle_scale<UScal>(a_, M, ps_up);
        CHECK((double)b_up.get_exp(0) == want_up,
              "vec c2b_maf ps_scale RoundUp scale got=" << (double)b_up.get_exp(0) << " want=" << want_up);
        CHECK(want_rne != want_up, "vec c2b_maf ps_scale directions differ (teeth on ps_scale wiring)");
    }

    // (g) lossy element path (UScal scale, e4m3 element): every element obeys the 0.5·ulp(ratio)·scale
    // bound and at least one genuinely re-rounds (rule 3 teeth).
    {
        const int r = 4, M = 8;
        double a_[M] = {1.7, 0.85, -0.4, 0.0, 1.6, -1.6, 0.2, 0.1};
        Vector<double, int> a(a_, M);
        Elem priv[M]; UScal exps[2];
        for (Elem& e : priv) e = static_cast<Elem>(-3.0);
        MX_Vector<Elem, UScal, int> b(priv, exps, M, 2, 1, r);
        convert_to_block_max_abs_finite(a, b, ps, ps);
        int rerounded = 0;
        for (int blk = 0; blk < 2; blk++) {
            double want_scale = oracle_scale<UScal>(a_ + blk * r, r, ps);
            CHECK((double)b.get_exp(blk * r) == want_scale, "vec c2b_maf lossy scale @" << blk);
        }
        for (int i = 0; i < M; i++) {
            double s = (double)b.get_exp(i);
            double got = (double)b[i] * s;
            double ratio = (s == 0.0) ? 0.0 : std::fabs(a_[i]) / s;
            CHECK(std::isfinite((double)b[i]), "vec c2b_maf lossy elem finite @" << i);
            CHECK(std::fabs(got - a_[i]) <= 0.5 * e4m3_ulp(ratio) * s + 1e-15,
                  "vec c2b_maf lossy half-ULP bound @" << i << " a=" << a_[i] << " got=" << got);
            if (got != a_[i]) rerounded++;
        }
        CHECK(rerounded > 0, "vec c2b_maf lossy path actually re-rounds (teeth)");
    }
}

// =====================================================================================
// 5. Vector block_op + aliases: add/sub/mul/div/abs/negate + generic driver
// =====================================================================================
static void test_vec_block_ops() {
    // Build two double-storage MX operands with known decoded values; result scale s_r drives the
    // final projection. Double target -> identity project, so out[i] == oproject_real(s_r, op(...)).
    const int M = 4;
    double xd[M] = {2.0, -3.0, 4.0, 0.0};      // decoded X (scale 1)
    double yd[M] = {1.0, 5.0, -2.0, 6.0};      // decoded Y (scale 1)
    double xdat[M], ydat[M], xe[1], ye[1];
    for (int i = 0; i < M; i++) { xdat[i] = xd[i]; ydat[i] = yd[i]; }
    xe[0] = 1.0; ye[0] = 1.0;
    MX_Vector<double, double, int> X(xdat, xe, M, 1, 1, M);
    MX_Vector<double, double, int> Y(ydat, ye, M, 1, 1, M);

    auto run_binary = [&](const char* name, auto op, auto call) {
        for (double s_r : {1.0, 2.0, 0.0, DINF}) {
            double odat[M]; double oe[1];
            for (double& d : odat) d = -777.0; oe[0] = -777.0;   // poison
            MX_Vector<double, double, int> O(odat, oe, M, 1, 1, M);
            call(X, Y, s_r, O);
            CHECK(eqd((double)O.get_exp(0), s_r), name << " result scale written s_r=" << s_r);
            for (int i = 0; i < M; i++) {
                double want = oproject_real(s_r, op(xd[i], yd[i]));
                CHECK(eqd((double)O[i], want), name << " @" << i << " s_r=" << s_r
                      << " got=" << (double)O[i] << " want=" << want);
            }
        }
    };

    run_binary("block_add", [](double a, double b){ return a + b; },
               [](auto& x, auto& y, double s, auto& o){ block_add(x, y, s, o); });
    run_binary("block_sub", [](double a, double b){ return a - b; },
               [](auto& x, auto& y, double s, auto& o){ block_sub(x, y, s, o); });
    run_binary("block_mul", [](double a, double b){ return a * b; },
               [](auto& x, auto& y, double s, auto& o){ block_mul(x, y, s, o); });
    // block_div carries the P3109 correction: decoded Y==0 -> NaN (not ±∞). Oracle mirrors it.
    run_binary("block_div", [](double a, double b){ return (b == 0.0) ? DNAN : a / b; },
               [](auto& x, auto& y, double s, auto& o){ block_div(x, y, s, o); });

    // Unary aliases.
    auto run_unary = [&](const char* name, auto op, auto call) {
        double odat[M]; double oe[1];
        for (double& d : odat) d = -777.0; oe[0] = -777.0;
        MX_Vector<double, double, int> O(odat, oe, M, 1, 1, M);
        double s_r = 2.0;
        call(X, s_r, O);
        for (int i = 0; i < M; i++) {
            double want = oproject_real(s_r, op(xd[i]));
            CHECK(eqd((double)O[i], want), name << " @" << i << " got=" << (double)O[i] << " want=" << want);
        }
    };
    run_unary("block_abs",    [](double a){ return std::fabs(a); },
              [](auto& x, double s, auto& o){ block_abs(x, s, o); });
    run_unary("block_negate", [](double a){ return -a; },
              [](auto& x, double s, auto& o){ block_negate(x, s, o); });

    // Div-by-zero teeth: Y=0 (decoded), block_div must give NaN, not ±inf. (yd[3]=6 nonzero, so
    // build a dedicated zero-Y operand.)
    {
        double zydat[M] = {0.0, 0.0, 0.0, 0.0}; double zye[1] = {1.0};
        MX_Vector<double, double, int> ZY(zydat, zye, M, 1, 1, M);
        double odat[M]; double oe[1];
        for (double& d : odat) d = -777.0; oe[0] = -777.0;
        MX_Vector<double, double, int> O(odat, oe, M, 1, 1, M);
        block_div(X, ZY, 1.0, O);
        for (int i = 0; i < M; i++)
            CHECK(std::isnan((double)O[i]), "block_div decoded-Y==0 -> NaN (not inf) @" << i);
    }

    // ∞ arithmetic teeth: +∞ + −∞ -> NaN (block_add), ∞×0 -> NaN (block_mul).
    {
        double id[M]   = {DINF, DINF, 3.0, 0.0};
        double jd[M]   = {-DINF, 2.0, -DINF, DINF};
        double idat[M], jdat[M], ie[1] = {1.0}, je[1] = {1.0};
        for (int i = 0; i < M; i++) { idat[i] = id[i]; jdat[i] = jd[i]; }
        MX_Vector<double, double, int> I(idat, ie, M, 1, 1, M);
        MX_Vector<double, double, int> J(jdat, je, M, 1, 1, M);
        double odat[M]; double oe[1];
        for (double& d : odat) d = -777.0; oe[0] = -777.0;
        MX_Vector<double, double, int> O(odat, oe, M, 1, 1, M);
        block_add(I, J, 1.0, O);
        CHECK(std::isnan((double)O[0]), "block_add +inf + -inf -> NaN");
        CHECK((double)O[1] == DINF,     "block_add +inf + finite -> +inf");
        block_mul(I, J, 1.0, O);
        CHECK((double)O[0] == -DINF,    "block_mul +inf * -inf -> -inf");
        CHECK(std::isnan((double)O[3]), "block_mul 0 * +inf -> NaN");
    }

    // Generic driver block_op(op, s_r, ps, out, in...) with a lambda -- must apply the functor
    // VERBATIM on doubles (IEEE semantics; no P3109 correction). A binary max lambda here.
    {
        double odat[M]; double oe[1];
        for (double& d : odat) d = -777.0; oe[0] = -777.0;
        MX_Vector<double, double, int> O(odat, oe, M, 1, 1, M);
        auto maxop = [](double a, double b){ return a > b ? a : b; };
        block_op(maxop, 1.0, ProjSpec{}, O, X, Y);
        for (int i = 0; i < M; i++) {
            double want = oproject_real(1.0, maxop(xd[i], yd[i]));
            CHECK(eqd((double)O[i], want), "block_op generic max @" << i);
        }
    }
    // NOTE: block_op empty-pack (assert >=1 operand) is a runtime assert-abort; not exercised here.
}

// =====================================================================================
// 6. Vector reductions: block_reduce_add / block_reduce_mul / block_dot
// =====================================================================================
static void test_vec_reductions() {
    // Decoded values via double storage (scale 1) so the fold oracle is exact.
    const int M = 5;
    double vd[M] = {1.5, -2.0, 0.5, 4.0, -1.0};
    double vdat[M]; double ve[1] = {1.0};
    for (int i = 0; i < M; i++) vdat[i] = vd[i];
    MX_Vector<double, double, int> V(vdat, ve, M, 1, 1, M);

    // (a) block_reduce_add / mul, default Out=double -> RAW double accumulator (no projection).
    {
        double sum = 0.0; for (double x : vd) sum += x;   // seed 0
        double prod = 1.0; for (double x : vd) prod *= x; // seed 1
        double got_add = block_reduce_add(V);
        double got_mul = block_reduce_mul(V);
        CHECK(got_add == sum,  "block_reduce_add default double == fold sum got=" << got_add << " want=" << sum);
        CHECK(got_mul == prod, "block_reduce_mul default double == fold product got=" << got_mul << " want=" << prod);
    }

    // (b) narrow Out projects: Round<Elem>(acc, ps).
    {
        ProjSpec ps{Rounding_Mode::RoundToNearestEven};
        double sum = 0.0; for (double x : vd) sum += x;
        double prod = 1.0; for (double x : vd) prod *= x;
        Elem got_add = block_reduce_add<Elem>(V, ps);
        Elem got_mul = block_reduce_mul<Elem>(V, ps);
        CHECK((double)got_add == (double)Round<Elem>(sum, ps),  "block_reduce_add<Elem> projects");
        CHECK((double)got_mul == (double)Round<Elem>(prod, ps), "block_reduce_mul<Elem> projects");
    }

    // (c) block_dot default double -> raw fold of products.
    {
        double wd[M] = {2.0, 1.0, -4.0, 0.5, 3.0};
        double wdat[M]; double we[1] = {1.0};
        for (int i = 0; i < M; i++) wdat[i] = wd[i];
        MX_Vector<double, double, int> W(wdat, we, M, 1, 1, M);
        double dot = 0.0; for (int i = 0; i < M; i++) dot += vd[i] * wd[i];
        double got = block_dot(V, W);
        CHECK(got == dot, "block_dot default double == fold got=" << got << " want=" << dot);
        // narrow Out projects
        ProjSpec ps{};
        double gotn = (double)block_dot<Elem>(V, W, ps);
        CHECK(gotn == (double)Round<Elem>(dot, ps), "block_dot<Elem> projects");
    }

    // (d) NaN propagation: any NaN element -> NaN result.
    {
        double nd[M] = {1.0, DNAN, 2.0, 3.0, 4.0};
        double ndat[M]; double ne[1] = {1.0};
        for (int i = 0; i < M; i++) ndat[i] = nd[i];
        MX_Vector<double, double, int> N(ndat, ne, M, 1, 1, M);
        CHECK(std::isnan(block_reduce_add(N)), "block_reduce_add NaN propagation");
        CHECK(std::isnan(block_reduce_mul(N)), "block_reduce_mul NaN propagation");
    }

    // (e) +∞ + −∞ -> NaN (genuine, from the double accumulator).
    {
        double pd[M] = {DINF, -DINF, 1.0, 2.0, 3.0};
        double pdat[M]; double pe[1] = {1.0};
        for (int i = 0; i < M; i++) pdat[i] = pd[i];
        MX_Vector<double, double, int> P(pdat, pe, M, 1, 1, M);
        CHECK(std::isnan(block_reduce_add(P)), "block_reduce_add +inf + -inf -> NaN");
    }

    // (f) ∞×0 in the product fold -> NaN.
    {
        double qd[M] = {DINF, 0.0, 1.0, 2.0, 3.0};
        double qdat[M]; double qe[1] = {1.0};
        for (int i = 0; i < M; i++) qdat[i] = qd[i];
        MX_Vector<double, double, int> Q(qdat, qe, M, 1, 1, M);
        CHECK(std::isnan(block_reduce_mul(Q)), "block_reduce_mul inf * 0 -> NaN");
    }
}

// =====================================================================================
// 7. SatMode cross product on the overflow path (convert_to_block with a small caller scale)
// =====================================================================================
template<class Tout>
static void test_satmode_one(const char* label) {
    const double maxf = (double)std::numeric_limits<Tout>::max();
    if (!(maxf > 0.0)) return;
    const bool has_inf = std::numeric_limits<Tout>::has_infinity;
    const double fmt_inf = (double)std::numeric_limits<Tout>::infinity();
    const bool real_inf = has_inf && std::isinf(fmt_inf);

    const Rounding_Mode rm = Rounding_Mode::RoundToNearestEven;
    ProjSpec ps_ovf {rm, Saturation_Mode::OvfInf};
    ProjSpec ps_fin {rm, Saturation_Mode::SatFinite};
    ProjSpec ps_prop{rm, Saturation_Mode::SatPropagate};

    // element = maxf*4 with caller scale s=1 -> ratio = 4*maxf overflows the finite range. Both signs.
    for (double sign : {+1.0, -1.0}) {
        double a_[1] = {sign * maxf * 4.0};
        Vector<double, int> a(a_, 1);
        double s = 1.0;  // caller scale as double so no scale rounding

        auto run = [&](ProjSpec ps) {
            Tout priv[1]; double e1[1];
            priv[0] = static_cast<Tout>(-3.0);                // poison
            MX_Vector<Tout, double, int> b(priv, e1, 1, 1, 1, 1);
            convert_to_block(a, s, b, ps);
            return (double)b[0];
        };
        double r_ovf = run(ps_ovf), r_fin = run(ps_fin), r_prop = run(ps_prop);
        double real = a_[0] / s;

        // Definitional oracle: block_project_elem == Round<Tout>(real, ps).
        CHECK(eqd(r_ovf,  (double)Round<Tout>(real, ps_ovf)),  label << " OvfInf == Round oracle sign=" << sign);
        CHECK(eqd(r_fin,  (double)Round<Tout>(real, ps_fin)),  label << " SatFinite == Round oracle sign=" << sign);
        CHECK(eqd(r_prop, (double)Round<Tout>(real, ps_prop)), label << " SatPropagate == Round oracle sign=" << sign);

        // High-level properties (independent of Round internals).
        CHECK(r_fin == sign * maxf, label << " SatFinite(ovf) -> ±maxf sign=" << sign);
        CHECK(r_prop == sign * maxf, label << " SatPropagate(finite-ovf) -> ±maxf sign=" << sign);
        if (real_inf)
            CHECK(std::isinf(r_ovf) && ((r_ovf > 0) == (sign > 0)), label << " OvfInf(ovf)->±inf sign=" << sign);
        else
            CHECK(r_ovf == sign * maxf, label << " OvfInf(ovf)->±maxf (no-inf fmt) sign=" << sign);
    }
    if (g_errors == 0) { /* quiet */ }
}

static void test_satmode() {
    test_satmode_one<SatElem>("SatMode(SatElem)");
    test_satmode_one<ExtElem>("SatMode(ExtElem)");
}

// =====================================================================================
// 8. Matrix ops: block_decode / convert_from_block / convert_to_block, tilings, ld-independence
// =====================================================================================
static void test_mat_basic() {
    // 6x5 ColMajor, byColumn r=4 (4x1 tiles, ragged last row-block: 4,2). Dyadic pow2 maxima.
    const int m = 6, n = 5, r = 4;
    double vals[6][5];
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) vals[i][j] = 0.0;
    vals[0][0] = 4.0; vals[1][0] = -2.0; vals[2][0] = 1.0; vals[3][0] = 0.0;
    vals[4][0] = 1.0; vals[5][0] = -0.5;
    vals[1][2] = 2.0;

    std::vector<double> data(m * n, -777.0);
    Matrix<double, int, ColMajor> A(data.data(), m, n, m);
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) A(i, j) = vals[i][j];

    std::vector<Elem> priv(m * n);
    std::vector<Scal> exps(2 * n);
    for (Elem& e : priv) e = static_cast<Elem>(-3.0);
    MX_Matrix<Elem, Scal, int, ColMajor> B(priv.data(), exps.data(), m, n, m, r, MX_Layout::byColumn);
    MX_real_quantize(A, B);

    // (a) block_decode(A_mx, double* Z): ROW-MAJOR LOGICAL order Z[row*n+col] (coordinator pin 3).
    std::vector<double> Z(m * n, -777.0);
    block_decode(B, Z.data());
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double want = B.template scaled_val<double>(i, j);
            CHECK(eqd(Z[i * n + j], want), "mat block_decode row-major order @(" << i << "," << j << ")");
        }

    // (b) convert_from_block into a dense double Matrix -> dequantize.
    std::vector<double> dense(m * n, -777.0);
    Matrix<double, int, ColMajor> Dn(dense.data(), m, n, m);
    convert_from_block(B, Dn);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            CHECK(Dn(i, j) == B.template scaled_val<double>(i, j),
                  "mat convert_from_block dequant @(" << i << "," << j << ")");

    // (c) convert_to_block with a caller scale into byRow and byTile MX matrices (double storage,
    // special value S=∞ over a tile).
    {
        const int mm = 4, nn = 6;
        std::vector<double> d(mm * nn);
        Matrix<double, int, RowMajor> P(d.data(), mm, nn, nn);
        for (int i = 0; i < mm; i++) for (int j = 0; j < nn; j++) P(i, j) = (i == 0 && j == 1) ? 0.0 : (double)(i - j);
        std::vector<double> pv(mm * nn, -777.0);
        std::vector<double> pe(mm * 2, -777.0);
        MX_Matrix<double, double, int, RowMajor> Q(pv.data(), pe.data(), mm, nn, nn, 3, MX_Layout::byRow);
        convert_to_block(P, DINF, Q, ProjSpec{});
        for (int i = 0; i < mm; i++)
            for (int j = 0; j < nn; j++) {
                double want = oproject_real(DINF, P(i, j));   // S=∞: X=0 ->0 else ±1
                CHECK(eqd((double)Q(i, j), want), "mat convert_to_block S=inf @(" << i << "," << j << ")");
            }
        // scale anchors written
        CHECK((double)Q.get_exp(0, 0) == DINF, "mat convert_to_block scale anchor written");
    }
}

// convert_to_block_max_abs_finite on matrices: byColumn/byRow/byTile + ragged + ld-independence.
static void test_mat_max_abs() {
    ProjSpec ps{}, ps_scale{};

    // (a) ld-independence regression (bug #10): same logical data at ld==m and ld==m+3 must give
    // bit-identical scales and elements. byColumn r=4 over a 6x5 grid.
    {
        const int m = 6, n = 5, r = 4;
        double vals[6][5];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) vals[i][j] = 0.0;
        vals[0][0] = 4.0; vals[1][0] = -2.0; vals[2][0] = 1.0; vals[4][0] = 1.0; vals[5][0] = -0.5;
        vals[1][2] = 2.0;
        const int ld_pad = m + 3;
        std::vector<double> d1(m * n, -777.0), d2(ld_pad * n, -777.0);
        Matrix<double, int, ColMajor> A1(d1.data(), m, n, m);
        Matrix<double, int, ColMajor> A2(d2.data(), m, n, ld_pad);
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) { A1(i, j) = vals[i][j]; A2(i, j) = vals[i][j]; }

        std::vector<Elem> p1(m * n), p2(ld_pad * n);
        std::vector<Scal> e1(2 * n), e2(2 * n);
        MX_Matrix<Elem, Scal, int, ColMajor> B1(p1.data(), e1.data(), m, n, m, r, MX_Layout::byColumn);
        MX_Matrix<Elem, Scal, int, ColMajor> B2(p2.data(), e2.data(), m, n, ld_pad, r, MX_Layout::byColumn);
        convert_to_block_max_abs_finite(A1, B1, ps_scale, ps);
        convert_to_block_max_abs_finite(A2, B2, ps_scale, ps);
        for (int k = 0; k < 2 * n; k++)
            CHECK((double)e1[k] == (double)e2[k], "mat c2b_maf ld-independent scale @" << k);
        for (int j = 0; j < n; j++) for (int i = 0; i < m; i++)
            CHECK((double)B1(i, j) == (double)B2(i, j), "mat c2b_maf ld-independent elem @(" << i << "," << j << ")");
        // spot-check scales against the double oracle
        CHECK((double)B1.get_exp(0, 0) == 4.0, "mat c2b_maf byColumn tile(0,0) scale");
        CHECK((double)B1.get_exp(4, 0) == 1.0, "mat c2b_maf byColumn tile(1,0) ragged scale");
        CHECK((double)B1.get_exp(0, 2) == 2.0, "mat c2b_maf byColumn tile(0,2) scale");
    }

    // (b) byRow tiling + a per-tile special-value oracle (double storage). Tile with some ∞ and
    // finite: finite max sets scale, ∞ preserved, per NOTE 3.
    {
        const int m = 2, n = 6, r = 3;
        std::vector<double> d(m * n);
        Matrix<double, int, RowMajor> A(d.data(), m, n, n);
        // row0 tile0 (cols 0..2): {2, ∞, -4} -> finite max 4; tile1 (3..5): {1, 0.5, -0.25} max 1
        A(0, 0) = 2.0; A(0, 1) = DINF; A(0, 2) = -4.0;
        A(0, 3) = 1.0; A(0, 4) = 0.5;  A(0, 5) = -0.25;
        // row1 tile0: all NaN -> NaN branch; tile1: {2,-2,1} max 2
        A(1, 0) = DNAN; A(1, 1) = DNAN; A(1, 2) = DNAN;
        A(1, 3) = 2.0;  A(1, 4) = -2.0; A(1, 5) = 1.0;

        std::vector<double> pv(m * n, -777.0), pe(m * 2, -777.0);
        MX_Matrix<double, double, int, RowMajor> B(pv.data(), pe.data(), m, n, n, r, MX_Layout::byRow);
        convert_to_block_max_abs_finite(A, B, ps_scale, ps);
        CHECK((double)B.get_exp(0, 0) == 4.0, "mat c2b_maf byRow tile0 finite-max scale (∞ ignored)");
        CHECK((double)B(0, 1) == DINF,        "mat c2b_maf byRow ∞ preserved");
        CHECK((double)B(0, 0) == 2.0 / 4.0,   "mat c2b_maf byRow finite 2/4");
        CHECK((double)B.get_exp(0, 3) == 1.0, "mat c2b_maf byRow tile1 scale");
        CHECK(std::isnan((double)B.get_exp(1, 0)), "mat c2b_maf byRow all-NaN tile scale NaN");
        for (int c = 0; c < 3; c++)
            CHECK(std::isnan((double)B(1, c)), "mat c2b_maf byRow all-NaN tile elem NaN @" << c);
        CHECK((double)B.get_exp(1, 3) == 2.0, "mat c2b_maf byRow tile scale 2");
    }

    // (c) explicit byTile 2x2 over a 4x4 ColMajor, ragged not needed here; dyadic.
    {
        const int tm = 4, tn = 4;
        std::vector<double> td(tm * tn, 0.0);
        Matrix<double, int, ColMajor> T(td.data(), tm, tn, tm);
        T(0, 0) = 4.0; T(1, 0) = -2.0; T(0, 1) = 1.0;
        T(2, 2) = 2.0; T(3, 2) = -1.0; T(2, 3) = 0.5;
        std::vector<Elem> tp(tm * tn); std::vector<Scal> te(2 * 2);
        for (Elem& e : tp) e = static_cast<Elem>(-3.0);
        MX_Matrix<Elem, Scal, int, ColMajor> TB(tp.data(), te.data(), tm, tn, tm, 2, 2);
        convert_to_block_max_abs_finite(T, TB, ps_scale, ps);
        CHECK((double)TB.get_exp(0, 0) == 4.0, "mat c2b_maf byTile tile(0,0) scale");
        CHECK((double)TB.get_exp(2, 2) == 2.0, "mat c2b_maf byTile tile(1,1) scale");
        CHECK((double)TB.get_exp(0, 2) == 0.0, "mat c2b_maf byTile all-zero tile scale 0");
        CHECK((double)TB(0, 0) == 1.0 && (double)TB(1, 0) == -0.5 && (double)TB(0, 1) == 0.25,
              "mat c2b_maf byTile tile(0,0) ratios");
    }
}

// Matrix block ops + reductions.
static void test_mat_block_ops_and_reductions() {
    const int m = 3, n = 4;
    double av[3][4] = {{1.0, -2.0, 3.0, 0.5}, {4.0, 0.0, -1.0, 2.0}, {0.25, 8.0, -4.0, 1.0}};
    double bv[3][4] = {{2.0, 1.0, -1.0, 4.0}, {0.5, 3.0, 2.0, -2.0}, {1.0, -0.5, 0.0, 6.0}};

    std::vector<double> ad(m * n), bd(m * n), ae(m * n), be(m * n);
    Matrix<double, int, ColMajor> A(ad.data(), m, n, m), Bp(bd.data(), m, n, m);
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) { A(i, j) = av[i][j]; Bp(i, j) = bv[i][j]; }
    // build MX matrices (scale 1, byColumn r=1 so each element its own tile -> decode == raw)
    std::vector<double> amv(m * n), bmv(m * n), ame(m * n), bme(m * n);
    MX_Matrix<double, double, int, ColMajor> AX(amv.data(), ame.data(), m, n, m, 1, MX_Layout::byColumn);
    MX_Matrix<double, double, int, ColMajor> BX(bmv.data(), bme.data(), m, n, m, 1, MX_Layout::byColumn);
    convert_to_block(A, 1.0, AX, ProjSpec{});
    convert_to_block(Bp, 1.0, BX, ProjSpec{});

    // block_add matrix, result scale 2.0, double target -> out = (a+b)/2.
    std::vector<double> ov(m * n, -777.0), oe(m * n, -777.0);
    MX_Matrix<double, double, int, ColMajor> O(ov.data(), oe.data(), m, n, m, 1, MX_Layout::byColumn);
    block_add(AX, BX, 2.0, O, ProjSpec{});
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
        double want = oproject_real(2.0, av[i][j] + bv[i][j]);
        CHECK(eqd((double)O(i, j), want), "mat block_add @(" << i << "," << j << ")");
    }

    // whole-matrix reduce (default double, raw fold).
    double sum = 0.0, prod = 1.0;
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) { sum += av[i][j]; prod *= av[i][j]; }
    CHECK(block_reduce_add(AX) == sum,  "mat block_reduce_add whole-matrix == fold");
    CHECK(block_reduce_mul(AX) == prod, "mat block_reduce_mul whole-matrix == fold");
}

// =====================================================================================
// 9. Interleaved storage flavors (vector + matrix)
// =====================================================================================
static void test_interleaved() {
    // Interleaved vector: dyadic decode + convert_to_block_max_abs_finite round-trip.
    {
        const int r = 4, M = 9;
        double a_[M] = {4.0, -2.0, 1.0, 0.0, 1.0, 0.5, -0.25, 0.0, 2.0};
        Vector<double, int> a(a_, M);
        std::vector<std::byte> buf(MX_Vector_Interleaved<Elem, Scal, int>::required_bytes(M, r));
        MX_Vector_Interleaved<Elem, Scal, int> v(buf.data(), M, r);
        convert_to_block_max_abs_finite(a, v, ProjSpec{}, ProjSpec{});
        double Z[M]; for (double& z : Z) z = -777.0;
        block_decode(v, Z);
        for (int i = 0; i < M; i++)
            CHECK(Z[i] == a_[i], "interleaved-vec block_decode round-trips dyadic @" << i);
    }
    // Interleaved matrix: convert_from_block dequant.
    {
        const int m = 4, n = 4, tr = 2, tc = 2;
        std::vector<double> d(m * n, 0.0);
        Matrix<double, int, ColMajor> A(d.data(), m, n, m);
        A(0, 0) = 4.0; A(1, 0) = -2.0; A(0, 1) = 1.0;
        A(2, 2) = 2.0; A(3, 2) = -1.0; A(2, 3) = 0.5;
        std::vector<std::byte> buf(MX_Matrix_Interleaved<Elem, Scal, int, ColMajor>::required_bytes(m, n, tr, tc));
        MX_Matrix_Interleaved<Elem, Scal, int, ColMajor> BI(buf.data(), m, n, tr, tc);
        convert_to_block_max_abs_finite(A, BI, ProjSpec{}, ProjSpec{});
        std::vector<double> dn(m * n, -777.0);
        Matrix<double, int, ColMajor> Dn(dn.data(), m, n, m);
        convert_from_block(BI, Dn);
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++)
            CHECK(Dn(i, j) == BI.template scaled_val<double>(i, j),
                  "interleaved-mat convert_from_block dequant @(" << i << "," << j << ")");
    }
}

int main() {
    test_vec_block_decode();
    test_vec_convert_from_block();
    test_vec_convert_to_block();
    test_vec_convert_to_block_max_abs_finite();
    test_vec_block_ops();
    test_vec_reductions();
    test_satmode();
    test_mat_basic();
    test_mat_max_abs();
    test_mat_block_ops_and_reductions();
    test_interleaved();

    if (g_errors == 0) {
        std::cout << "test_block_ops: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_block_ops: " << g_errors << " FAILURES\n";
    return 1;
}
