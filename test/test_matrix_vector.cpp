// Tests for src/Vector.h and src/Matrix.h: Vector, MX_Vector, MX_Vector_Interleaved, Matrix,
// MX_Matrix, MX_Matrix_Interleaved -- construction, element access, copy/quantize utilities,
// slicing, and the type-trait/concept helpers.
//
// See loop/arrays.md for the fault audit + design this test suite verifies (esp. par.6, the
// copy-semantics consistency requirement: regular->MX copy quantizes for real, MX->MX copy
// dequantizes to double and re-microscales per the destination tiling), and
// loop/mx_transpose.md for the MX transpose semantics tested in test_mx_transpose (checks
// (a)-(e) there map to that doc's par.3.4).
//
// Test blocks are built with exact-power-of-two block scales so every private-element ratio
// (elem / scale) lands on a dyadic fraction exactly representable in the target format -- this
// lets checks be bit-exact equality rather than an error-bound, per
// loop/NUMERICAL_TESTING.md rule 2 ("exactly-representable inputs must round-trip exactly").
//
// Harness convention: each test_* function accumulates into the global error counter and
// prints on failure; 0 total errors = pass (loop/NUMERICAL_TESTING.md rule 11).

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

using Elem = ocp_e4m3;   // private-element format (has a NaN/Inf; small dynamic range)
using Scal = ocp_e8m0;   // MX shared-scale format: MUST be Unsigned, power-of-two-only

// =====================================================================================
// 1. Plain Vector
// =====================================================================================
static void test_plain_vector() {
    const int N = 17;
    float buf[N];
    for (int i = 0; i < N; i++) buf[i] = (float)(i - 8) * 0.5f;

    Vector<float, int> v(buf, N);
    CHECK(v.len() == N, "Vector::len");
    for (int i = 0; i < N; i++) {
        CHECK(v[i] == buf[i], "Vector::operator[] readback @" << i);
        CHECK(v(i) == buf[i], "Vector::operator() readback @" << i);
    }

    // arbitrary stride
    float buf2[2 * N];
    for (int i = 0; i < 2 * N; i++) buf2[i] = (float)i;
    Vector<float, int> vs(buf2, N, 2);
    for (int i = 0; i < N; i++) CHECK(vs[i] == buf2[i * 2], "Vector strided access @" << i);

    // cross-type copy (float -> double), exact
    double out[N];
    Vector<double, int> vout(out, N);
    copy(v, vout);
    for (int i = 0; i < N; i++) CHECK(vout[i] == (double)buf[i], "copy(Vector,Vector) exact @" << i);

    // slice
    Vector<float, int> sl = slice(v, range<int>(3, 10));
    CHECK(sl.len() == 7, "slice(Vector) length");
    for (int i = 0; i < 7; i++) CHECK(sl[i] == buf[i + 3], "slice(Vector) content @" << i);

    // poison: writing/reading NaN and Inf must round-trip through the plain accessor untouched
    float poison[4] = {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(), 0.0f};
    Vector<float, int> vp(poison, 4);
    CHECK(std::isnan(vp[0]), "Vector poison: NaN preserved");
    CHECK(std::isinf(vp[1]) && vp[1] > 0, "Vector poison: +Inf preserved");
    CHECK(std::isinf(vp[2]) && vp[2] < 0, "Vector poison: -Inf preserved");
}

// =====================================================================================
// 2. MX_Vector (separate-array storage)
// =====================================================================================
static void test_mx_vector() {
    const int r = 4;
    // 9 elements -> blocks of {4, 4, 1} (ragged last block), block maxima are exact powers of
    // two (4.0, 1.0, 2.0) so scale conversion to Scal (e8m0, power-of-two only) is exact, and
    // every ratio below is an exact dyadic fraction representable in Elem (e4m3).
    const int M = 9;
    float src[M] = {4.0f, -2.0f, 1.0f, 0.0f,      // block0 max=4.0
                     1.0f, 0.5f, -0.25f, 0.0f,     // block1 max=1.0
                     2.0f};                        // block2 (ragged, 1 elem) max=2.0
    Vector<float, int> a(src, M);

    Elem priv[M];
    Scal exps[3]; // ceil(9/4) = 3 blocks
    MX_Vector<Elem, Scal, int> b(priv, exps, M, 3, /*stride=*/1, /*r=*/r);

    MX_real_quantize(a, b);

    CHECK((double)b.get_exp(0) == 4.0, "MX_real_quantize block0 scale");
    CHECK((double)b.get_exp(4) == 1.0, "MX_real_quantize block1 scale");
    CHECK((double)b.get_exp(8) == 2.0, "MX_real_quantize ragged-last-block scale");

    double expect_ratio[M] = {1.0, -0.5, 0.25, 0.0, 1.0, 0.5, -0.25, 0.0, 1.0};
    for (int i = 0; i < M; i++) {
        CHECK((double)b[i] == expect_ratio[i], "MX_real_quantize private elem @" << i);
        // templated operator(): both default (Elem) and an explicit wider R (double)
        CHECK((double)b.template operator()<double>(i) == expect_ratio[i] * (double)b.get_exp(i),
              "MX_Vector::operator()<double> scaled value @" << i);
    }

    // all-zero block must not divide by zero: scale 0, elements 0
    float zsrc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    Vector<float, int> za(zsrc, 4);
    Elem zpriv[4];
    Scal zexp[1];
    MX_Vector<Elem, Scal, int> zb(zpriv, zexp, 4, 1, 1, 4);
    MX_real_quantize(za, zb);
    CHECK((double)zb.get_exp(0) == 0.0, "MX_real_quantize all-zero block scale==0");
    for (int i = 0; i < 4; i++) CHECK((double)zb[i] == 0.0, "MX_real_quantize all-zero block elem @" << i);

    // exp_stride: exponents laid out with a gap (every other slot used). Scales must be
    // powers of two -- Scal is e8m0 (power-of-two-only), anything else would round.
    Scal exps_strided[3 * 2];
    for (auto &e : exps_strided) e = Scal{};
    MX_Vector<Elem, Scal, int> bs(priv, exps_strided, M, 3, 1, r, /*exp_stride=*/2);
    for (int blk = 0; blk < 3; blk++) bs.set_exp(blk * r, static_cast<Scal>((double)(1 << blk)));
    CHECK((double)exps_strided[0] == 1.0 && (double)exps_strided[2] == 2.0 && (double)exps_strided[4] == 4.0,
          "MX_Vector exp_stride places scales at strided offsets");
    for (int blk = 0; blk < 3; blk++)
        CHECK((double)bs.get_exp(blk * r) == (double)(1 << blk), "MX_Vector exp_stride get_exp readback @" << blk);

    // dequantize: copy(MX_Vector, Vector)
    float back[M];
    Vector<float, int> vback(back, M);
    copy(b, vback);
    for (int i = 0; i < M; i++)
        CHECK((double)back[i] == expect_ratio[i] * (double)b.get_exp(i), "copy(MX_Vector,Vector) dequantize @" << i);

    // copy(MX_Vector, MX_Vector): same-shape cross-type copy
    double priv2[M];
    double exps2[3];
    MX_Vector<double, double, int> b2(priv2, exps2, M, 3, 1, r);
    copy(b, b2);
    for (int i = 0; i < M; i++) CHECK(b2[i] == (double)b[i], "copy(MX_Vector,MX_Vector) private elem @" << i);
    for (int blk = 0; blk < 3; blk++)
        CHECK(b2.get_exp(blk * r) == (double)b.get_exp(blk * r), "copy(MX_Vector,MX_Vector) scale @" << blk);

    // slice: aligned sub-range [4,9) == blocks {1,2}
    MX_Vector<Elem, Scal, int> bsl = slice(b, range<int>(4, 9));
    CHECK(bsl.len() == 5, "slice(MX_Vector) length");
    CHECK((double)bsl.get_exp(0) == 1.0, "slice(MX_Vector) block0(=parent block1) scale");
    CHECK((double)bsl.get_exp(4) == 2.0, "slice(MX_Vector) block1(=parent block2) scale");
    for (int i = 0; i < 5; i++) CHECK((double)bsl[i] == (double)b[i + 4], "slice(MX_Vector) content @" << i);
}

// =====================================================================================
// 3. fake_mxcopy (virtual/simulated quantization -- stays a plain Vector)
// =====================================================================================
static void test_fake_mxcopy() {
    const int r = 4;
    const int M = 9; // ragged last block, same layout as test_mx_vector
    float src[M] = {4.0f, -2.0f, 1.0f, 0.0f, 1.0f, 0.5f, -0.25f, 0.0f, 2.0f};
    Vector<float, int> a(src, M);
    float dst[M];
    Vector<float, int> b(dst, M);

    fake_mxcopy<float, int, float, int, Scal, float>(a, b, r);

    double expect[M] = {4.0, -2.0, 1.0, 0.0, 1.0, 0.5, -0.25, 0.0, 2.0};
    for (int i = 0; i < M; i++) CHECK((double)b[i] == expect[i], "fake_mxcopy value @" << i);

    // all-zero block
    float zsrc[3] = {0, 0, 0};
    Vector<float, int> za(zsrc, 3);
    float zdst[3];
    Vector<float, int> zb(zdst, 3);
    fake_mxcopy<float, int, float, int, Scal, float>(za, zb, 4);
    for (int i = 0; i < 3; i++) CHECK(zb[i] == 0.0f, "fake_mxcopy all-zero block @" << i);
}

// =====================================================================================
// 4. MX_Vector_Interleaved
// =====================================================================================
static void test_mx_vector_interleaved() {
    const int r = 4;
    const int M = 9; // ragged last block
    size_t nbytes = MX_Vector_Interleaved<Elem, Scal, int>::required_bytes(M, r);
    CHECK(nbytes == 3 * (sizeof(Scal) + r * sizeof(Elem)), "MX_Vector_Interleaved::required_bytes");

    std::vector<std::byte> buf(nbytes);
    MX_Vector_Interleaved<Elem, Scal, int> v(buf.data(), M, r);
    CHECK(v.num_blocks() == 3, "MX_Vector_Interleaved::num_blocks (ragged)");

    double scales[3] = {4.0, 1.0, 2.0};
    double ratios[M] = {1.0, -0.5, 0.25, 0.0, 1.0, 0.5, -0.25, 0.0, 1.0};
    for (int blk = 0; blk < 3; blk++) v.set_exp(blk * r, static_cast<Scal>(scales[blk]));
    for (int i = 0; i < M; i++) v[i] = static_cast<Elem>(ratios[i]);

    for (int i = 0; i < M; i++) {
        CHECK((double)v[i] == ratios[i], "MX_Vector_Interleaved elem readback @" << i);
        CHECK((double)v.template operator()<double>(i) == ratios[i] * scales[i / r],
              "MX_Vector_Interleaved scaled operator() @" << i);
    }
    for (int blk = 0; blk < 3; blk++)
        CHECK((double)v.get_exp(blk * r) == scales[blk], "MX_Vector_Interleaved get_exp @" << blk);
}

// =====================================================================================
// 5. Plain Matrix
// =====================================================================================
static void test_plain_matrix() {
    const int m = 5, n = 4, ld = m + 2; // arbitrary ld > m (padded)
    std::vector<float> buf(ld * n, -999.0f); // poison the padding/workspace
    Matrix<float, int, ColMajor> A(buf.data(), m, n, ld);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) A(i, j) = (float)(i * 10 + j);

    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            CHECK(A(i, j) == (float)(i * 10 + j), "Matrix(ColMajor,ld>m) element @(" << i << "," << j << ")");

    CHECK(A.rows() == m && A.cols() == n, "Matrix rows/cols");
    CHECK(!A.isNaN() && !A.isInf(), "Matrix isNaN/isInf false on finite data");

    // RowMajor variant
    const int ldr = n + 3;
    std::vector<float> bufr(m * ldr, -999.0f);
    Matrix<float, int, RowMajor> Ar(bufr.data(), m, n, ldr);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) Ar(i, j) = (float)(i * 10 + j);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) CHECK(Ar(i, j) == (float)(i * 10 + j), "Matrix(RowMajor,ld>n) element");

    // poison: NaN/Inf detection
    Ar(2, 1) = std::numeric_limits<float>::quiet_NaN();
    CHECK(Ar.isNaN(), "Matrix isNaN true when a NaN is present");
    Ar(2, 1) = std::numeric_limits<float>::infinity();
    CHECK(Ar.isInf() && !Ar.isNaN(), "Matrix isInf true / isNaN false for +Inf");

    // slice
    Matrix<float, int, ColMajor> sub = slice(A, range<int>(1, 4), range<int>(1, 3));
    CHECK(sub.rows() == 3 && sub.cols() == 2, "slice(Matrix) shape");
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++) CHECK(sub(i, j) == A(i + 1, j + 1), "slice(Matrix) content");

    // transpose round-trip (plain Matrix only -- MX transpose is out of scope, see file header)
    std::vector<float> tbuf(n * m), t2buf(m * n);
    Matrix<float, int, ColMajor> At(tbuf.data(), n, m, n);
    Matrix<float, int, ColMajor> Att(t2buf.data(), m, n, m);
    transpose(A, At);
    transpose(At, Att);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) CHECK(Att(i, j) == A(i, j), "transpose(transpose(A)) == A (ColMajor)");

    // cross-layout transpose
    std::vector<float> trbuf(n * m);
    Matrix<float, int, RowMajor> AtR(trbuf.data(), n, m, m);
    transpose(A, AtR);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) CHECK(AtR(j, i) == A(i, j), "transpose(ColMajor->RowMajor) content");
}

// =====================================================================================
// 6. MX_Matrix (separate-array storage)
// =====================================================================================
static void test_mx_matrix_basic_and_ld_independence() {
    // 6x5 matrix, byColumn grouping with r=4 -> tiles are 4x1 (ragged last row-block: 4,2).
    const int m = 6, n = 5, r = 4;
    double vals[6][5];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) vals[i][j] = 0.0;
    // column 0: block0 rows[0..3] max=4.0, block1 rows[4..5] max=1.0 (both exact powers of two)
    vals[0][0] = 4.0; vals[1][0] = -2.0; vals[2][0] = 1.0; vals[3][0] = 0.0;
    vals[4][0] = 1.0; vals[5][0] = -0.5;
    // column 2: single nonzero exact power of two, rest zero
    vals[1][2] = 2.0;

    // Two physical placements of the same logical data: tight (ld == m) and padded (ld > m).
    // NOTE the MX private-element array shares B's ld, so it must be sized ld*n (not m*n) --
    // and all buffers must outlive the MX views (an earlier version built them in a lambda
    // and returned dangling views).
    const int ld_pad = m + 3;
    std::vector<float> data1(static_cast<size_t>(m) * n, -777.0f);       // poison padding
    std::vector<float> data2(static_cast<size_t>(ld_pad) * n, -777.0f);
    Matrix<float, int, ColMajor> A1(data1.data(), m, n, m);
    Matrix<float, int, ColMajor> A2(data2.data(), m, n, ld_pad);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) { A1(i, j) = (float)vals[i][j]; A2(i, j) = (float)vals[i][j]; }

    std::vector<Elem> priv1(static_cast<size_t>(m) * n);
    std::vector<Elem> priv2(static_cast<size_t>(ld_pad) * n);
    Scal exp_tight[10]; // 2 row-blocks x 5 cols
    Scal exp_padded[10];
    MX_Matrix<Elem, Scal, int, ColMajor> B1(priv1.data(), exp_tight, m, n, m, r, MX_Layout::byColumn);
    MX_Matrix<Elem, Scal, int, ColMajor> B2(priv2.data(), exp_padded, m, n, ld_pad, r, MX_Layout::byColumn);
    MX_real_quantize(A1, B1);
    MX_real_quantize(A2, B2);

    // Bug #10 regression: block membership / scale content must be IDENTICAL regardless of ld.
    for (int k = 0; k < 10; k++)
        CHECK((double)exp_tight[k] == (double)exp_padded[k], "MX_Matrix block scale independent of ld @" << k);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            CHECK((double)B1(i, j) == (double)B2(i, j), "MX_Matrix private elem independent of ld @(" << i << "," << j << ")");

    CHECK((double)B1.get_exp(0, 0) == 4.0, "MX_Matrix byColumn block0 col0 scale");
    CHECK((double)B1.get_exp(4, 0) == 1.0, "MX_Matrix byColumn block1 col0 scale");
    CHECK((double)B1.get_exp(0, 2) == 2.0, "MX_Matrix byColumn block0 col2 scale");
    CHECK((double)B1(0, 0) == 1.0 && (double)B1(1, 0) == -0.5 && (double)B1(2, 0) == 0.25 && (double)B1(3, 0) == 0.0,
          "MX_Matrix byColumn col0 block0 ratios");
    CHECK((double)B1(4, 0) == 1.0 && (double)B1(5, 0) == -0.5, "MX_Matrix byColumn col0 block1 (ragged) ratios");
    CHECK((double)B1(1, 2) == 1.0, "MX_Matrix byColumn col2 nonzero ratio");

    CHECK(!B1.isNaN() && !B1.isInf(), "MX_Matrix isNaN/isInf false on finite quantized data");

    // scaled_val matches (T)(row,col) * scale
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            CHECK(B1.template scaled_val<double>(i, j) == (double)B1(i, j) * (double)B1.get_exp(i, j),
                  "MX_Matrix::scaled_val @(" << i << "," << j << ")");
}

static void test_mx_matrix_byrow_and_bytile() {
    const int m = 4, n = 6;
    // byRow, r=3: tiles are 1x3 -- row 0 has two tiles across columns.
    std::vector<float> data(m * n);
    Matrix<float, int, RowMajor> A(data.data(), m, n, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) A(i, j) = 0.0f;
    A(0, 0) = 4.0f; A(0, 1) = -2.0f; A(0, 2) = 1.0f;   // tile0: max 4.0
    A(0, 3) = 1.0f; A(0, 4) = 0.5f;  A(0, 5) = -0.25f;  // tile1: max 1.0

    std::vector<Elem> priv(m * n);
    std::vector<Scal> exps(m * 2); // 4 rows x ceil(6/3)=2 col-blocks
    MX_Matrix<Elem, Scal, int, RowMajor> B(priv.data(), exps.data(), m, n, n, /*r=*/3, MX_Layout::byRow);
    MX_real_quantize(A, B);

    CHECK((double)B.get_exp(0, 0) == 4.0, "MX_Matrix byRow tile0 scale");
    CHECK((double)B.get_exp(0, 3) == 1.0, "MX_Matrix byRow tile1 scale");
    CHECK((double)B(0, 0) == 1.0 && (double)B(0, 1) == -0.5 && (double)B(0, 2) == 0.25,
          "MX_Matrix byRow tile0 ratios");
    CHECK((double)B(0, 3) == 1.0 && (double)B(0, 4) == 0.5 && (double)B(0, 5) == -0.25,
          "MX_Matrix byRow tile1 ratios");

    // byTile: explicit 2x2 tiles over a 4x4 ColMajor matrix
    const int tm = 4, tn = 4;
    std::vector<float> tdata(tm * tn, 0.0f);
    Matrix<float, int, ColMajor> T(tdata.data(), tm, tn, tm);
    // top-left 2x2 tile: values with max 4.0 (power of two)
    T(0, 0) = 4.0f; T(1, 0) = -2.0f; T(0, 1) = 1.0f; T(1, 1) = 0.0f;
    // bottom-right 2x2 tile (rows 2-3, cols 2-3): max 2.0
    T(2, 2) = 2.0f; T(3, 2) = -1.0f; T(2, 3) = 0.5f; T(3, 3) = 0.0f;

    std::vector<Elem> tpriv(tm * tn);
    std::vector<Scal> texps(2 * 2); // 2x2 tiles
    MX_Matrix<Elem, Scal, int, ColMajor> TB(tpriv.data(), texps.data(), tm, tn, tm, /*tile_rows=*/2, /*tile_cols=*/2);
    MX_real_quantize(T, TB);

    CHECK((double)TB.get_exp(0, 0) == 4.0, "MX_Matrix byTile top-left scale");
    CHECK((double)TB.get_exp(2, 2) == 2.0, "MX_Matrix byTile bottom-right scale");
    CHECK((double)TB.get_exp(0, 2) == 0.0, "MX_Matrix byTile top-right (all-zero) scale");
    CHECK((double)TB(0, 0) == 1.0 && (double)TB(1, 0) == -0.5 && (double)TB(0, 1) == 0.25,
          "MX_Matrix byTile top-left ratios");
    CHECK((double)TB(2, 2) == 1.0 && (double)TB(3, 2) == -0.5 && (double)TB(2, 3) == 0.25,
          "MX_Matrix byTile bottom-right ratios");
}

static void test_mx_matrix_slice_and_lacpy() {
    const int m = 4, n = 4, r = 2;
    std::vector<float> data(m * n, 0.0f);
    Matrix<float, int, ColMajor> A(data.data(), m, n, m);
    A(0, 0) = 4.0f; A(1, 0) = -2.0f;   // tile(0,0) max 4.0
    A(2, 0) = 1.0f; A(3, 0) = 0.5f;    // tile(1,0) max 1.0

    std::vector<Elem> priv(m * n);
    std::vector<Scal> exps(2 * 2); // 2x2 blocks (byColumn: tile_rows=2, tile_cols=1 -> 2 row-blocks x n col-blocks(=n)=4? )
    // byColumn with r=2 -> tile_rows=2, tile_cols=1: num_block_rows=2, num_block_cols=n=4
    exps.resize(2 * n);
    MX_Matrix<Elem, Scal, int, ColMajor> B(priv.data(), exps.data(), m, n, m, r, MX_Layout::byColumn);
    MX_real_quantize(A, B);

    // Slice rows [2,4) x cols [0,2) -- aligned to tile boundaries (tile_rows=2, tile_cols=1)
    MX_Matrix<Elem, Scal, int, ColMajor> S = slice(B, range<int>(2, 4), range<int>(0, 2));
    CHECK(S.rows() == 2 && S.cols() == 2, "slice(MX_Matrix) shape");
    CHECK((double)S.get_exp(0, 0) == 1.0, "slice(MX_Matrix) scale matches parent tile(1,0)");
    CHECK((double)S(0, 0) == 1.0 && (double)S(1, 0) == 0.5, "slice(MX_Matrix) content matches parent");

    // lacpy(MatrixType, MX_MatrixType): scale=1 re-tag stub -- verify it stays a lossless re-tag
    std::vector<float> plain(m * n);
    Matrix<float, int, ColMajor> P(plain.data(), m, n, m);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) P(i, j) = (float)(i + j);
    std::vector<Elem> priv2(m * n);
    std::vector<Scal> exps2(2 * n);
    MX_Matrix<Elem, Scal, int, ColMajor> B2(priv2.data(), exps2.data(), m, n, m, r, MX_Layout::byColumn);
    lacpy(P, B2, Uplo::General);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            CHECK((double)B2.get_exp(i, j) == 1.0, "lacpy(Matrix,MX_Matrix) neutral scale 1.0");
            CHECK((double)B2(i, j) == (double)Elem(P(i, j)), "lacpy(Matrix,MX_Matrix) element re-tag");
        }

    // lacpy(MX_MatrixType, MX_MatrixType): same-shape cross-type copy
    std::vector<double> priv3(m * n);
    std::vector<double> exps3(2 * n);
    MX_Matrix<double, double, int, ColMajor> B3(priv3.data(), exps3.data(), m, n, m, r, MX_Layout::byColumn);
    lacpy(B, B3, Uplo::General);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            CHECK(B3(i, j) == (double)B(i, j), "lacpy(MX_Matrix,MX_Matrix) element");
            CHECK(B3.get_exp(i, j) == (double)B.get_exp(i, j), "lacpy(MX_Matrix,MX_Matrix) scale");
        }
}

// =====================================================================================
// 7. MX_Matrix_Interleaved
// =====================================================================================
static void test_mx_matrix_interleaved() {
    const int m = 4, n = 4, tr = 2, tc = 2;
    size_t nbytes = MX_Matrix_Interleaved<Elem, Scal, int, ColMajor>::required_bytes(m, n, tr, tc);
    CHECK(nbytes == 4 * (sizeof(Scal) + tr * tc * sizeof(Elem)), "MX_Matrix_Interleaved::required_bytes");

    std::vector<std::byte> buf(nbytes);
    MX_Matrix_Interleaved<Elem, Scal, int, ColMajor> M(buf.data(), m, n, tr, tc);
    CHECK(M.num_block_rows() == 2 && M.num_block_cols() == 2, "MX_Matrix_Interleaved block grid shape");

    M.set_exp(0, 0, static_cast<Scal>(4.0));
    M(0, 0) = static_cast<Elem>(1.0); M(1, 0) = static_cast<Elem>(-0.5);
    M(0, 1) = static_cast<Elem>(0.25); M(1, 1) = static_cast<Elem>(0.0);

    M.set_exp(2, 2, static_cast<Scal>(2.0));
    M(2, 2) = static_cast<Elem>(1.0); M(3, 2) = static_cast<Elem>(-0.5);
    M(2, 3) = static_cast<Elem>(0.25); M(3, 3) = static_cast<Elem>(0.0);

    CHECK((double)M.get_exp(0, 0) == 4.0 && (double)M.get_exp(2, 2) == 2.0, "MX_Matrix_Interleaved get_exp readback");
    CHECK((double)M(0, 0) == 1.0 && (double)M(1, 0) == -0.5 && (double)M(0, 1) == 0.25,
          "MX_Matrix_Interleaved element readback tile(0,0)");
    CHECK((double)M(2, 2) == 1.0 && (double)M(3, 2) == -0.5 && (double)M(2, 3) == 0.25,
          "MX_Matrix_Interleaved element readback tile(1,1)");
    CHECK(M.template scaled_val<double>(0, 0) == 4.0 && M.template scaled_val<double>(1, 0) == -2.0,
          "MX_Matrix_Interleaved scaled_val");
}

// =====================================================================================
// 8. copy() semantics, vector side (arrays.md par.6)
// =====================================================================================
static void test_copy_semantics_vector() {
    const int r = 4, M = 9;
    float src[M] = {4.0f, -2.0f, 1.0f, 0.0f, 1.0f, 0.5f, -0.25f, 0.0f, 2.0f};
    Vector<float, int> a(src, M);

    // copy(Vector -> MX_Vector) IS quantization: must match MX_real_quantize bit-for-bit
    Elem p1[M], p2[M];
    Scal e1[3], e2[3];
    MX_Vector<Elem, Scal, int> b1(p1, e1, M, 3, 1, r);
    MX_Vector<Elem, Scal, int> b2(p2, e2, M, 3, 1, r);
    MX_real_quantize(a, b1);
    copy(a, b2);
    for (int i = 0; i < M; i++) CHECK((double)b2[i] == (double)b1[i], "copy(Vector,MX_Vector)==quantize elem @" << i);
    for (int blk = 0; blk < 3; blk++)
        CHECK((double)b2.get_exp(blk * r) == (double)b1.get_exp(blk * r), "copy(Vector,MX_Vector)==quantize scale @" << blk);

    // copy(MX -> MX) with a DIFFERENT block size requantizes: r=4 source -> r=2 destination.
    // Dyadic data: dequantized values must be preserved exactly and each dst scale must be
    // the true per-2-block max of the dequantized source (double oracle).
    const int M2 = 8;
    float src2[M2] = {4.0f, -2.0f, 1.0f, 0.0f, 1.0f, 0.5f, -0.25f, 0.0f};
    Vector<float, int> a2(src2, M2);
    Elem sp[M2];
    Scal se[2];
    MX_Vector<Elem, Scal, int> bsrc(sp, se, M2, 2, 1, 4);
    MX_real_quantize(a2, bsrc);

    Elem dp[M2];
    Scal de[4];
    MX_Vector<Elem, Scal, int> bdst(dp, de, M2, 4, 1, 2);
    copy(bsrc, bdst);
    for (int i = 0; i < M2; i++)
        CHECK((double)bdst[i] * (double)bdst.get_exp(i) == (double)bsrc[i] * (double)bsrc.get_exp(i),
              "copy(MX,MX) r=4->r=2 dequant preserved @" << i);
    for (int blk = 0; blk < 4; blk++) {
        double mx = 0.0;
        for (int j = 0; j < 2; j++)
            mx = std::max(mx, std::fabs((double)bsrc[blk * 2 + j] * (double)bsrc.get_exp(blk * 2 + j)));
        CHECK((double)bdst.get_exp(blk * 2) == mx, "copy(MX,MX) r=4->r=2 dst scale is per-block max @" << blk);
    }

    // quantize/dequantize through INTERLEAVED storage via the same generic copy surface
    std::vector<std::byte> ibuf(MX_Vector_Interleaved<Elem, Scal, int>::required_bytes(M, r));
    MX_Vector_Interleaved<Elem, Scal, int> vi(ibuf.data(), M, r);
    copy(a, vi); // quantize into interleaved
    for (int i = 0; i < M; i++) CHECK((double)vi[i] == (double)b1[i], "copy(Vector,MX_interleaved) elem @" << i);
    for (int blk = 0; blk < 3; blk++)
        CHECK((double)vi.get_exp(blk * r) == (double)b1.get_exp(blk * r), "copy(Vector,MX_interleaved) scale @" << blk);
    float back[M];
    for (auto &x : back) x = -999.0f; // poison
    Vector<float, int> vb(back, M);
    copy(vi, vb); // dequantize out of interleaved
    for (int i = 0; i < M; i++)
        CHECK((double)back[i] == (double)b1[i] * (double)b1.get_exp(i), "copy(MX_interleaved,Vector) dequant @" << i);
}

// =====================================================================================
// 9. copy() semantics, matrix side (arrays.md par.6)
// =====================================================================================
static void test_copy_semantics_matrix() {
    const int m = 4, n = 4;
    std::vector<float> data(m * n, 0.0f);
    Matrix<float, int, ColMajor> A(data.data(), m, n, m);
    A(0, 0) = 4.0f; A(1, 0) = -2.0f; A(0, 1) = 1.0f; A(1, 1) = 0.0f;
    A(2, 2) = 2.0f; A(3, 2) = -1.0f; A(2, 3) = 0.5f; A(3, 3) = 0.0f;

    // copy(plain -> MX, General) IS quantization: must match MX_real_quantize bit-for-bit
    std::vector<Elem> p1(m * n), p2(m * n);
    std::vector<Scal> e1(2 * 2), e2(2 * 2);
    MX_Matrix<Elem, Scal, int, ColMajor> B1(p1.data(), e1.data(), m, n, m, 2, 2);
    MX_Matrix<Elem, Scal, int, ColMajor> B2(p2.data(), e2.data(), m, n, m, 2, 2);
    MX_real_quantize(A, B1);
    copy(A, B2);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            CHECK((double)B2(i, j) == (double)B1(i, j), "copy(Matrix,MX_Matrix)==quantize elem @(" << i << "," << j << ")");
            CHECK((double)B2.get_exp(i, j) == (double)B1.get_exp(i, j), "copy(Matrix,MX_Matrix)==quantize scale");
        }

    // copy(MX -> MX) with a DIFFERENT tiling requantizes: 2x2 byTile -> byColumn r=2.
    // Dyadic data: dequant preserved exactly; dst scales are the per-dst-tile max (oracle).
    std::vector<Elem> p3(m * n);
    std::vector<Scal> e3(2 * n);
    MX_Matrix<Elem, Scal, int, ColMajor> B3(p3.data(), e3.data(), m, n, m, 2, MX_Layout::byColumn);
    copy(B1, B3);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            CHECK(B3.template scaled_val<double>(i, j) == B1.template scaled_val<double>(i, j),
                  "copy(MX,MX) retile dequant preserved @(" << i << "," << j << ")");
    for (int j = 0; j < n; j++)
        for (int bi = 0; bi < 2; bi++) {
            double mx = 0.0;
            for (int i = bi * 2; i < bi * 2 + 2; i++)
                mx = std::max(mx, std::fabs(B1.template scaled_val<double>(i, j)));
            CHECK((double)B3.get_exp(bi * 2, j) == mx, "copy(MX,MX) retile dst scale @(" << bi << "," << j << ")");
        }

    // copy(MX -> plain): dequantize
    std::vector<float> dense(m * n, -999.0f); // poison
    Matrix<float, int, ColMajor> Dn(dense.data(), m, n, m);
    copy(B1, Dn);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            CHECK((double)Dn(i, j) == B1.template scaled_val<double>(i, j),
                  "copy(MX_Matrix,Matrix) dequant @(" << i << "," << j << ")");

    // lacpy(MX -> plain, Upper): dequantizes the triangle only, leaves the rest untouched
    std::vector<float> tri(m * n, -999.0f);
    Matrix<float, int, ColMajor> Tri(tri.data(), m, n, m);
    lacpy(B1, Tri, Uplo::Upper);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            if (j >= i)
                CHECK((double)Tri(i, j) == B1.template scaled_val<double>(i, j), "lacpy(MX,plain,Upper) triangle");
            else
                CHECK(Tri(i, j) == -999.0f, "lacpy(MX,plain,Upper) below-diagonal untouched");
        }

    // copy into INTERLEAVED matrix storage via the same wrapper (quantize), then back out
    std::vector<std::byte> ibuf(MX_Matrix_Interleaved<Elem, Scal, int, ColMajor>::required_bytes(m, n, 2, 2));
    MX_Matrix_Interleaved<Elem, Scal, int, ColMajor> BI(ibuf.data(), m, n, 2, 2);
    copy(A, BI);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            CHECK((double)BI(i, j) == (double)B1(i, j), "copy(Matrix,MX_interleaved) elem @(" << i << "," << j << ")");
            CHECK((double)BI.get_exp(i, j) == (double)B1.get_exp(i, j), "copy(Matrix,MX_interleaved) scale");
        }
    std::vector<float> dense2(m * n, -999.0f);
    Matrix<float, int, ColMajor> Dn2(dense2.data(), m, n, m);
    copy(BI, Dn2);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            CHECK((double)Dn2(i, j) == B1.template scaled_val<double>(i, j), "copy(MX_interleaved,Matrix) dequant");
}

// =====================================================================================
// 10. MX transpose (loop/mx_transpose.md par.3.4 checks (a)-(e) + lossy bound)
// =====================================================================================
static void test_mx_transpose() {
    // Source: 4x6 ColMajor, every value a power of two (or 0) so all tile maxima are exact
    // in e8m0 and all ratios exact in e4m3; column 3 all-zero for check (e).
    const int m = 4, n = 6, r = 2;
    std::vector<float> dbuf(m * n);
    Matrix<float, int, ColMajor> D(dbuf.data(), m, n, m);
    const float pool[6] = {4.0f, -2.0f, 1.0f, -0.5f, 0.25f, 2.0f};
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) D(i, j) = (j == 3) ? 0.0f : pool[(i * 7 + j * 5) % 6];

    std::vector<Elem> apriv(m * n);
    std::vector<Scal> aexp((m / r) * n);
    MX_Matrix<Elem, Scal, int, ColMajor> A(apriv.data(), aexp.data(), m, n, m, r, MX_Layout::byColumn);
    MX_real_quantize(D, A);

    // Transpose with the transpose-image tiling (byColumn r=2 -> byRow r=2): lossless path.
    std::vector<Elem> tpriv(n * m);
    std::vector<Scal> texp(n * (m / r));
    MX_Matrix<Elem, Scal, int, ColMajor> At(tpriv.data(), texp.data(), n, m, n, r, transpose_image(A.mx_layout));
    CHECK(At.tile_rows == 1 && At.tile_cols == 2, "transpose_image(byColumn) gives 1x2 tiles");
    transpose(A, At);

    // (b) dequant identity, exact on dyadic data
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            CHECK(At.template scaled_val<double>(j, i) == A.template scaled_val<double>(i, j),
                  "MX transpose dequant identity @(" << i << "," << j << ")");

    // (c) every destination tile's scale == max |dequant(A)| over that tile (double oracle)
    for (int j = 0; j < n; j++)         // At block-row j (tile_rows=1)
        for (int bk = 0; bk < 2; bk++) { // At block-col bk (tile_cols=2)
            double mx = 0.0;
            for (int c = bk * 2; c < bk * 2 + 2; c++)
                mx = std::max(mx, std::fabs(A.template scaled_val<double>(c, j)));
            CHECK((double)At.get_exp(j, bk * 2) == mx, "MX transpose dst scale @(" << j << "," << bk << ")");
        }

    // (e) all-zero source column 3 -> all-zero At row 3: scales 0, elements 0
    for (int bk = 0; bk < 2; bk++) CHECK((double)At.get_exp(3, bk * 2) == 0.0, "MX transpose zero tile scale");
    for (int c = 0; c < m; c++) CHECK((double)At(3, c) == 0.0, "MX transpose zero tile elem @" << c);

    // (a) exact round-trip: transpose back with A's own tiling, bit-compare
    std::vector<Elem> rpriv(m * n);
    std::vector<Scal> rexp((m / r) * n);
    MX_Matrix<Elem, Scal, int, ColMajor> Att(rpriv.data(), rexp.data(), m, n, m, r, MX_Layout::byColumn);
    transpose(At, Att);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            CHECK((double)Att(i, j) == (double)A(i, j), "transpose(transpose(A))==A elem @(" << i << "," << j << ")");
            CHECK((double)Att.get_exp(i, j) == (double)A.get_exp(i, j), "transpose(transpose(A))==A scale");
        }

    // (d) ragged shapes: 5x3 with 2x1 tiles -> 3x5 with 1x2 tiles (ragged col-blocks)
    const int rm = 5, rn = 3;
    std::vector<float> rdbuf(rm * rn);
    Matrix<float, int, ColMajor> RD(rdbuf.data(), rm, rn, rm);
    for (int j = 0; j < rn; j++)
        for (int i = 0; i < rm; i++) RD(i, j) = pool[(i * 5 + j * 7) % 6];
    std::vector<Elem> rap(rm * rn);
    std::vector<Scal> rae(3 * rn); // ceil(5/2)=3 row-blocks x 3 cols
    MX_Matrix<Elem, Scal, int, ColMajor> RA(rap.data(), rae.data(), rm, rn, rm, r, MX_Layout::byColumn);
    MX_real_quantize(RD, RA);
    std::vector<Elem> rtp(rn * rm);
    std::vector<Scal> rte(rn * 3);
    MX_Matrix<Elem, Scal, int, ColMajor> RAt(rtp.data(), rte.data(), rn, rm, rn, r, MX_Layout::byRow);
    transpose(RA, RAt);
    for (int i = 0; i < rm; i++)
        for (int j = 0; j < rn; j++)
            CHECK(RAt.template scaled_val<double>(j, i) == RA.template scaled_val<double>(i, j),
                  "MX transpose ragged dequant identity @(" << i << "," << j << ")");
    std::vector<Elem> rrp(rm * rn);
    std::vector<Scal> rre(3 * rn);
    MX_Matrix<Elem, Scal, int, ColMajor> RAtt(rrp.data(), rre.data(), rm, rn, rm, r, MX_Layout::byColumn);
    transpose(RAt, RAtt);
    for (int j = 0; j < rn; j++)
        for (int i = 0; i < rm; i++) {
            CHECK((double)RAtt(i, j) == (double)RA(i, j), "MX transpose ragged round-trip elem");
            CHECK((double)RAtt.get_exp(i, j) == (double)RA.get_exp(i, j), "MX transpose ragged round-trip scale");
        }

    // Interleaved destination through the same generic transpose (image tiling, lossless)
    std::vector<std::byte> ibuf(MX_Matrix_Interleaved<Elem, Scal, int, ColMajor>::required_bytes(n, m, 1, 2));
    MX_Matrix_Interleaved<Elem, Scal, int, ColMajor> AtI(ibuf.data(), n, m, 1, 2);
    transpose(A, AtI);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            CHECK(AtI.template scaled_val<double>(j, i) == A.template scaled_val<double>(i, j),
                  "MX transpose to interleaved dequant identity @(" << i << "," << j << ")");

    // (bonus) lossy path: NON-image tiling (same-direction byColumn) with a double scale
    // format, data chosen so a destination tile's max (3.75) is not a power of two -> at
    // least one element genuinely re-rounds; every element obeys the 0.5*ulp(ratio)*scale
    // bound of mx_transpose.md par.2.2 (ulp floored at the e4m3 subnormal step below 2^-6).
    const int lm = 2, ln = 2;
    float lvals[lm][ln] = {{3.7f, 0.9f}, {1.0f, 0.5f}};
    std::vector<float> lbuf(lm * ln);
    Matrix<float, int, ColMajor> LD(lbuf.data(), lm, ln, lm);
    for (int j = 0; j < ln; j++)
        for (int i = 0; i < lm; i++) LD(i, j) = lvals[i][j];
    std::vector<Elem> lap(lm * ln);
    std::vector<Scal> lae(1 * ln);
    MX_Matrix<Elem, Scal, int, ColMajor> LA(lap.data(), lae.data(), lm, ln, lm, 2, MX_Layout::byColumn);
    MX_real_quantize(LD, LA);
    std::vector<Elem> ltp(ln * lm);
    std::vector<double> lte(1 * lm);
    MX_Matrix<Elem, double, int, ColMajor> LAt(ltp.data(), lte.data(), ln, lm, ln, 2, MX_Layout::byColumn);
    transpose(LA, LAt);
    int lossy_diffs = 0;
    for (int i = 0; i < lm; i++)
        for (int j = 0; j < ln; j++) {
            double v = LA.template scaled_val<double>(i, j);
            double w = LAt.template scaled_val<double>(j, i);
            double s = LAt.get_exp(j, i);
            if (s == 0.0) { CHECK(w == 0.0, "MX transpose lossy zero tile"); continue; }
            double ratio = std::fabs(v) / s;
            double ulp = (ratio >= std::exp2(-6)) ? std::exp2(std::floor(std::log2(ratio)) - 3)
                                                  : std::exp2(-9); // e4m3 subnormal step
            CHECK(std::fabs(w - v) <= 0.5 * ulp * s + 1e-15,
                  "MX transpose lossy bound @(" << i << "," << j << ") v=" << v << " w=" << w);
            if (w != v) lossy_diffs++;
        }
    CHECK(lossy_diffs > 0, "MX transpose lossy path actually re-rounds (test has teeth)");
}

// =====================================================================================
// 11. pack / transpose_and_pack
// =====================================================================================
static void test_pack_functions() {
    const int m = 3, n = 4;
    std::vector<float> a(m * n);
    Matrix<float, int, ColMajor> A(a.data(), m, n, m);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) A(i, j) = (float)i + 0.5f * (float)j; // dyadic, exact in e4m3

    // pack: elementwise cast copy into a (possibly different-typed) tightly packed matrix
    std::vector<double> p(m * n, -999.0);
    Matrix<double, int, ColMajor> Ap(p.data(), m, n, m);
    pack(A, Ap);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            CHECK(Ap(i, j) == (double)A(i, j), "pack(Matrix,Matrix) @(" << i << "," << j << ")");

    // transpose_and_pack: transposed scale=1 re-tag into an MX matrix
    std::vector<Elem> tp(n * m);
    std::vector<Scal> te(n * m); // r=1 -> one scale slot per element
    MX_Matrix<Elem, Scal, int, ColMajor> Apk(tp.data(), te.data(), n, m, n, 1, MX_Layout::byColumn);
    transpose_and_pack(A, Apk);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            CHECK((double)Apk(j, i) == (double)static_cast<Elem>(A(i, j)), "transpose_and_pack elem @(" << i << "," << j << ")");
            CHECK((double)Apk.get_exp(j, i) == 1.0, "transpose_and_pack neutral scale");
        }
}

// =====================================================================================
// 12. Non-e8m0 scale formats (arrays.md par.6: "any unsigned floating-point format")
// =====================================================================================
// Everything above pins Scal = e8m0 (the OCP MX standard) because its power-of-two-only
// scales keep every check bit-exact. This section uses an unsigned 8-bit P3109 format WITH
// mantissa bits (p=3) as the scale, exercising the path where static_cast<T_scal>(max)
// itself rounds the scale (mx_transpose.md par.2.3): when the block max is representable the
// result is still exact; when it is not, elements re-round once and must obey the
// 0.5*ulp(ratio)*scale bound (with ratio > 1 possible if the scale rounded down -- e4m3 is
// Saturating, so the result must stay finite).
using UScal = P_3109_float<8, 3, lo_float::Signedness::Unsigned>; // unsigned, 2 mantissa bits

static double e4m3_ulp(double ratio) {
    double a = std::fabs(ratio);
    // e4m3: 3 mantissa bits; min normal 2^-6 -> fixed subnormal step 2^-9 below it
    return (a >= std::exp2(-6)) ? std::exp2(std::floor(std::log2(a)) - 3) : std::exp2(-9);
}

static void test_mx_alternative_scale_formats() {
    // (a) Block max REPRESENTABLE in UScal (1.5 = 1.1b x 2^0): quantization stays exact.
    {
        const int r = 4, M = 4;
        float src[M] = {1.5f, 0.75f, -0.375f, 0.0f}; // ratios {1, 0.5, -0.25, 0}: exact in e4m3
        Vector<float, int> a(src, M);
        Elem p[M];
        UScal e[1];
        MX_Vector<Elem, UScal, int> b(p, e, M, 1, 1, r);
        MX_real_quantize(a, b);
        CHECK((double)b.get_exp(0) == 1.5, "UScal representable block max stored exactly");
        double expect[M] = {1.0, 0.5, -0.25, 0.0};
        for (int i = 0; i < M; i++) {
            CHECK((double)b[i] == expect[i], "UScal exact-case ratio @" << i);
            CHECK((double)b[i] * (double)b.get_exp(i) == (double)src[i], "UScal exact-case dequant @" << i);
        }
    }

    // (b) Block max NOT representable in UScal: the scale is the static_cast of the true max
    // (double oracle for the max; the cast is the documented MX_real_quantize semantic), and
    // every element obeys the half-ULP-of-ratio bound; results stay finite (saturating e4m3).
    {
        const int r = 4, M = 8;
        float src[M] = {1.7f, 0.85f, -0.4f, 0.0f,   // block0: max 1.7 (between UScal grid pts)
                        1.6f, -1.6f, 0.2f, 0.1f};   // block1: max 1.6 (may round DOWN -> ratio > 1)
        Vector<float, int> a(src, M);
        Elem p[M];
        UScal e[2];
        MX_Vector<Elem, UScal, int> b(p, e, M, 2, 1, r);
        MX_real_quantize(a, b);

        for (int blk = 0; blk < 2; blk++) {
            double true_max = 0.0;
            for (int j = 0; j < r; j++) true_max = std::max(true_max, std::fabs((double)src[blk * r + j]));
            CHECK((double)b.get_exp(blk * r) == (double)static_cast<UScal>(true_max),
                  "UScal scale is cast of true block max @" << blk);
        }
        int rerounded = 0;
        for (int i = 0; i < M; i++) {
            double s = (double)b.get_exp(i);
            double got = (double)b[i] * s;
            double ratio = std::fabs((double)src[i]) / s;
            CHECK(std::isfinite((double)b[i]), "UScal non-representable-max elem finite @" << i);
            CHECK(std::fabs(got - (double)src[i]) <= 0.5 * e4m3_ulp(ratio) * s + 1e-15,
                  "UScal half-ULP dequant bound @" << i << " src=" << src[i] << " got=" << got);
            if (got != (double)src[i]) rerounded++;
        }
        CHECK(rerounded > 0, "UScal non-power-of-two scale path actually re-rounds (teeth)");
    }

    // (c) Same format through the matrix quantize + MX transpose (image tiling). NOTE the
    // sharpened condition E (mx_transpose.md par.2.3): bit-preservation additionally needs
    // max|ratio| == 1 in every tile, i.e. each tile max must be REPRESENTABLE in UScal (if
    // the quantize's scale cast rounded the max down, stored ratios exceed 1 and the
    // transpose legitimately recomputes a LARGER scale -- caught by this test's first
    // version). So the maxima here (1.5, 1.25, 1.75, 0) all sit on the UScal grid.
    {
        const int m = 2, n = 4, r = 2;
        float vals[m][n] = {{1.5f, 1.25f, 1.75f, 0.0f}, {0.75f, -0.625f, 0.4375f, 0.0f}};
        std::vector<float> dbuf(m * n);
        Matrix<float, int, ColMajor> D(dbuf.data(), m, n, m);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) D(i, j) = vals[i][j];
        std::vector<Elem> ap(m * n);
        std::vector<UScal> ae(1 * n);
        MX_Matrix<Elem, UScal, int, ColMajor> A(ap.data(), ae.data(), m, n, m, r, MX_Layout::byColumn);
        MX_real_quantize(D, A);
        std::vector<Elem> tp(n * m);
        std::vector<UScal> te(n * 1);
        MX_Matrix<Elem, UScal, int, ColMajor> At(tp.data(), te.data(), n, m, n, r, MX_Layout::byRow);
        transpose(A, At);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                CHECK(At.template scaled_val<double>(j, i) == A.template scaled_val<double>(i, j),
                      "UScal image-tiling transpose bit-preserving @(" << i << "," << j << ")");
        std::vector<Elem> rp(m * n);
        std::vector<UScal> re(1 * n);
        MX_Matrix<Elem, UScal, int, ColMajor> Att(rp.data(), re.data(), m, n, m, r, MX_Layout::byColumn);
        transpose(At, Att);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                CHECK((double)Att(i, j) == (double)A(i, j), "UScal transpose round-trip elem");
                CHECK((double)Att.get_exp(i, j) == (double)A.get_exp(i, j), "UScal transpose round-trip scale");
            }
    }
}

// =====================================================================================
// 13. Type traits / concepts
// =====================================================================================
static void test_traits() {
    static_assert(is_VectorType_t<Vector<float, int>>, "is_VectorType_t true for Vector");
    static_assert(!is_VectorType_t<int>, "is_VectorType_t false for non-vector");
    static_assert(is_MXVectorType_t<MX_Vector<Elem, Scal, int>>, "is_MXVectorType_t true for MX_Vector");
    static_assert(!is_MXVectorType_t<Vector<float, int>>, "is_MXVectorType_t false for plain Vector");

    static_assert(is_Vanilla_MatrixType_t<Matrix<float, int, ColMajor>>, "is_Vanilla_MatrixType_t true for Matrix");
    static_assert(!is_Vanilla_MatrixType_t<int>, "is_Vanilla_MatrixType_t false for non-matrix");
    static_assert(is_MX_MatrixType_t<MX_Matrix<Elem, Scal, int, ColMajor>>, "is_MX_MatrixType_t true for MX_Matrix");
    static_assert(!is_MX_MatrixType_t<Matrix<float, int, ColMajor>>, "is_MX_MatrixType_t false for plain Matrix");

    static_assert(!is_MX_format<Matrix<float, int, ColMajor>>::value, "is_MX_format false for plain Matrix");
    static_assert(is_MX_format<MX_Matrix<Elem, Scal, int, ColMajor>>::value, "is_MX_format true for MX_Matrix");

    static_assert(is_MXVectorInterleavedType_t<MX_Vector_Interleaved<Elem, Scal, int>>,
                  "is_MXVectorInterleavedType_t true for MX_Vector_Interleaved");
    static_assert(!is_MXVectorInterleavedType_t<MX_Vector<Elem, Scal, int>>,
                  "is_MXVectorInterleavedType_t false for MX_Vector");
    static_assert(AnyMXMatrixType<MX_Matrix_Interleaved<Elem, Scal, int, ColMajor>>,
                  "AnyMXMatrixType true for MX_Matrix_Interleaved");
    static_assert(AnyMXMatrixType<MX_Matrix<Elem, Scal, int, ColMajor>>,
                  "AnyMXMatrixType true for MX_Matrix");
    static_assert(!AnyMXMatrixType<Matrix<float, int, ColMajor>>,
                  "AnyMXMatrixType false for plain Matrix");
}

int main() {
    test_plain_vector();
    test_mx_vector();
    test_fake_mxcopy();
    test_mx_vector_interleaved();
    test_plain_matrix();
    test_mx_matrix_basic_and_ld_independence();
    test_mx_matrix_byrow_and_bytile();
    test_mx_matrix_slice_and_lacpy();
    test_mx_matrix_interleaved();
    test_copy_semantics_vector();
    test_copy_semantics_matrix();
    test_mx_transpose();
    test_pack_functions();
    test_mx_alternative_scale_formats();
    test_traits();

    if (g_errors == 0) {
        std::cout << "test_matrix_vector: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_matrix_vector: " << g_errors << " FAILURES\n";
    return 1;
}
