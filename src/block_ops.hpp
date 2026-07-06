/// @author LoFloat loop
/// IEEE P3109 (interim) §5 block operations on the MicroScaled objects (MX_Vector / MX_Matrix and
/// their interleaved flavors). Spec: loop/block_ops_3109.md; frozen API contract:
/// loop/block_ops_api.md. Vector and matrix ops both live in namespace lo_float (where MX_Vector* /
/// MX_Matrix* live); shared per-element primitives in
/// lo_float::block_detail. Elementwise ops are LOFLOAT_HOST_DEVICE (device-safe math on double);
/// reductions are host-only for v1 (a serial double fold is the wrong GPU unit — a device path
/// delegates to the existing parallel reduce / Dot.hpp cub path later).
///
/// DEVICE CAVEAT (v1): the elementwise ops compile under nvcc and are bit-exact CPU<->GPU for the
/// device-safe surface — block_decode / block_project_elem into a WIDE Out (e.g. double) and the
/// decode buffer. Projecting into a NARROW typed Out on device (e.g. Round<ocp_e4m3>) currently
/// returns a zero-magnitude result on device: the typed lo_float ConvertImpl (strategy-1 "native
/// typed values") narrow conversion is NOT device-safe (Round<narrow>(x) -> 0 in nvcc device code,
/// sign preserved), a PRE-EXISTING lo_float.h limitation distinct from the already-fixed
/// virtual_round(params) path. Until that core bug is fixed (tracked in BACKLOG.md "Bugs"), use the
/// HOST path for any narrow-Out block op; the GPU-execution container path was deferred by design
/// anyway (see loop/block_ops_3109.md / block_ops_api.md).
#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <limits>
#include <type_traits>
#include "lo_float.h"
#include "Vector.h"
#include "Matrix.h"

namespace lo_float {
namespace block_detail {

// ωBlockDecode of one element (§1). ∞×0 / 0×∞ -> NaN come free from IEEE double.
LOFLOAT_HOST_DEVICE inline double block_decode_elem(double elem, double scale) noexcept {
    return elem * scale;
}

// ωBlockProject of one real X against scale S, projected into Out (§2/§6.2). Guard order is
// first-match-wins and must not be reordered. Not literally single-rounding (real->double->Out),
// but for the narrow targets here it equals single-rounding; for a power-of-two scale (e8m0) X/S
// is exact.
template <class Out>
LOFLOAT_HOST_DEVICE inline Out block_project_elem(double S, double X, ProjSpec ps) noexcept {
    if (std::isnan(S) || std::isnan(X))
        return lo_float::Round<Out>(std::numeric_limits<double>::quiet_NaN(), ps);   // NaN
    if (S == 0.0)
        return lo_float::Round<Out>(0.0, ps);                                        // NOTE 1
    if (std::isinf(S)) {
        double v = (X == 0.0) ? 0.0 : std::copysign(1.0, X) * std::copysign(1.0, S); // NOTE 2: ±1
        return lo_float::Round<Out>(v, ps);
    }
    return lo_float::Round<Out>(X / S, ps);                                          // X/±∞ -> 0
}

} // namespace block_detail
} // namespace lo_float


// ============================ VECTOR ops (namespace lo_float) ============================
namespace lo_float {

// --- 0. block_decode: decode to a real (double) buffer (§1) ---
template <AnyMXVectorType Src>
LOFLOAT_HOST_DEVICE void block_decode(const Src& a, double* Z) {
    for (decltype(a.m) i = 0; i < a.m; ++i)
        Z[i] = lo_float::block_detail::block_decode_elem(static_cast<double>(a[i]),
                                                         static_cast<double>(a.get_exp(i)));
}

// --- 1. convert_from_block: block -> dense (ConvertFromBlock, §3) ---
template <AnyMXVectorType Src, VectorType Dst>
LOFLOAT_HOST_DEVICE void convert_from_block(const Src& a, Dst& b, ProjSpec ps = ProjSpec{}) {
    assert(a.m == b.m);
    using Tb = typename Dst::scalar_type;
    for (decltype(a.m) i = 0; i < a.m; ++i) {
        double dec = lo_float::block_detail::block_decode_elem(static_cast<double>(a[i]),
                                                               static_cast<double>(a.get_exp(i)));
        b[i] = lo_float::Round<Tb>(dec, ps);
    }
}

// --- 2. convert_to_block: dense -> block with CALLER scale (ConvertToBlock, §4) ---
template <VectorType Src, AnyMXVectorType Dst>
LOFLOAT_HOST_DEVICE void convert_to_block(const Src& a, typename Dst::shared_exp_type s, Dst& b,
                                          ProjSpec ps = ProjSpec{}) {
    assert(a.m == b.m);
    using Tb = typename Dst::scalar_type;
    const double S = static_cast<double>(s);
    decltype(b.m) i = 0;
    while (i < b.m) {
        auto blk = std::min(b.r, b.m - i);
        b.set_exp(i, s);
        for (decltype(blk) j = 0; j < blk; ++j)
            b[i + j] = lo_float::block_detail::block_project_elem<Tb>(S, static_cast<double>(a[i + j]), ps);
        i += blk;
    }
}

// --- 3. convert_to_block_max_abs_finite: dense -> block, COMPUTED scale (§5) ---
// Fully uniform: classify the block to fix the STORED scale, then every element is
// block_project_elem(double(stored_scale), x, ps) — the NaN/zero/±∞ cases fall out of the primitive.
template <VectorType Src, AnyMXVectorType Dst>
LOFLOAT_HOST_DEVICE void convert_to_block_max_abs_finite(const Src& a, Dst& b,
                                                         ProjSpec ps_scale = ProjSpec{},
                                                         ProjSpec ps = ProjSpec{}) {
    assert(a.m == b.m);
    using Tb = typename Dst::scalar_type;
    using Ts = typename Dst::shared_exp_type;
    decltype(b.m) i = 0;
    while (i < b.m) {
        auto blk = std::min(b.r, b.m - i);
        double Sf = 0.0;
        bool has_inf = false, has_nan = false;
        for (decltype(blk) j = 0; j < blk; ++j) {
            double x = static_cast<double>(a[i + j]);
            if (std::isnan(x))      has_nan = true;
            else if (std::isinf(x)) has_inf = true;
            else { double mag = std::fabs(x); if (mag > Sf) Sf = mag; }
        }
        Ts scale;
        if (Sf > 0.0)        scale = lo_float::Round<Ts>(Sf, ps_scale);                              // NOTE 3
        else if (has_inf)    scale = lo_float::Round<Ts>(std::numeric_limits<double>::infinity(), ps_scale); // NOTE 2
        else if (has_nan)    scale = lo_float::Round<Ts>(std::numeric_limits<double>::quiet_NaN(), ps_scale); // NOTE 1
        else                 scale = Ts{};                                                            // all zero
        b.set_exp(i, scale);
        const double S = static_cast<double>(scale);
        for (decltype(blk) j = 0; j < blk; ++j)
            b[i + j] = lo_float::block_detail::block_project_elem<Tb>(S, static_cast<double>(a[i + j]), ps);
        i += blk;
    }
}

// --- 4. block_op + aliases (BlockOp, §7) ---
template <class Op, AnyMXVectorType Dst, AnyMXVectorType... Srcs>
LOFLOAT_HOST_DEVICE void block_op(Op op, typename Dst::shared_exp_type s_r, ProjSpec ps, Dst& out,
                                  const Srcs&... in) {
    static_assert(sizeof...(Srcs) >= 1, "block_op needs >= 1 operand");
    assert(((out.m == in.m) && ...) && "block_op operands must be conformable");
    using Tb = typename Dst::scalar_type;
    auto dec = [](const auto& v, auto k) {
        return static_cast<double>(v[k]) * static_cast<double>(v.get_exp(k));
    };
    const double S = static_cast<double>(s_r);
    decltype(out.m) i = 0;
    while (i < out.m) {
        auto blk = std::min(out.r, out.m - i);
        out.set_exp(i, s_r);
        for (decltype(blk) j = 0; j < blk; ++j) {
            double Z = op(dec(in, i + j)...);
            out[i + j] = lo_float::block_detail::block_project_elem<Tb>(S, Z, ps);
        }
        i += blk;
    }
}

template <AnyMXVectorType A, AnyMXVectorType B, AnyMXVectorType Out>
LOFLOAT_HOST_DEVICE void block_add(const A& x, const B& y, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    block_op([](double a, double b) { return a + b; }, s_r, ps, out, x, y);
}
template <AnyMXVectorType A, AnyMXVectorType B, AnyMXVectorType Out>
LOFLOAT_HOST_DEVICE void block_sub(const A& x, const B& y, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    block_op([](double a, double b) { return a - b; }, s_r, ps, out, x, y);
}
template <AnyMXVectorType A, AnyMXVectorType B, AnyMXVectorType Out>
LOFLOAT_HOST_DEVICE void block_mul(const A& x, const B& y, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    block_op([](double a, double b) { return a * b; }, s_r, ps, out, x, y);
}
template <AnyMXVectorType A, AnyMXVectorType B, AnyMXVectorType Out>
LOFLOAT_HOST_DEVICE void block_div(const A& x, const B& y, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    // P3109 Divide(*,0) -> NaN (IEEE double would give ±∞); every other case matches IEEE double.
    block_op([](double a, double b) { return (b == 0.0) ? std::numeric_limits<double>::quiet_NaN() : a / b; },
             s_r, ps, out, x, y);
}
template <AnyMXVectorType A, AnyMXVectorType Out>
LOFLOAT_HOST_DEVICE void block_abs(const A& x, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    block_op([](double a) { return std::fabs(a); }, s_r, ps, out, x);
}
template <AnyMXVectorType A, AnyMXVectorType Out>
LOFLOAT_HOST_DEVICE void block_negate(const A& x, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    block_op([](double a) { return -a; }, s_r, ps, out, x);
}

// --- 5. reductions (Group E; host-only v1; wide Out=double default) ---
template <class Out = double, AnyMXVectorType Src>
LOFLOAT_HOST Out block_reduce_add(const Src& a, ProjSpec ps = ProjSpec{}) {
    double acc = 0.0;
    for (decltype(a.m) i = 0; i < a.m; ++i)
        acc += static_cast<double>(a[i]) * static_cast<double>(a.get_exp(i));
    if constexpr (std::is_same_v<Out, double>) { (void)ps; return acc; }
    else return lo_float::Round<Out>(acc, ps);
}
template <class Out = double, AnyMXVectorType Src>
LOFLOAT_HOST Out block_reduce_mul(const Src& a, ProjSpec ps = ProjSpec{}) {
    double acc = 1.0;
    for (decltype(a.m) i = 0; i < a.m; ++i)
        acc *= static_cast<double>(a[i]) * static_cast<double>(a.get_exp(i));
    if constexpr (std::is_same_v<Out, double>) { (void)ps; return acc; }
    else return lo_float::Round<Out>(acc, ps);
}
template <class Out = double, AnyMXVectorType X, AnyMXVectorType Y>
LOFLOAT_HOST Out block_dot(const X& x, const Y& y, ProjSpec ps = ProjSpec{}) {
    assert(x.m == y.m);
    double acc = 0.0;
    for (decltype(x.m) i = 0; i < x.m; ++i) {
        double xd = static_cast<double>(x[i]) * static_cast<double>(x.get_exp(i));
        double yd = static_cast<double>(y[i]) * static_cast<double>(y.get_exp(i));
        acc += xd * yd;
    }
    if constexpr (std::is_same_v<Out, double>) { (void)ps; return acc; }
    else return lo_float::Round<Out>(acc, ps);
}

} // namespace lo_float


// ============================ MATRIX ops (namespace lo_float) ============================
// Same math as the vector ops; the length-r inner loop is replaced by the logical (ld-independent)
// 2D tile-grid walk, the scale is stored once at the tile anchor (row0,col0), and the decoded term
// is scaled_val<double>(row,col). See loop/block_ops_3109.md §9.
namespace lo_float {

// --- 0. block_decode: fill Z in ROW-MAJOR LOGICAL order (Z[row*cols + col]) ---
template <AnyMXMatrixType Src>
LOFLOAT_HOST_DEVICE void block_decode(const Src& A, double* Z) {
    for (decltype(A.rows()) r = 0; r < A.rows(); ++r)
        for (decltype(A.cols()) c = 0; c < A.cols(); ++c)
            Z[static_cast<size_t>(r) * A.cols() + c] = A.template scaled_val<double>(r, c);
}

// --- 1. convert_from_block (§3/§9.1) ---
template <AnyMXMatrixType Src, MatrixType Dst>
LOFLOAT_HOST_DEVICE void convert_from_block(const Src& A, Dst& B, ProjSpec ps = ProjSpec{}) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using Tb = typename Dst::scalar_type;
    for (decltype(A.rows()) r = 0; r < A.rows(); ++r)
        for (decltype(A.cols()) c = 0; c < A.cols(); ++c)
            B(r, c) = lo_float::Round<Tb>(A.template scaled_val<double>(r, c), ps);
}

// --- 2. convert_to_block: dense -> block, CALLER scale (§4/§9.2) ---
template <MatrixType Src, AnyMXMatrixType Dst>
LOFLOAT_HOST_DEVICE void convert_to_block(const Src& A, typename Dst::shared_exp_type s, Dst& B,
                                          ProjSpec ps = ProjSpec{}) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using Tb = typename Dst::scalar_type;
    using idx = decltype(B.num_block_rows());
    const double S = static_cast<double>(s);
    for (idx br = 0; br < B.num_block_rows(); ++br) {
        idx row0 = br * B.tile_rows, row1 = std::min<idx>(row0 + B.tile_rows, B.rows());
        for (idx bc = 0; bc < B.num_block_cols(); ++bc) {
            idx col0 = bc * B.tile_cols, col1 = std::min<idx>(col0 + B.tile_cols, B.cols());
            B.set_exp(row0, col0, s);
            for (idx r = row0; r < row1; ++r)
                for (idx c = col0; c < col1; ++c)
                    B(r, c) = lo_float::block_detail::block_project_elem<Tb>(S, static_cast<double>(A(r, c)), ps);
        }
    }
}

// --- 3. convert_to_block_max_abs_finite: dense -> block, COMPUTED scale (§5/§9.3) ---
template <MatrixType Src, AnyMXMatrixType Dst>
LOFLOAT_HOST_DEVICE void convert_to_block_max_abs_finite(const Src& A, Dst& B,
                                                         ProjSpec ps_scale = ProjSpec{},
                                                         ProjSpec ps = ProjSpec{}) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using Tb = typename Dst::scalar_type;
    using Ts = typename Dst::shared_exp_type;
    using idx = decltype(B.num_block_rows());
    for (idx br = 0; br < B.num_block_rows(); ++br) {
        idx row0 = br * B.tile_rows, row1 = std::min<idx>(row0 + B.tile_rows, B.rows());
        for (idx bc = 0; bc < B.num_block_cols(); ++bc) {
            idx col0 = bc * B.tile_cols, col1 = std::min<idx>(col0 + B.tile_cols, B.cols());
            double Sf = 0.0;
            bool has_inf = false, has_nan = false;
            for (idx r = row0; r < row1; ++r)
                for (idx c = col0; c < col1; ++c) {
                    double x = static_cast<double>(A(r, c));
                    if (std::isnan(x))      has_nan = true;
                    else if (std::isinf(x)) has_inf = true;
                    else { double mag = std::fabs(x); if (mag > Sf) Sf = mag; }
                }
            Ts scale;
            if (Sf > 0.0)     scale = lo_float::Round<Ts>(Sf, ps_scale);
            else if (has_inf) scale = lo_float::Round<Ts>(std::numeric_limits<double>::infinity(), ps_scale);
            else if (has_nan) scale = lo_float::Round<Ts>(std::numeric_limits<double>::quiet_NaN(), ps_scale);
            else              scale = Ts{};
            B.set_exp(row0, col0, scale);
            const double S = static_cast<double>(scale);
            for (idx r = row0; r < row1; ++r)
                for (idx c = col0; c < col1; ++c)
                    B(r, c) = lo_float::block_detail::block_project_elem<Tb>(S, static_cast<double>(A(r, c)), ps);
        }
    }
}

// --- 4. block_op + aliases (§7/§9.4) ---
template <class Op, AnyMXMatrixType Dst, AnyMXMatrixType... Srcs>
LOFLOAT_HOST_DEVICE void block_op(Op op, typename Dst::shared_exp_type s_r, ProjSpec ps, Dst& out,
                                  const Srcs&... in) {
    static_assert(sizeof...(Srcs) >= 1, "block_op needs >= 1 operand");
    assert(((out.rows() == in.rows() && out.cols() == in.cols()) && ...) &&
           "block_op operands must be conformable");
    using Tb = typename Dst::scalar_type;
    using idx = decltype(out.num_block_rows());
    auto dec = [](const auto& M, idx r, idx c) { return M.template scaled_val<double>(r, c); };
    const double S = static_cast<double>(s_r);
    for (idx br = 0; br < out.num_block_rows(); ++br) {
        idx row0 = br * out.tile_rows, row1 = std::min<idx>(row0 + out.tile_rows, out.rows());
        for (idx bc = 0; bc < out.num_block_cols(); ++bc) {
            idx col0 = bc * out.tile_cols, col1 = std::min<idx>(col0 + out.tile_cols, out.cols());
            out.set_exp(row0, col0, s_r);
            for (idx r = row0; r < row1; ++r)
                for (idx c = col0; c < col1; ++c) {
                    double Z = op(dec(in, r, c)...);
                    out(r, c) = lo_float::block_detail::block_project_elem<Tb>(S, Z, ps);
                }
        }
    }
}

template <AnyMXMatrixType A, AnyMXMatrixType B, AnyMXMatrixType Out>
LOFLOAT_HOST_DEVICE void block_add(const A& x, const B& y, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    block_op([](double a, double b) { return a + b; }, s_r, ps, out, x, y);
}
template <AnyMXMatrixType A, AnyMXMatrixType B, AnyMXMatrixType Out>
LOFLOAT_HOST_DEVICE void block_sub(const A& x, const B& y, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    block_op([](double a, double b) { return a - b; }, s_r, ps, out, x, y);
}
template <AnyMXMatrixType A, AnyMXMatrixType B, AnyMXMatrixType Out>
LOFLOAT_HOST_DEVICE void block_mul(const A& x, const B& y, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    block_op([](double a, double b) { return a * b; }, s_r, ps, out, x, y);
}
template <AnyMXMatrixType A, AnyMXMatrixType B, AnyMXMatrixType Out>
LOFLOAT_HOST_DEVICE void block_div(const A& x, const B& y, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    block_op([](double a, double b) { return (b == 0.0) ? std::numeric_limits<double>::quiet_NaN() : a / b; },
             s_r, ps, out, x, y);
}
template <AnyMXMatrixType A, AnyMXMatrixType Out>
LOFLOAT_HOST_DEVICE void block_abs(const A& x, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    block_op([](double a) { return std::fabs(a); }, s_r, ps, out, x);
}
template <AnyMXMatrixType A, AnyMXMatrixType Out>
LOFLOAT_HOST_DEVICE void block_negate(const A& x, typename Out::shared_exp_type s_r, Out& out, ProjSpec ps = ProjSpec{}) {
    block_op([](double a) { return -a; }, s_r, ps, out, x);
}

// --- 5. reductions over ALL decoded elements (host-only v1; wide Out=double default) ---
template <class Out = double, AnyMXMatrixType Src>
LOFLOAT_HOST Out block_reduce_add(const Src& A, ProjSpec ps = ProjSpec{}) {
    double acc = 0.0;
    for (decltype(A.rows()) r = 0; r < A.rows(); ++r)
        for (decltype(A.cols()) c = 0; c < A.cols(); ++c)
            acc += A.template scaled_val<double>(r, c);
    if constexpr (std::is_same_v<Out, double>) { (void)ps; return acc; }
    else return lo_float::Round<Out>(acc, ps);
}
template <class Out = double, AnyMXMatrixType Src>
LOFLOAT_HOST Out block_reduce_mul(const Src& A, ProjSpec ps = ProjSpec{}) {
    double acc = 1.0;
    for (decltype(A.rows()) r = 0; r < A.rows(); ++r)
        for (decltype(A.cols()) c = 0; c < A.cols(); ++c)
            acc *= A.template scaled_val<double>(r, c);
    if constexpr (std::is_same_v<Out, double>) { (void)ps; return acc; }
    else return lo_float::Round<Out>(acc, ps);
}

} // namespace lo_float
