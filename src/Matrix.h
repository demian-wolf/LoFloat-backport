/// @author Sudhanva Kulkarni
/// Simplke implementation of Varying Layout dense matrix
#pragma once
#include <algorithm>
#include <concepts>
#include <type_traits>
#include <cassert>
#include <cstddef>
#include <cmath>
#include "lo_float.h"
#include "layouts.h"



using namespace std;
using namespace lo_float;

namespace lo_float {



template<Float T, Int idx = int, Layout L = ColMajor>
class Matrix {
    public:

    idx m;
    idx n;
    idx ld;
    T* data;
    static constexpr Layout layout = L;
    using scalar_type = T;
    using idx_type = idx;
    using matrix_type_tag = void;


    Matrix(T* data, idx m, idx n, idx ld) : data(data), m(m), n(n), ld(ld) {
        assert(ld >= (L == ColMajor ? m : n) && "leading dimension must be >= the strided dimension");
    }

    inline T& operator()(idx row, idx col) const {
        if constexpr (L == ColMajor)
            return data[col * ld + row];
        else
            return data[row * ld + col];
    }

    // Added for consistency and use in transpose function
    constexpr inline idx get_idx(idx row, idx col) const {
        if constexpr (L == ColMajor)
            return col * ld + row;
        else
            return row * ld + col;
    }

    bool isNaN() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isnan((*this)(row,col))) return true;
            }
        }
        return false;
    }

    bool isInf() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isinf((*this)(row,col))) return true;
            }
        }
        return false;
    }

    constexpr inline idx rows() const {
        return this->m;
    }

    constexpr inline idx cols() const {
        return this->n;
    }

};

// MicroScaled matrix, separate-array storage. Private elements (type T) are grouped into 2D
// tiles of shape (tile_rows x tile_cols) that each share one scale (type T_scal); tile
// membership is computed from the LOGICAL (row, col) position in the m x n grid, independent
// of the physical leading dimension `ld` (so `ld > m`/`n` padding never perturbs which elements
// share a scale -- see loop/arrays.md bug #10/#11 for why the old `(col*ld+row)/r` formula was
// wrong). `mx_layout` records how a tile's shape was derived:
//   - byColumn: tile_rows = r, tile_cols = 1  (a block runs down r contiguous rows of one column)
//   - byRow:    tile_rows = 1, tile_cols = r  (a block runs across r contiguous cols of one row)
//   - byTile:   tile_rows/tile_cols given explicitly (general 2D tile)
// `shared_exps` is itself conceptually a (num_block_rows x num_block_cols) matrix in the SAME
// Layout L as the private elements, with its own leading dimension `exp_ld` (defaults to
// tightly packed, but overridable so a caller can pre-pad the exponent array).
//
// Transpose: see the MX transpose free function below and loop/mx_transpose.md -- it
// dequantizes and re-microscales per the destination's own tiling; pass the transpose-image
// tiling (swapped tile dims, byColumn<->byRow) for a lossless transpose.
template<Float T, Float T_scal , Int idx = int, Layout L = ColMajor>
class MX_Matrix {
    public:

    idx m;
    idx n;
    idx ld;
    idx tile_rows;      // number of logical rows in one MX tile
    idx tile_cols;      // number of logical cols in one MX tile
    idx exp_ld;         // leading dimension of the shared_exps "tile matrix" (same Layout as L)
    MX_Layout mx_layout;
    T* data;
    T_scal* shared_exps;
    static constexpr Layout layout = L;
    using scalar_type = T;
    using shared_exp_type = T_scal;
    using MX_matrix_type_tag = void;

    static constexpr MX_Layout default_mx_layout = (L == ColMajor) ? MX_Layout::byColumn : MX_Layout::byRow;

    constexpr inline idx num_block_rows() const { return (m + tile_rows - 1) / tile_rows; }
    constexpr inline idx num_block_cols() const { return (n + tile_cols - 1) / tile_cols; }

    // Convenience ctor: a single block length `r` along byColumn/byRow (1D grouping, matching
    // the original API). Cannot express byTile (needs two independent dimensions) -- use the
    // (tile_rows, tile_cols) ctor below for that.
    MX_Matrix(T* data, T_scal* shared_exps, idx m, idx n, idx ld, idx r,
              MX_Layout mxl = default_mx_layout) {
        assert(ld >= (L == ColMajor ? m : n) && "leading dimension must be >= the strided dimension");
        assert(mxl != MX_Layout::byTile && "byTile needs an explicit (tile_rows, tile_cols); use that ctor");
        this->data = data; this->shared_exps = shared_exps;
        this->m = m; this->n = n; this->ld = ld; this->mx_layout = mxl;
        this->tile_rows = (mxl == MX_Layout::byColumn) ? r : static_cast<idx>(1);
        this->tile_cols = (mxl == MX_Layout::byRow)    ? r : static_cast<idx>(1);
        this->exp_ld = (L == ColMajor) ? num_block_rows() : num_block_cols();
    }

    // Explicit 2D-tile ctor (byTile by default), with an optional exp_ld override for a
    // pre-padded exponent array (0 = tightly packed, the default).
    MX_Matrix(T* data, T_scal* shared_exps, idx m, idx n, idx ld, idx tile_rows, idx tile_cols,
              MX_Layout mxl = MX_Layout::byTile, idx exp_ld_override = static_cast<idx>(0)) {
        assert(ld >= (L == ColMajor ? m : n) && "leading dimension must be >= the strided dimension");
        this->data = data; this->shared_exps = shared_exps;
        this->m = m; this->n = n; this->ld = ld; this->mx_layout = mxl;
        this->tile_rows = tile_rows; this->tile_cols = tile_cols;
        this->exp_ld = exp_ld_override != static_cast<idx>(0)
                           ? exp_ld_override
                           : ((L == ColMajor) ? num_block_rows() : num_block_cols());
    }

    constexpr inline idx get_idx(idx row, idx col) const {
        if constexpr (L == ColMajor) return col*ld + row;
        else return row*ld + col;
    }

    // Logical tile-block index for (row, col), independent of `ld`.
    constexpr inline idx block_idx(idx row, idx col) const {
        idx br = row / tile_rows;
        idx bc = col / tile_cols;
        if constexpr (L == ColMajor) return bc * exp_ld + br;
        else return br * exp_ld + bc;
    }

    inline T& operator() (idx row, idx col) const {
        if constexpr (L == ColMajor)
            return data[col*ld + row];
        else
            return data[row*ld + col];
    }

    inline const T_scal& get_exp(idx row, idx col) const {
        return shared_exps[block_idx(row, col)];
    }

    template<typename V>
    constexpr inline void set_exp(idx row, idx col, V value) const {
        shared_exps[block_idx(row, col)] = static_cast<shared_exp_type>(value);
    }

    template<typename V>
    constexpr inline V scaled_val(idx row, idx col) const {
        return static_cast<V>(operator()(row, col)) * static_cast<V>(get_exp(row, col));
    }



    bool isNaN() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isnan((this)->template scaled_val<double>(row,col))) return true;
            }
        }
        return false;
    }

    bool isInf() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isinf((this)->template scaled_val<double>(row,col))) return true;
            }
        }
        return false;
    }


    constexpr inline idx rows() const {
        return this->m;
    }

    constexpr inline idx cols() const {
        return this->n;
    }



};

// Interleaved-storage MicroScaled matrix: one physical buffer holds each tile as
// [scale][elem]x(tile_rows*tile_cols), tiles laid out in Layout L order (matching MX_Matrix's
// separate-array version). Within a tile, elements are stored in the same L order as the
// overall matrix. Byte-addressed since T and T_scal may differ in size.
template<Float T, Float T_scal, Int idx = int, Layout L = ColMajor>
class MX_Matrix_Interleaved {
    public:
    idx m, n;
    idx tile_rows, tile_cols;
    std::byte* buf;
    static constexpr Layout layout = L;
    using scalar_type = T;
    using shared_exp_type = T_scal;
    using MX_matrix_interleaved_type_tag = void;

    MX_Matrix_Interleaved(std::byte* buf, idx m, idx n, idx tile_rows, idx tile_cols)
        : m(m), n(n), tile_rows(tile_rows), tile_cols(tile_cols), buf(buf) {}

    constexpr inline idx num_block_rows() const { return (m + tile_rows - 1) / tile_rows; }
    constexpr inline idx num_block_cols() const { return (n + tile_cols - 1) / tile_cols; }

    static constexpr size_t tile_bytes(idx tile_rows, idx tile_cols) {
        return sizeof(T_scal) + static_cast<size_t>(tile_rows) * static_cast<size_t>(tile_cols) * sizeof(T);
    }

    static constexpr size_t required_bytes(idx m, idx n, idx tile_rows, idx tile_cols) {
        idx nbr = (m + tile_rows - 1) / tile_rows;
        idx nbc = (n + tile_cols - 1) / tile_cols;
        return static_cast<size_t>(nbr) * static_cast<size_t>(nbc) * tile_bytes(tile_rows, tile_cols);
    }

    inline std::byte* tile_ptr(idx row, idx col) const {
        idx br = row / tile_rows, bc = col / tile_cols;
        idx tile_linear = (L == ColMajor) ? (bc * num_block_rows() + br) : (br * num_block_cols() + bc);
        return buf + static_cast<size_t>(tile_linear) * tile_bytes(tile_rows, tile_cols);
    }

    inline T& operator()(idx row, idx col) const {
        idx lr = row % tile_rows, lc = col % tile_cols;
        idx local = (L == ColMajor) ? (lc * tile_rows + lr) : (lr * tile_cols + lc);
        return *reinterpret_cast<T*>(tile_ptr(row, col) + sizeof(T_scal) + static_cast<size_t>(local) * sizeof(T));
    }

    inline const T_scal& get_exp(idx row, idx col) const {
        return *reinterpret_cast<const T_scal*>(tile_ptr(row, col));
    }

    inline void set_exp(idx row, idx col, T_scal value) const {
        *reinterpret_cast<T_scal*>(tile_ptr(row, col)) = value;
    }

    template<typename V>
    constexpr inline V scaled_val(idx row, idx col) const {
        return static_cast<V>(operator()(row, col)) * static_cast<V>(get_exp(row, col));
    }

    constexpr inline idx rows() const { return m; }
    constexpr inline idx cols() const { return n; }
};



// Concept: matches if T has either matrix_type_tag or MX_matrix_type_tag
template<typename T>
concept MatrixType = requires (T t) {
    typename T::matrix_type_tag;    // Check if matrix_type_tag exists
};

template<typename T>
concept MX_MatrixType = requires (T t) {
    typename T::MX_matrix_type_tag; // Check if MX_matrix_type_tag exists
};

template<typename T>
concept MX_MatrixInterleavedType = requires (T t) {
    typename T::MX_matrix_interleaved_type_tag;
};

// Either MicroScaled storage flavor (separate-array or interleaved) -- both expose the same
// accessor surface (operator(), get_exp, set_exp, scaled_val, tile_rows/tile_cols,
// num_block_rows/cols), so the quantize/requantize/transpose utilities below are written once
// against it.
template<typename T>
concept AnyMXMatrixType = MX_MatrixType<T> || MX_MatrixInterleavedType<T>;

// Trait: true if T has MX_matrix_type_tag
template<typename T>
struct is_MX_MatrixType {
    static constexpr bool value = requires { typename T::MX_matrix_type_tag; };
};

template<typename T>
inline constexpr bool is_MX_MatrixType_t = is_MX_MatrixType<T>::value;

// Concept: matches only if T has matrix_type_tag (Vanilla Matrix)
template<typename T>
concept is_Vanilla_MatrixType = requires {
    typename T::matrix_type_tag;
};

template<typename T>
inline constexpr bool is_Vanilla_MatrixType_t = is_Vanilla_MatrixType<T>;

//helper that returns if the format is MX by checking if the type has a shared_exps member
template<typename T>
struct is_MX_format {
    static constexpr bool value = false;
};

template<typename T, typename T_scal, typename idx, Layout L>
struct is_MX_format<MX_Matrix<T, T_scal, idx, L>> {
    static constexpr bool value = true;
};

template<typename T, typename idx, Layout L>
struct is_MX_format<Matrix<T, idx, L>> {
    static constexpr bool value = false;
};

template<typename T>
concept AnyMatrixType = MatrixType<T> || MX_MatrixType<T>;

template<AnyMatrixType Mat_type>
using scalar_type = typename Mat_type::scalar_type;


template<MX_MatrixType Mat_type>
using shared_exp_type = typename Mat_type::shared_exp_type;




template<MatrixType SrcMatrixType, MatrixType DstMatrixType>
void lacpy(const SrcMatrixType& A, DstMatrixType& B, Uplo uplo) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using B_type = scalar_type<DstMatrixType>;
    using A_type = scalar_type<SrcMatrixType>;

    if (uplo == Uplo::Upper) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = i; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
            }
        }
    } else if (uplo == Uplo::Lower) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j <= i; j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
            }
        }
    } else {
        // General case
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
            }
        }
    }

    return;

}

// MX -> MX lacpy: raw same-tiling copy (private elements + scales copied slot-for-slot).
// This assumes A and B share the same tile geometry; for the arrays.md par.6 requantizing
// copy (dequantize -> re-microscale per B's own tiling) use copy()/MX_requantize instead.
template<AnyMXMatrixType SrcMatrixType, AnyMXMatrixType DstMatrixType>
void lacpy(const SrcMatrixType& A, DstMatrixType& B, Uplo uplo) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using B_type = typename DstMatrixType::scalar_type;
    using B_exp_type = typename DstMatrixType::shared_exp_type;

    if (uplo == Uplo::Upper) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = i; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, static_cast<B_exp_type>(A.get_exp(i,j)));
            }
        }

    } else if (uplo == Uplo::Lower) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j <= i; j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, static_cast<B_exp_type>(A.get_exp(i,j)));
            }
        }
    } else {
        // General case
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, static_cast<B_exp_type>(A.get_exp(i,j)));
            }
        }
    }

    return;

}

// MX -> plain lacpy: dequantize (materialize elem * scale in double, round into B's type).
template<AnyMXMatrixType SrcMatrixType, MatrixType DstMatrixType>
void lacpy(const SrcMatrixType& A, DstMatrixType& B, Uplo uplo) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using B_type = typename DstMatrixType::scalar_type;

    if (uplo == Uplo::Upper) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = i; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A.template scaled_val<double>(i,j));
            }
        }
    } else if (uplo == Uplo::Lower) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j <= i; j++) {
                B(i,j) = static_cast<B_type>(A.template scaled_val<double>(i,j));
            }
        }
    } else {
        // General case
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A.template scaled_val<double>(i,j));
            }
        }
    }

    return;

}

template<MatrixType SrcMatrixType, AnyMXMatrixType DstMatrixType>
void lacpy(const SrcMatrixType& A, DstMatrixType& B, Uplo uplo) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using B_type = typename DstMatrixType::scalar_type;
    using A_type = typename SrcMatrixType::scalar_type;
    using B_exp_type = typename DstMatrixType::shared_exp_type;

    if (uplo == Uplo::Upper) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = i; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                // A is not an MX matrix and has no get_exp method -- this is a lossless re-tag
                // (neutral scale of 1), NOT real quantization. Use MX_real_quantize()/copy()
                // below for an actual per-tile-max-scaled quantization of dense data.
                B.set_exp(i,j, static_cast<B_exp_type>(1.0));
            }
        }

    } else if (uplo == Uplo::Lower) {
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j <= i; j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, static_cast<B_exp_type>(1.0));
            }
        }
    } else {
        // General case
        for(int i = 0; i < A.rows(); i++) {
            for(int j = 0; j < A.cols(); j++) {
                B(i,j) = static_cast<B_type>(A(i,j));
                B.set_exp(i,j, static_cast<B_exp_type>(1.0));
            }
        }
    }

    return;

}


// Quantizes a dense Matrix into a MicroScaled matrix (either storage flavor): for each
// (tile_rows x tile_cols) tile, computes the true max magnitude (abs value) over that tile as
// the shared scale, and stores each private element as the ratio to that scale (an all-zero
// tile gets scale 0 and elements 0, avoiding a divide-by-zero). This is the REAL (storage)
// quantization counterpart to the scale=1 `lacpy(MatrixType, AnyMXMatrixType)` re-tag above.
template<MatrixType SrcMatrixType, AnyMXMatrixType DstMatrixType>
void MX_real_quantize(const SrcMatrixType& A, DstMatrixType& B) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using T2 = typename DstMatrixType::scalar_type;
    using T_scal = typename DstMatrixType::shared_exp_type;
    using idx = decltype(B.num_block_rows());
    idx nbr = B.num_block_rows();
    idx nbc = B.num_block_cols();
    for (idx br = 0; br < nbr; br++) {
        idx row0 = br * B.tile_rows;
        idx row1 = std::min<idx>(row0 + B.tile_rows, A.rows());
        for (idx bc = 0; bc < nbc; bc++) {
            idx col0 = bc * B.tile_cols;
            idx col1 = std::min<idx>(col0 + B.tile_cols, A.cols());

            double maximum = 0.0;
            for (idx row = row0; row < row1; row++) {
                for (idx col = col0; col < col1; col++) {
                    double mag = std::fabs(static_cast<double>(A(row, col)));
                    maximum = mag > maximum ? mag : maximum;
                }
            }
            T_scal scale = static_cast<T_scal>(maximum);
            B.set_exp(row0, col0, scale);
            for (idx row = row0; row < row1; row++) {
                for (idx col = col0; col < col1; col++) {
                    B(row, col) = (scale == T_scal{})
                                      ? T2{}
                                      : static_cast<T2>(static_cast<double>(A(row, col)) / static_cast<double>(scale));
                }
            }
        }
    }
    return;
}

// MX -> MX requantization (arrays.md par.6 semantics): undo the microscaling of the SOURCE
// (dequantize each element to double), then re-microscale the intermediate per the
// DESTINATION's own tiling (which may differ from the source's in direction and tile dims),
// then write. On matched tilings with exactly representable data it reproduces the source bits.
template<AnyMXMatrixType SrcMatrixType, AnyMXMatrixType DstMatrixType>
void MX_requantize(const SrcMatrixType& A, DstMatrixType& B) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    using T2 = typename DstMatrixType::scalar_type;
    using T_scal = typename DstMatrixType::shared_exp_type;
    using idx = decltype(B.num_block_rows());
    idx nbr = B.num_block_rows();
    idx nbc = B.num_block_cols();
    for (idx br = 0; br < nbr; br++) {
        idx row0 = br * B.tile_rows;
        idx row1 = std::min<idx>(row0 + B.tile_rows, B.rows());
        for (idx bc = 0; bc < nbc; bc++) {
            idx col0 = bc * B.tile_cols;
            idx col1 = std::min<idx>(col0 + B.tile_cols, B.cols());

            double maximum = 0.0;
            for (idx row = row0; row < row1; row++) {
                for (idx col = col0; col < col1; col++) {
                    double mag = std::fabs(A.template scaled_val<double>(row, col));
                    maximum = mag > maximum ? mag : maximum;
                }
            }
            T_scal scale = static_cast<T_scal>(maximum);
            B.set_exp(row0, col0, scale);
            for (idx row = row0; row < row1; row++) {
                for (idx col = col0; col < col1; col++) {
                    B(row, col) = (scale == T_scal{})
                                      ? T2{}
                                      : static_cast<T2>(A.template scaled_val<double>(row, col) / static_cast<double>(scale));
                }
            }
        }
    }
    return;
}

// copy() carries the arrays.md par.6 semantics ("copying is identical to calling one of the
// Round functions"): plain -> plain rounds elementwise (lacpy); plain -> MX quantizes for real
// (per-tile max scale); MX -> MX dequantizes to double and re-microscales per the destination
// tiling; MX -> plain dequantizes. Triangular (Uplo::Upper/Lower) copies always take the raw
// lacpy path, since quantizing a partial tile that straddles the triangle boundary is
// ill-defined.
template<typename SrcMatrixType, typename DstMatrixType>
    requires ((MatrixType<SrcMatrixType> || AnyMXMatrixType<SrcMatrixType>) &&
              (MatrixType<DstMatrixType> || AnyMXMatrixType<DstMatrixType>))
inline void copy(const SrcMatrixType& A, DstMatrixType& B, Uplo uplo = Uplo::General) {
    if constexpr (MatrixType<SrcMatrixType> && AnyMXMatrixType<DstMatrixType>) {
        if (uplo == Uplo::General) { MX_real_quantize(A, B); return; }
    } else if constexpr (AnyMXMatrixType<SrcMatrixType> && AnyMXMatrixType<DstMatrixType>) {
        if (uplo == Uplo::General) { MX_requantize(A, B); return; }
    }
    lacpy(A, B, uplo);
}


template<MatrixType MatrixA, MatrixType MatrixA_t, int n_block_size = 1, int m_block_size = 1>
void transpose(MatrixA& A, MatrixA_t& At) {

    At.m = A.n;
    At.n = A.m;

    using At_type = typename MatrixA_t::scalar_type;

    //break into cases - if layouts are different, a simple data copy works. If same, block the transpose
    if constexpr (MatrixA::layout == MatrixA_t::layout) {
        // blocked transpose for same-layout
        int n_blocks = (At.rows() + n_block_size - 1) / n_block_size;
        int m_blocks = (At.cols() + m_block_size - 1) / m_block_size;

        for(int b_row = 0; b_row < n_blocks; b_row++) {
            for(int b_col = 0; b_col < m_blocks; b_col++) {
                // FIX: Corrected loop bounds to iterate over blocks in the destination matrix At.
                for(int row = b_row * n_block_size; row < std::min((b_row + 1) * n_block_size, (int)At.rows()); row++) {
                    for(int col = b_col * m_block_size; col < std::min((b_col + 1) * m_block_size, (int)At.cols()); col++) {
                            // At(row, col) = A(col, row)
                            At.data[At.get_idx(row, col)] = static_cast<At_type>(A.data[A.get_idx(col, row)]);
                    }
                }
            }
        }
    } else {
        // Different layouts (e.g. ColMajor -> RowMajor): must go through the indexed
        // accessors -- a raw linear copy is only valid when BOTH matrices are tightly packed
        // (ld == m/n), which the arbitrary-ld requirement violates.
        for(int i = 0; i < At.rows(); i++) {
            for(int j = 0; j < At.cols(); j++) {
                At(i, j) = static_cast<At_type>(A(j, i));
            }
        }
    }
}

// The MX_Layout a destination should use so that its tiles are the transpose-image of the
// source's (i.e. the same element sets share a scale on both sides -- the lossless choice;
// see loop/mx_transpose.md par.1): byColumn <-> byRow swap, byTile stays byTile (with
// tile_rows/tile_cols swapped by the caller).
constexpr inline MX_Layout transpose_image(MX_Layout mxl) {
    return mxl == MX_Layout::byColumn ? MX_Layout::byRow
         : mxl == MX_Layout::byRow    ? MX_Layout::byColumn
                                      : MX_Layout::byTile;
}

// Transpose of a MicroScaled matrix (either storage flavor), out-of-place, per
// loop/mx_transpose.md: `At` must be CONSTRUCTED by the caller with shape
// (A.cols() x A.rows()) and its own tiling (mx_layout / tile_rows / tile_cols). Dequantizes
// each destination tile in double, takes the tile max as the new shared scale, and writes
// rounded ratios -- i.e. it is MX_real_quantize applied to the transposed dense image, so it
// re-microscales per At's tiling. Pass the transpose-image tiling (At.tile_rows ==
// A.tile_cols, At.tile_cols == A.tile_rows, transpose_image(A.mx_layout)) with the same T /
// T_scal formats for a bit-preserving transpose (then transpose(transpose(A)) == A); any
// other tiling is legal but requantizes (<= 0.5 ULP(T2) * newscale per element for nearest
// rounding). O(m*n), each element touched once; destination-tile-driven so each scale is
// written exactly once.
template<AnyMXMatrixType SrcMatrixType, AnyMXMatrixType DstMatrixType>
void transpose(const SrcMatrixType& A, DstMatrixType& At) {
    assert(At.rows() == A.cols() && At.cols() == A.rows());
    using T2 = typename DstMatrixType::scalar_type;
    using T_scal = typename DstMatrixType::shared_exp_type;
    using idx = decltype(At.num_block_rows());
    idx nbr = At.num_block_rows();
    idx nbc = At.num_block_cols();
    for (idx br = 0; br < nbr; br++) {
        idx row0 = br * At.tile_rows;
        idx row1 = std::min<idx>(row0 + At.tile_rows, At.rows());
        for (idx bc = 0; bc < nbc; bc++) {
            idx col0 = bc * At.tile_cols;
            idx col1 = std::min<idx>(col0 + At.tile_cols, At.cols());

            // pass 1: tile max of the dequantized transposed source (At(r,c) <- A(c,r))
            double maximum = 0.0;
            for (idx row = row0; row < row1; row++) {
                for (idx col = col0; col < col1; col++) {
                    double mag = std::fabs(A.template scaled_val<double>(col, row));
                    maximum = mag > maximum ? mag : maximum;
                }
            }
            T_scal scale = static_cast<T_scal>(maximum);
            At.set_exp(row0, col0, scale);

            // pass 2: write rounded ratios (all-zero tile -> scale 0, elements 0)
            for (idx row = row0; row < row1; row++) {
                for (idx col = col0; col < col1; col++) {
                    At(row, col) = (scale == T_scal{})
                                       ? T2{}
                                       : static_cast<T2>(A.template scaled_val<double>(col, row) / static_cast<double>(scale));
                }
            }
        }
    }
    return;
}


//slice to get submatrix
template<typename T, Layout L, typename idx, typename idx2, typename idx3>
Matrix<T, idx, L> slice(const Matrix<T, idx, L>& a, range<idx2> rows, range<idx3> cols) {
    assert(cols.first >= 0 && cols.second <= a.n && rows.first >= 0 && rows.second <= a.m);
    assert(cols.first <= cols.second && rows.first <= rows.second);

    // FIX: Starting pointer calculation must account for memory layout.
    idx offset;
    if constexpr (L == ColMajor) {
        offset = cols.first * a.ld + rows.first;
    }
    else { // RowMajor
        offset = rows.first * a.ld + cols.first;
    }

    return Matrix<T, idx, L>(a.data + offset,
                          rows.second - rows.first,
                          cols.second - cols.first,
                          a.ld);
}

// Slices an MX_Matrix to the logical [rows.first,rows.second) x [cols.first,cols.second)
// sub-block. Requires the slice to start on a tile boundary in both dimensions (a partial
// leading tile would otherwise need to share a scale with elements outside the slice).
template<typename T, typename T_scal, Layout L, typename idx, typename idx2, typename idx3>
MX_Matrix<T, T_scal, idx, L> slice(const MX_Matrix<T, T_scal, idx, L>& a, range<idx2> rows, range<idx3> cols) {
    assert(cols.first >= 0 && cols.second <= a.n && rows.first >= 0 && rows.second <= a.m);
    assert(cols.first <= cols.second && rows.first <= rows.second);
    assert(rows.first % a.tile_rows == 0 && cols.first % a.tile_cols == 0 &&
           "MX_Matrix slice must start on a tile boundary");

    idx offset;
    if constexpr (L == ColMajor) {
        offset = cols.first * a.ld + rows.first;
    } else { // RowMajor
        offset = rows.first * a.ld + cols.first;
    }

    idx block_row0 = rows.first / a.tile_rows;
    idx block_col0 = cols.first / a.tile_cols;
    idx exp_offset = (L == ColMajor) ? (block_col0 * a.exp_ld + block_row0)
                                      : (block_row0 * a.exp_ld + block_col0);

    return MX_Matrix<T, T_scal, idx, L>(a.data + offset, a.shared_exps + exp_offset,
                                         rows.second - rows.first, cols.second - cols.first,
                                         a.ld, a.tile_rows, a.tile_cols, a.mx_layout, a.exp_ld);
}

template<Float T, Layout L, Int idx = int, Float T2 = T>
void pack(const Matrix<T, idx, L>& A, Matrix<T2, idx, L>& A_pack) {

    for(int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            A_pack(i,j) =(T2)A(i,j);
        }
    }
    return;
}

// (Signature note: an earlier version led with an undeducible `T_scal` template parameter,
// so this function could never actually be instantiated -- all parameters are now deduced
// from the arguments.)
template<Float T, Int idx, Layout L, Float T2, Float T2_scal>
void transpose_and_pack(const Matrix<T, idx, L>& A, MX_Matrix<T2, T2_scal, idx, L>& A_pack) {

    assert(A.rows() == A_pack.cols() && A.cols() == A_pack.rows());

    for(int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            A_pack(j,i) =(T2)A(i,j);
            A_pack.set_exp(j,i, static_cast<T2_scal>(1.0));
        }
    }
    return;
}


} //namespace lo_float
