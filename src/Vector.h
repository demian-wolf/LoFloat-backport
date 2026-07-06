/// @author Sudhanva Kulkarni
/// Simple implementation of Vector and MXVector objects

#pragma once
#include <cassert>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include "lo_float.h"
#include "layouts.h"
#include <type_traits>


using namespace lo_float;

namespace lo_float {




template<Float T, Int idx = int>
class Vector {
    public :
    idx m;
    T* data;
    idx stride;
    using scalar_type = T;
    using vector_type_tag = void;

    Vector(T* data, idx m, idx stride = static_cast<idx>(1)) : data(data), m(m), stride(stride) {}

    inline  T& operator[]  (idx i) const {
        return data[i*stride];
    }

    inline T& operator() (idx i) const {
        return data[i*stride];
    }

    inline const idx len() {
        return m;
    }


};

// MicroScaled vector, separate-array storage: `m` private elements (type T) sharing one scale
// (type T_scal) per block of `r` contiguous logical elements. Scales live in a SEPARATE array
// `shared_exps` of length `n == ceil(m/r)`, itself addressable with its own stride
// `exp_stride` (mirrors the data vector's own `stride`) so the exponent array can be laid out
// sparsely/padded independent of the data array.
template<Float T, Float T_scal, Int idx = int>
class MX_Vector {
    public:
    idx m;  //length of data vector
    idx n;  //length of shared_exps vector (number of blocks, == ceil(m/r))
    idx stride; //stride of data vector
    idx r; //number of contiguos elems that share common exp
    idx exp_stride; //stride of the shared_exps array
    T_scal* shared_exps;
    T* data;
    using scalar_type = T;
    using shared_exp_type = T_scal;
    using mx_vector_type_void = void;

    MX_Vector(T* data, T_scal* shared_exps, idx m, idx n, idx stride = static_cast<idx>(1),
              idx r = static_cast<idx>(1), idx exp_stride = static_cast<idx>(1))
        : data(data), shared_exps(shared_exps), m(m), n(n), stride(stride), r(r), exp_stride(exp_stride) {}

    // Raw (unscaled) private-element access.
    inline T& operator[] (idx i) const {
        return data[i*stride];
    }

    // Scaled value at logical index i, returned BY VALUE as R (default R = T). Templated so a
    // caller can request a wider type (float/double) since T(private_elem) * T_scal(scale)
    // generally doesn't fit back into T's own dynamic range. (Returning T& here used to bind a
    // reference to a temporary product -- a dangling reference; returning by value fixes it.)
    template<typename R = T>
    inline R operator() (idx i) const {
        return static_cast<R>(data[i*stride]) * static_cast<R>(shared_exps[(i/r)*exp_stride]);
    }

    // Returns a reference to the actual stored T_scal (no implicit T conversion, so no
    // dangling temporary -- the previous `const T&` return type silently converted through a
    // temporary whenever T != T_scal).
    inline const T_scal& get_exp(idx i) const {
        return shared_exps[(i/r)*exp_stride];
    }

    inline void set_exp(idx i, T_scal value) const {
        shared_exps[(i/r)*exp_stride] = value;
    }

    constexpr inline idx len() const {
        return m;
    }

    constexpr inline idx len_exp() const {
        return n;
    }



};

// Interleaved-storage MicroScaled vector: private elements and their shared scale live in ONE
// physical buffer, laid out as [scale_0][elem]x r [scale_1][elem]x r ... The buffer is
// addressed byte-wise (not as an array of a single element type) since T and T_scal may be
// different sizes/types. `m` = number of private elements, `r` = block size, number of blocks
// == ceil(m/r), and each block occupies `block_bytes(r) == sizeof(T_scal) + r*sizeof(T)` bytes.
template<Float T, Float T_scal, Int idx = int>
class MX_Vector_Interleaved {
    public:
    std::byte* buf;
    idx m;
    idx r;
    using scalar_type = T;
    using shared_exp_type = T_scal;
    using mx_vector_interleaved_type_void = void;

    MX_Vector_Interleaved(std::byte* buf, idx m, idx r = static_cast<idx>(1))
        : buf(buf), m(m), r(r) {}

    static constexpr size_t block_bytes(idx r) {
        return sizeof(T_scal) + static_cast<size_t>(r) * sizeof(T);
    }

    constexpr inline idx num_blocks() const {
        return (m + r - 1) / r;
    }

    // Total buffer size (bytes) required to hold `m` elements at block size `r` -- the
    // caller uses this to size/allocate `buf` before constructing.
    static constexpr size_t required_bytes(idx m, idx r) {
        idx nblk = (m + r - 1) / r;
        return static_cast<size_t>(nblk) * block_bytes(r);
    }

    inline std::byte* block_ptr(idx i) const {
        return buf + static_cast<size_t>(i / r) * block_bytes(r);
    }

    inline T& operator[] (idx i) const {
        std::byte* blk = block_ptr(i);
        return *reinterpret_cast<T*>(blk + sizeof(T_scal) + static_cast<size_t>(i % r) * sizeof(T));
    }

    template<typename R = T>
    inline R operator() (idx i) const {
        return static_cast<R>((*this)[i]) * static_cast<R>(get_exp(i));
    }

    inline const T_scal& get_exp(idx i) const {
        return *reinterpret_cast<const T_scal*>(block_ptr(i));
    }

    inline void set_exp(idx i, T_scal value) const {
        *reinterpret_cast<T_scal*>(block_ptr(i)) = value;
    }

    constexpr inline idx len() const {
        return m;
    }
};


// Concepts (needed by the generic utilities below; the matching *_t traits live at the
// bottom of this file with the other type traits).
template<typename T>
concept VectorType = requires { typename T::vector_type_tag; };

template<typename T>
concept MXVectorType = requires { typename T::mx_vector_type_void; };

template<typename T>
concept MXVectorInterleavedType = requires { typename T::mx_vector_interleaved_type_void; };

// Either MicroScaled storage flavor (separate-array or interleaved) -- both expose the same
// accessor surface (operator[], operator()<R>, get_exp, set_exp, r, m), so the quantize/copy
// utilities below are written once against it.
template<typename T>
concept AnyMXVectorType = MXVectorType<T> || MXVectorInterleavedType<T>;


template<Float T1, Int idx1, Float T2, Int idx2>
void copy(const Vector<T1, idx1>& a, Vector<T2, idx2>& b) {
    assert(a.m == b.m);
        for (int i = 0; i < a.m; ++i) {
            b[i] = static_cast<T2>(a[i]);
        }
        return;
    }



// MX -> MX copy (arrays.md par.6 semantics): undo the microscaling of the SOURCE (dequantize
// each element to a higher precision -- double), then re-microscale the intermediate per the
// DESTINATION's own block size, then write. This is a real requantization, so a.r and b.r may
// differ; on matched blocks with exactly representable data it reproduces the source bits.
template<AnyMXVectorType Src, AnyMXVectorType Dst>
void copy(const Src& a, Dst& b) {
    assert(a.m == b.m);
    using T2 = typename Dst::scalar_type;
    using T_scal2 = typename Dst::shared_exp_type;
    auto dequant = [&](auto i) {
        return static_cast<double>(a[i]) * static_cast<double>(a.get_exp(i));
    };
    decltype(b.m) i = 0;
    while (i < b.m) {
        auto blk_len = std::min(b.r, b.m - i);
        double maximum = 0.0;
        for (decltype(blk_len) j = 0; j < blk_len; ++j) {
            double mag = std::fabs(dequant(i + j));
            maximum = mag > maximum ? mag : maximum;
        }
        T_scal2 scale = static_cast<T_scal2>(maximum);
        b.set_exp(i, scale);
        for (decltype(blk_len) j = 0; j < blk_len; ++j) {
            b[i + j] = (scale == T_scal2{})
                           ? T2{}
                           : static_cast<T2>(dequant(i + j) / static_cast<double>(scale));
        }
        i += blk_len;
    }
    return;
}



// Quantizes a plain Vector into a MicroScaled vector (either storage flavor): computes the
// true per-block max magnitude (over abs values) as the block's shared scale, and stores each
// private element as the ratio to that scale. Handles a ragged final block (a.m % b.r != 0)
// and an all-zero block (scale 0 -> elements written as 0, avoiding a divide-by-zero). This is
// the REAL (storage) quantization counterpart to `fake_mxcopy` below (which simulates the
// rounding but keeps the output as a full-precision Vector, not a true MX vector).
template<VectorType Src, AnyMXVectorType Dst>
void MX_real_quantize(const Src& a, Dst& b) {
    assert(a.m == b.m);
    using T2 = typename Dst::scalar_type;
    using T_scal = typename Dst::shared_exp_type;
    decltype(b.m) i = 0;
    while (i < b.m) {
        auto blk_len = std::min(b.r, b.m - i);
        double maximum = 0.0;
        for (decltype(blk_len) j = 0; j < blk_len; ++j) {
            double mag = std::fabs(static_cast<double>(a[i + j]));
            maximum = mag > maximum ? mag : maximum;
        }
        T_scal scale = static_cast<T_scal>(maximum);
        b.set_exp(i, scale);
        for (decltype(blk_len) j = 0; j < blk_len; ++j) {
            b[i + j] = (scale == T_scal{})
                           ? T2{}
                           : static_cast<T2>(static_cast<double>(a[i + j]) / static_cast<double>(scale));
        }
        i += blk_len;
    }
    return;
}

// Regular -> MX copy (arrays.md par.6 semantics): copying a plain Vector into an MX vector IS
// quantization -- find the largest element in each block, scale, then round all private
// elements. Delegates to MX_real_quantize.
template<VectorType Src, AnyMXVectorType Dst>
void copy(const Src& a, Dst& b) {
    MX_real_quantize(a, b);
}

// MX -> regular copy: dequantize (materialize elem * scale in double, then round into T2).
template<AnyMXVectorType Src, VectorType Dst>
void copy(const Src& a, Dst& b) {
    assert(a.m == b.m);
    using T2 = typename Dst::scalar_type;
    for (decltype(a.m) i = 0; i < a.m; ++i) {
        b[i] = static_cast<T2>(static_cast<double>(a[i]) * static_cast<double>(a.get_exp(i)));
    }
    return;
}

template<Float T1, Int idx>
Vector<T1, idx> slice(const Vector<T1, idx>& a, range<idx> range) {
    assert(range.first >= 0 && range.second <= a.m && range.first <= range.second);

    return Vector<T1, idx>(a.data + range.first * a.stride, range.second - range.first, a.stride);
}

// Slices an MX_Vector to the logical element range [range.first, range.second). Requires the
// slice to start on a block boundary (range.first % a.r == 0) since a partial leading block
// would otherwise need to share a scale with elements outside the slice.
template<Float T1, Int idx, Float T_scal>
MX_Vector<T1, T_scal, idx> slice(const MX_Vector<T1, T_scal, idx>& a, range<idx> range) {
    assert(range.first >= 0 && range.second <= a.m && range.first <= range.second);
    assert(range.first % a.r == 0 && "MX_Vector slice must start on a block boundary");

    idx new_m = range.second - range.first;
    idx new_n = (new_m + a.r - 1) / a.r;
    idx block_offset = range.first / a.r;

    return MX_Vector<T1, T_scal, idx>(a.data + range.first * a.stride,
                                       a.shared_exps + block_offset * a.exp_stride,
                                       new_m, new_n, a.stride, a.r, a.exp_stride);
}

//converts normal vector<t1, idx1> to vector<T2, idx2> while simulating the effects of microscaling.
//`r` is the block size (number of contiguous elements sharing one simulated scale); the plain
//Vector<T2,idx2> destination has no `r` member of its own, so it is passed explicitly.
template<Float T1, Int idx1, Float T2, Int idx2, Float T_scal, Float T_buf>
void fake_mxcopy(const Vector<T1, idx1>& a, Vector<T2, idx2>& b, idx1 r)
{
    assert(a.m == b.m);
    idx1 i = 0;
    while (i < a.m) {
        idx1 blk_len = std::min<idx1>(r, a.m - i);
        T1 maximum = T1{};
        for (idx1 j = 0; j < blk_len; ++j) {
            maximum = maximum > abs(a[i + j]) ? maximum : abs(a[i + j]);
        }
        T_scal exp = static_cast<T_scal>(maximum);
        for (idx1 j = 0; j < blk_len; ++j) {
            if (exp == T_scal{}) {
                b[i + j] = T2{};
            } else {
                b[i + j] = static_cast<T2>(static_cast<T_buf>(static_cast<double>(a[i + j]) / static_cast<double>(exp)))
                           * static_cast<T2>(exp);
            }
        }
        i += blk_len;
    }
    return;
}


// Trait: true if T has vector_type_tag
template<typename T>
struct is_VectorType {
    static constexpr bool value = requires { typename T::vector_type_tag; };
};

template<typename T>
inline constexpr bool is_VectorType_t = is_VectorType<T>::value;

// Trait: true if T has mx_vector_type_void
template<typename T>
struct is_MXVectorType {
    static constexpr bool value = requires { typename T::mx_vector_type_void; };
};

template<typename T>
inline constexpr bool is_MXVectorType_t = is_MXVectorType<T>::value;

// Trait: true if T has mx_vector_interleaved_type_void
template<typename T>
struct is_MXVectorInterleavedType {
    static constexpr bool value = requires { typename T::mx_vector_interleaved_type_void; };
};

template<typename T>
inline constexpr bool is_MXVectorInterleavedType_t = is_MXVectorInterleavedType<T>::value;


} // namespace lo_float
