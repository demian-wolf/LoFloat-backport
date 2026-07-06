#pragma once
#include <cstdint>



namespace lo_float {

template<typename T>
struct range : std::pair<T, T> {
    using std::pair<T, T>::pair; 
};
template<typename T>
range(T, T) -> range<T>;


enum Layout : uint8_t {
    ColMajor = 0,
    RowMajor = 1
};

// Controls which direction a shared MX scale's block of private elements runs in a 2D
// MX_Matrix: byColumn groups r contiguous rows within one column; byRow groups r contiguous
// columns within one row; byTile groups an explicit tile_rows x tile_cols 2D block (see
// MX_Matrix's tile_rows/tile_cols). Block membership is always computed from the LOGICAL
// (row, col) position in the m x n grid, independent of the physical leading dimension `ld`.
enum MX_Layout : uint8_t {
    byColumn = 0,
    byRow = 1,
    byTile = 2
};

enum Uplo : uint8_t {
    Upper = 0,
    Lower = 1, 
    General = 2
};

struct MX_tuple {
    MX_Layout layout;
    int m;
    int n;

    MX_tuple(MX_Layout layout, int m, int n = 1) : layout(layout), m(m), n(n) {}
};

enum Arch_extensions : uint8_t {
    NONE = 0,
    AVX256 = 1,
    AVX512 = 2,
    NEON = 3
};
}
