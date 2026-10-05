#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace runtime {
enum class DType { FP32, INT32 };

inline std::size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::FP32: return sizeof(float);
    case DType::INT32: return sizeof(std::int32_t);
    }
    throw std::invalid_argument("unknown dtype");
}
inline std::size_t dtype_alignment(DType dtype) {
    switch (dtype) {
    case DType::FP32: return alignof(float);
    case DType::INT32: return alignof(std::int32_t);
    }
    throw std::invalid_argument("unknown dtype");
}
static_assert(sizeof(float) == 4 && sizeof(std::int32_t) == 4,
              "runtime requires 32-bit FP32 and INT32");

template<class T> DType dtype_of() {
    using Value = std::remove_cv_t<T>;
    static_assert(std::is_same_v<Value, float> || std::is_same_v<Value, std::int32_t>,
                  "typed tensor access supports only float and int32_t");
    if constexpr (std::is_same_v<Value, float>) return DType::FP32;
    else return DType::INT32;
}
} // namespace runtime
