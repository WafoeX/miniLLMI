#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace runtime {
enum class DType { FP32, INT32, INT8 };

inline std::size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::FP32: return sizeof(float);
    case DType::INT32: return sizeof(std::int32_t);
    case DType::INT8: return sizeof(std::int8_t);
    }
    throw std::invalid_argument("unknown dtype");
}
inline std::size_t dtype_alignment(DType dtype) {
    switch (dtype) {
    case DType::FP32: return alignof(float);
    case DType::INT32: return alignof(std::int32_t);
    case DType::INT8: return alignof(std::int8_t);
    }
    throw std::invalid_argument("unknown dtype");
}
static_assert(sizeof(float) == 4 && sizeof(std::int32_t) == 4 && sizeof(std::int8_t) == 1,
              "runtime requires 32-bit FP32/INT32 and 8-bit INT8");

template<class T> DType dtype_of() {
    using Value = std::remove_cv_t<T>;
    static_assert(std::is_same_v<Value, float> || std::is_same_v<Value, std::int32_t> ||
                  std::is_same_v<Value, std::int8_t>,
                  "typed tensor access supports only float, int32_t, and int8_t");
    if constexpr (std::is_same_v<Value, float>) return DType::FP32;
    else if constexpr (std::is_same_v<Value, std::int32_t>) return DType::INT32;
    else return DType::INT8;
}
} // namespace runtime
