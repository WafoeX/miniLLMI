#pragma once

#include <stdexcept>

namespace runtime {
// CUDA metadata is always available; it does not include or require CUDA headers.
enum class DeviceType { CPU, CUDA };
class Device {
public:
    explicit Device(DeviceType type = DeviceType::CPU, int index = 0)
        : type_(type), index_(index) {
        if (type != DeviceType::CPU && type != DeviceType::CUDA)
            throw std::invalid_argument("unknown device type");
        if (index < 0 || (type == DeviceType::CPU && index != 0))
            throw std::invalid_argument("invalid device index");
    }
    DeviceType type() const noexcept { return type_; }
    int index() const noexcept { return index_; }
    bool operator==(Device other) const noexcept {
        return type_ == other.type_ && index_ == other.index_;
    }
    bool operator!=(Device other) const noexcept { return !(*this == other); }
    bool operator<(Device other) const noexcept {
        return type_ < other.type_ || (type_ == other.type_ && index_ < other.index_);
    }
private:
    DeviceType type_;
    int index_;
};
} // namespace runtime
