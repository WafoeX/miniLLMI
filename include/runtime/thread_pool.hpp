#pragma once

#include "runtime/operator.hpp"
#include <cstddef>
#include <functional>
#include <memory>

namespace runtime {

// A deliberately simple FIFO pool. parallel_for serializes submitted batches,
// while workers execute the batch's disjoint callbacks and report the first
// failure after all queued tasks have completed.
class ThreadPool {
public:
    using Task = std::function<Status(std::size_t)>;

    explicit ThreadPool(std::size_t worker_count);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    Status parallel_for(std::size_t tasks, const Task& task);
    std::size_t worker_count() const noexcept;
    std::size_t creation_count() const noexcept;

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace runtime
