#include "runtime/thread_pool.hpp"

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace runtime {
struct ThreadPool::State {
    std::mutex queue_mutex;
    std::condition_variable work_ready;
    std::deque<std::function<void()>> queue;
    bool stopping = false;
    std::vector<std::thread> workers;
    std::mutex submission_mutex;
};

ThreadPool::ThreadPool(std::size_t worker_count) : state_(std::make_unique<State>()) {
    if (worker_count == 0) throw std::invalid_argument("thread pool requires at least one worker");
    try {
        state_->workers.reserve(worker_count);
        for (std::size_t i = 0; i < worker_count; ++i) {
            state_->workers.emplace_back([state = state_.get()] {
                for (;;) {
                    std::function<void()> work;
                    {
                        std::unique_lock<std::mutex> lock(state->queue_mutex);
                        state->work_ready.wait(lock, [state] { return state->stopping || !state->queue.empty(); });
                        if (state->stopping && state->queue.empty()) return;
                        work = std::move(state->queue.front());
                        state->queue.pop_front();
                    }
                    work();
                }
            });
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(state_->queue_mutex);
            state_->stopping = true;
        }
        state_->work_ready.notify_all();
        for (auto& worker : state_->workers) if (worker.joinable()) worker.join();
        throw;
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(state_->queue_mutex);
        state_->stopping = true;
    }
    state_->work_ready.notify_all();
    for (auto& worker : state_->workers) if (worker.joinable()) worker.join();
}

Status ThreadPool::parallel_for(std::size_t tasks, const Task& task) {
    if (!task) return Status::failure(StatusCode::InvalidArgument, "thread-pool task is empty");
    if (tasks == 0) return Status::success();
    std::unique_lock<std::mutex> submission_lock(state_->submission_mutex);
    struct Batch {
        explicit Batch(std::size_t count) : remaining(count) {}
        std::mutex mutex;
        std::condition_variable complete;
        std::size_t remaining;
        Status status = Status::success();
    };
    const auto batch = std::make_shared<Batch>(tasks);
    {
        std::lock_guard<std::mutex> queue_lock(state_->queue_mutex);
        if (state_->stopping) return Status::failure(StatusCode::ResourceExhausted, "thread pool is shutting down");
        for (std::size_t index = 0; index < tasks; ++index) {
            state_->queue.emplace_back([batch, task, index] {
                Status status;
                try {
                    status = task(index);
                } catch (const std::exception& error) {
                    status = Status::failure(StatusCode::InvalidArgument, std::string("thread-pool task threw: ") + error.what());
                } catch (...) {
                    status = Status::failure(StatusCode::InvalidArgument, "thread-pool task threw a non-standard exception");
                }
                std::lock_guard<std::mutex> lock(batch->mutex);
                if (!status.ok() && batch->status.ok()) batch->status = std::move(status);
                if (--batch->remaining == 0) batch->complete.notify_one();
            });
        }
    }
    state_->work_ready.notify_all();
    std::unique_lock<std::mutex> batch_lock(batch->mutex);
    batch->complete.wait(batch_lock, [&batch] { return batch->remaining == 0; });
    return batch->status;
}

std::size_t ThreadPool::worker_count() const noexcept { return state_->workers.size(); }
std::size_t ThreadPool::creation_count() const noexcept { return state_->workers.size(); }
} // namespace runtime
