#ifndef HEDGEHOG_LPBF_UTILITY_H
#define HEDGEHOG_LPBF_UTILITY_H

#include <condition_variable>
#include <Kokkos_Core.hpp>

template<typename MemorySpace>
concept KokkosMemorySpace = Kokkos::is_memory_space_v<MemorySpace>;

template<typename ExecutionSpace>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<ExecutionSpace>;

template<typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template<typename T>
concept KokkosMemoryTraits = Kokkos::is_memory_traits_v<T>;

template<KokkosExecutionSpace ExecutionSpace>
[[nodiscard]] auto createExecutionSpace(const int32_t deviceId = 0, const bool selfManaged = true, [[maybe_unused]] ExecutionSpace* = nullptr) {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Serial>) {
        return Kokkos::Serial();
    }

#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(deviceId));
        cudaStream_t stream;
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));
        return Kokkos::Cuda(stream, static_cast<Kokkos::Impl::ManageStream>(selfManaged));
    }
#endif

#ifdef KOKKOS_ENABLE_HIP
    if constexpr(std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
        KOKKOS_IMPL_HIP_SAFE_CALL(hipSetDevice(deviceId));
        hipStream_t stream;
        KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        return Kokkos::HIP(stream, static_cast<Kokkos::Impl::ManageStream>(selfManaged));
    }
#endif

#ifdef KOKKOS_ENABLE_SYCL
    if constexpr(std::is_same_v<ExecutionSpace, Kokkos::SYCL>) {
        throw std::runtime_error("fixme!");
        return Kokkos::DefaultExecutionSpace();
    }
#endif
}

[[nodiscard, deprecated]] inline auto createDefaultExecutionSpace(const bool selfManaged = true) {
    return createExecutionSpace<Kokkos::DefaultExecutionSpace>(0, selfManaged);
}

namespace detail {
    template<typename>
    struct is_duration: std::false_type {};

    template<typename Rep, typename Period>
    struct is_duration<std::chrono::duration<Rep, Period>>: std::true_type {};
}

template<typename T>
concept StdChronoDuration = detail::is_duration<T>::value;

template<StdChronoDuration Duration>
[[nodiscard]] auto toSeconds(Duration duration) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

template<StdChronoDuration Duration>
[[nodiscard]] auto toMilliSeconds(Duration duration) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count();
}

template<StdChronoDuration Duration>
[[nodiscard]] auto toMicroSeconds(Duration duration) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(duration).count();
}

template<StdChronoDuration Duration>
[[nodiscard]] auto toNanoSeconds(Duration duration) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(duration).count();
}

enum class MemoryManagerAllocateMode {
    Wait,    ///< get is blocking and waits for the memory to be available
    Dynamic, ///< allocates new memory if none is available
    Fail,    ///< return nullptr directly if no memory is availble
};

template<class T>
class AutoReleaseMemoryPool: public std::enable_shared_from_this<AutoReleaseMemoryPool<T>> {
public:
    struct Deleter {
        std::weak_ptr<AutoReleaseMemoryPool> weakPool_;

        void operator()(T *data) const {
            if(const auto pool = weakPool_.lock()) {
                dynamic_cast<AutoReleaseMemoryPool*>(pool.get())->release(data);
            }
            else {
                delete data;
            }
        }
    };

    ~AutoReleaseMemoryPool() = default;

    AutoReleaseMemoryPool(const AutoReleaseMemoryPool&) = delete;
    AutoReleaseMemoryPool& operator=(const AutoReleaseMemoryPool&) = delete;
    AutoReleaseMemoryPool(AutoReleaseMemoryPool&&) = delete;
    AutoReleaseMemoryPool& operator=(AutoReleaseMemoryPool&&) = delete;

    static std::shared_ptr<AutoReleaseMemoryPool> create(const std::string &label = "") {
        return std::shared_ptr<AutoReleaseMemoryPool>(new AutoReleaseMemoryPool(label));
    }

    [[nodiscard]] std::shared_ptr<T> allocate(const MemoryManagerAllocateMode mode = MemoryManagerAllocateMode::Fail) {
        auto lock = std::unique_lock(mutex_);

        if(memory_.empty()) {
            switch(mode) {
                case MemoryManagerAllocateMode::Wait: {
                    const auto start = std::chrono::steady_clock::now();
                    cv_.wait(lock, [&]() { return !memory_.empty(); });
                    waitTime_ += toNanoSeconds(std::chrono::steady_clock::now() - start);
                    break;
                }

                case MemoryManagerAllocateMode::Dynamic: {
                    if constexpr(std::is_default_constructible_v<T>) {
                        memory_.push_back(std::make_unique<T>());
                    }
                    else {
                        return nullptr;
                    }
                    break;
                }

                case MemoryManagerAllocateMode::Fail: {
                    return nullptr;
                }
            }
        }

        auto data = std::move(memory_.back());
        memory_.pop_back();
        return std::shared_ptr<T>(data.release(), Deleter{this->shared_from_this()});
    }

    void release(T *data) {
        if(data == nullptr) return;

        if constexpr(true) {
            const auto lg = std::lock_guard(mutex_);
            if(!contains(data))
                memory_.emplace_back(std::unique_ptr<T>(data));
        }
        cv_.notify_all();
    }

    // FIXME: RISKY AF, expects raw pointers
    // WARNING: DON'T PASS pointers from smart pointers!
    void emplaceUnmanaged(T *pData) {
        const auto lg = std::lock_guard(mutex_);
        memory_.emplace_back(std::unique_ptr<T>(pData));
        capacity_++;
    }

    template<typename ...Args>
    void fill(const int32_t capacity, Args&& ...args) {
        capacity_ = capacity;
        const auto lg = std::lock_guard(mutex_);
        for(int32_t i = 0; i < capacity; ++i) {
            memory_.emplace_back(std::make_unique<T>(args...));
        }
    }

    [[nodiscard]] auto capacity() const { return capacity_; }

    [[nodiscard]] auto available() const {
        const auto lg = std::lock_guard(mutex_);
        return memory_.size();
    }

    [[nodiscard]] std::string extraPrintingInformation() const {
        if(1.e9 <= waitTime_) {
            return "Memory Wait Time: " + std::to_string(waitTime_/1.e9) + "s";
        }
        if(1.e6 <= waitTime_) {
            return "Memory Wait Time: " + std::to_string(waitTime_/1.e6) + "ms";
        }
        if(1.e3 <= waitTime_) {
            return "Memory Wait Time: " + std::to_string(waitTime_/1.e3) + "us";
        }
        return "Memory Wait Time: " + std::to_string(waitTime_) + "ns";
    }

protected:
    explicit AutoReleaseMemoryPool(std::string label): label_(std::move(label)) {}

private:
    [[nodiscard]] bool contains(T *pData) {
        for(auto& el: memory_) {
            if(pData == el.get())
                return true;
        }

        return false;
    }

private:
    std::string                     label_    = {};
    std::vector<std::unique_ptr<T>> memory_   = {};
    mutable std::mutex              mutex_    = {};
    std::condition_variable         cv_       = {};
    int32_t                         capacity_ = 0;
    double                          waitTime_ = 0;
};


#endif //HEDGEHOG_LPBF_UTILITY_H
