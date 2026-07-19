#ifndef HEDGEHOG_LPBF_UTILITY_H
#define HEDGEHOG_LPBF_UTILITY_H

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

#endif //HEDGEHOG_LPBF_UTILITY_H
