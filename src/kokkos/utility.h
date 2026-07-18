#ifndef HEDGEHOG_LPBF_UTILITY_H
#define HEDGEHOG_LPBF_UTILITY_H

#include "params.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <Kokkos_Core.hpp>

using namespace lpbf;

// ---------------------------------------------------------------------------
// CLI  (superset of src/main.cpp parse_cli)
// ---------------------------------------------------------------------------
struct CLI {
    std::string case_id   = "0";
    int         n_threads = 4;        // parsed & ignored on GPU (kept for parity)
    int         ntiles    = 8;        // parsed & ignored on GPU (kept for parity)
    double      t_end     = 1.5e-3;
    int         snap_every = 100;
    std::string physics   = "base";
    std::string out_dir;
    double P_ovr = -1, V_ovr = -1, r0_ovr = -1;
    double eta_ovr = -1, dsub_ovr = -1, hconv_ovr = -1;
    double kmult = 1.0;
    // GPU-only:
    int    device      = 0;
    bool   do_check    = false;
    bool   do_bench    = false;
    double check_rtol  = 1e-6;
    double check_atol  = 1e-3;
};

static CLI parse_cli(int argc, char** argv) {
    CLI c;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto starts = [&](char const* p) { return a.rfind(p, 0) == 0; };
        if      (starts("--case="))       c.case_id    = a.substr(7);
        else if (starts("--threads="))    c.n_threads  = std::atoi(a.c_str() + 10);
        else if (starts("--tiles="))      c.ntiles     = std::atoi(a.c_str() +  8);
        else if (starts("--tend="))       c.t_end      = std::atof(a.c_str() +  7);
        else if (starts("--snap-every=")) c.snap_every = std::atoi(a.c_str() + 13);
        else if (starts("--physics="))    c.physics    = a.substr(10);
        else if (starts("--out="))        c.out_dir    = a.substr(6);
        else if (starts("--P="))          c.P_ovr      = std::atof(a.c_str() +  4);
        else if (starts("--V="))          c.V_ovr      = std::atof(a.c_str() +  4);
        else if (starts("--r0="))         c.r0_ovr     = std::atof(a.c_str() +  5);
        else if (starts("--eta="))        c.eta_ovr    = std::atof(a.c_str() +  6);
        else if (starts("--dsub="))       c.dsub_ovr   = std::atof(a.c_str() +  7);
        else if (starts("--hconv="))      c.hconv_ovr  = std::atof(a.c_str() +  8);
        else if (starts("--kmult="))      c.kmult      = std::atof(a.c_str() +  8);
        else if (starts("--device="))     c.device     = std::atoi(a.c_str() +  9);
        else if (starts("--check-rtol=")) c.check_rtol = std::atof(a.c_str() + 13);
        else if (starts("--check-atol=")) c.check_atol = std::atof(a.c_str() + 13);
        else if (a == "--check")          c.do_check   = true;
        else if (a == "--bench")          c.do_bench   = true;
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "Usage: hedgehog_lpbf_gpu [options]   (CUDA 2D z-averaged solver)\n"
                "  --case=ID            0,1.1,1.2,2.1,2.2,3.1,3.2 (default 0)\n"
                "  --physics=base|ext   constant props | Mills k(T),cp(T)+latent\n"
                "  --P=W --V=M/S --r0=M     process overrides (SI)\n"
                "  --eta=F --dsub=M --hconv=W/M2K   model overrides\n"
                "  --kmult=F            melt-pool conductivity enhancement\n"
                "  --out=DIR            output directory override\n"
                "  --tend=SECS          physical end-time (default 1.5e-3)\n"
                "  --snap-every=N       snapshot stride (0: final only)\n"
                "  --threads=N --tiles=N   accepted for CLI parity, ignored on GPU\n"
                "  --device=N           CUDA device (default 0)\n"
                "  --check              CPU vs GPU deviation report (exit!=0 on fail)\n"
                "  --check-rtol=F       --check relative-T tolerance (default 1e-6)\n"
                "  --check-atol=F       --check absolute-T tolerance K (default 1e-3;\n"
                "                       long runs: build STRICT_FMAD=ON or use ~1e-1)\n"
                "  --bench              CPU & GPU steps/s + wall time\n";
            std::exit(0);
        }
    }
    return c;
}

static void apply_case(std::string const& id, Process& pr) {
    if      (id == "0")   { pr.P = 285.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else if (id == "1.1") { pr.P = 285.0; pr.V = 0.96; pr.r0 = 24.5e-6; }
    else if (id == "1.2") { pr.P = 285.0; pr.V = 0.96; pr.r0 = 41.0e-6; }
    else if (id == "2.1") { pr.P = 285.0; pr.V = 1.20; pr.r0 = 33.5e-6; }
    else if (id == "2.2") { pr.P = 285.0; pr.V = 0.80; pr.r0 = 33.5e-6; }
    else if (id == "3.1") { pr.P = 325.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else if (id == "3.2") { pr.P = 245.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else { std::cerr << "unknown case '" << id << "'\n"; std::exit(1); }
}

struct StepMetrics { double peak_T=0, W_um=0, W_sub_um=0; int x_hot=0; };

struct RunResult { double peak_T, peak_W, steady_W_sub; long long n_steps; double wall_ms; std::vector<double> Tfinal; };

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
