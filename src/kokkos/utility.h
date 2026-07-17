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

namespace lpbf_dev {

// params.hpp:11  Stefan-Boltzmann
// NOTE ON LINKAGE: every helper below carries the `inline` keyword (not merely
// __forceinline__, which is an inlining hint with no bearing on linkage) and
// the __constant__ tables are `static`.  This makes the header safe to #include
// from more than one translation unit without duplicate-symbol link errors on
// nvcc 12.x.  Today it is pulled in only by src/gpu/lpbf_cuda.cu.
inline double sigma_sb() { return 5.670374419e-8; }

// params.hpp:63-65
inline double T_SOL() { return 1533.0; }  // K
inline double T_LIQ() { return 1609.0; }  // K
inline double LF()    { return 210.0e3; } // J/kg

// params.hpp:67  N_PTS = 9
static constexpr int N_PTS = 9;

// params.hpp:68-73  Mills 9-point piecewise-linear tables.
// __constant__ so device code can index them; a mirrored host array (below)
// serves the host reference stepper.  `static` keeps them TU-local.
static double d_T_PTS [N_PTS] = { 298,  473,  673,  873, 1073,
                                                  1273, 1473, 1533, 1609 };
static double d_K_PTS [N_PTS] = { 8.9, 12.9, 15.8, 18.7, 21.9,
                                                  25.6, 28.6, 29.3, 29.6 };
static double d_CP_PTS[N_PTS] = { 435,  479,  515,  558,  580,
                                                   628,  670,  685,  720 };

static const double h_T_PTS [N_PTS] = { 298,  473,  673,  873, 1073,
                                       1273, 1473, 1533, 1609 };
static const double h_K_PTS [N_PTS] = { 8.9, 12.9, 15.8, 18.7, 21.9,
                                       25.6, 28.6, 29.3, 29.6 };
static const double h_CP_PTS[N_PTS] = { 435,  479,  515,  558,  580,
                                        628,  670,  685,  720 };

// params.hpp:75-83  clamp-at-ends + linear-search piecewise-linear interp.
inline
double interp(double T, const double* xs, const double* ys, int n) {
    if (T <= xs[0])     return ys[0];
    if (T >= xs[n - 1]) return ys[n - 1];
    int i = 1;
    while (T > xs[i]) ++i;
    const double f = (T - xs[i - 1]) / (xs[i] - xs[i - 1]);
    return ys[i - 1] + f * (ys[i] - ys[i - 1]);
}

// params.hpp:85-86  k_of / cp_of.  Split host/device to pick the right table.
inline double k_of_dev (double T) { return interp(T, d_T_PTS, d_K_PTS,  N_PTS); }
inline double cp_of_dev(double T) { return interp(T, d_T_PTS, d_CP_PTS, N_PTS); }
inline double k_of_host (double T) { return interp(T, h_T_PTS, h_K_PTS,  N_PTS); }
inline double cp_of_host(double T) { return interp(T, h_T_PTS, h_CP_PTS, N_PTS); }

// params.hpp:93-100  Marangoni conductivity enhancement (kmult<=1 -> no-op).
inline double k_enh(double T, double kmult) {
    if (kmult <= 1.0) return 1.0;
    double s;
    if      (T <= T_SOL()) s = 0.0;
    else if (T >= T_LIQ()) s = 1.0;
    else                   s = (T - T_SOL()) / (T_LIQ() - T_SOL());
    return 1.0 + (kmult - 1.0) * s;
}

// params.hpp:101-103  k_eff = k_of * k_enh
inline double k_eff_dev(double T, double kmult) {
    return k_of_dev(T) * k_enh(T, kmult);
}
inline double k_eff_host(double T, double kmult) {
    return k_of_host(T) * k_enh(T, kmult);
}

// params.hpp:107-111  apparent-cp latent heat over the mushy interval.
inline double cp_eff_dev(double T) {
    double c = cp_of_dev(T);
    if (T >= T_SOL() && T <= T_LIQ()) c += LF() / (T_LIQ() - T_SOL());
    return c;
}
inline double cp_eff_host(double T) {
    double c = cp_of_host(T);
    if (T >= T_SOL() && T <= T_LIQ()) c += LF() / (T_LIQ() - T_SOL());
    return c;
}

// params.hpp:116-121  host-only CFL precompute is intentionally NOT mirrored
// here: dt for BOTH the CPU and GPU paths is taken from params.hpp's
// dt_cfl_ext -> in718::alpha_eff_max (see lpbf_cuda.cu:579), so a local copy
// would be dead code that could silently drift from params.hpp.  Keep the
// single source of truth in params.hpp:116-121.

// ---------------------------------------------------------------------------
// 3D anisotropic extension (build spec §2).  Two independent enhancement
// multipliers replace the single scalar kmult: `m_lat` scales the lateral
// (x,y) face conductivity, `m_z` scales the vertical (z) face conductivity.
// Both reuse the SAME temperature ramp k_enh(T,M) above, so m_lat==m_z==M
// recovers the isotropic 2D enhancement exactly and m_lat==m_z==1 recovers
// pure Mills conduction.  These are two-line copies of k_eff_{dev,host}.
// ---------------------------------------------------------------------------
inline double k_eff_lat_dev(double T, double m_lat) {
    return k_of_dev(T) * k_enh(T, m_lat);
}
inline double k_eff_z_dev(double T, double m_z) {
    return k_of_dev(T) * k_enh(T, m_z);
}
inline double k_eff_lat_host(double T, double m_lat) {
    return k_of_host(T) * k_enh(T, m_lat);
}
inline double k_eff_z_host(double T, double m_z) {
    return k_of_host(T) * k_enh(T, m_z);
}

// Optional evaporative recoil-cooling sink (build spec §1), gated in the kernel
// by StepConst3D::evaporation (default OFF).  Simplified Anisimov/Knight form:
// a saturated-vapour mass flux driven by a Clausius-Clapeyron vapour pressure
// carries away its latent heat of vaporisation.  Lumped Ni-based-alloy vapour
// constants (IN718 is ~Ni/Fe/Cr); this is an honest second knob for the deepest
// keyhole rows, NOT a validated evaporation model.  Returns W/m^2.
//
// CALIBRATION POLICY: this sink is kept OFF (StepConst3D::evaporation=false, only
// enabled by --evap) for the headline eta/Mlat/Mz calibration.  Because it
// monotonically removes surface heat above ~300 K it would act as a free
// peak_T/width reducer that could absorb model-form error into an unvalidated
// parameter; if it is ever enabled it MUST be disclosed as a fitted knob and the
// posterior reported both with and without it.
inline double q_evap(double T) {
    const double Lv = 6.4e6;      // J/kg  latent heat of vaporisation
    const double Mm = 59.0e-3;    // kg/mol mean atomic mass (~Ni/Fe)
    const double Rg = 8.314462618;// J/mol/K
    const double Tb = 3190.0;     // K   boiling point at 1 atm
    const double p0 = 101325.0;   // Pa
    const double PI = 3.14159265358979323846;
    if (T <= 300.0) return 0.0;
    const double psat = p0 * exp((Lv * Mm / Rg) * (1.0 / Tb - 1.0 / T));
    const double jmass = 0.82 * psat * sqrt(Mm / (2.0 * PI * Rg * T));
    return Lv * jmass;
}

}  // namespace lpbf_dev

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
