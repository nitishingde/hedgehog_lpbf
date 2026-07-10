// SPDX-License-Identifier: 0BSD
// ===========================================================================
// hedgehog_lpbf_gpu — single-GPU CUDA port of the 2D z-averaged explicit-FD
// LPBF heat-equation step (src/main.cpp + include/step_task.hpp).
//
// SCOPE: 2D z-averaged single-track solver ONLY.
//   The 3D solver (step3d_task.hpp) uses a different storage layout
//   ((r*nz+k)*nx+i), a 7-point Laplacian, a Dirichlet substrate floor, and
//   has NO extended physics — a genuinely different kernel pattern.  Per the
//   porting contract it is intentionally left out of this port; only the 2D
//   z-averaged step (baseline + extended physics) is ported here.
//
// FAITHFULNESS: the field update is a full-grid stencil kernel, one CUDA
// thread per cell, fp64, one T/T_new double-buffer swap per time step.  The
// CPU driver's per-tile halo mirror is replaced by an in-kernel "ghost =
// centre" mirror at all four domain edges, which reproduces the CPU boundary
// conditions exactly (see contract §2: the update is tiling-independent, so a
// single full-grid kernel == the CPU result up to exp()/FMA rounding).
//
// The physics is mirrored, line-for-line, in include/lpbf_device_physics.cuh.
//
// CLI: identical to the CPU binary, plus:
//   --device=N     select CUDA device (default 0)
//   --check        run CPU + GPU for the full step count, print max abs/rel
//                  T deviation at the final step and per-metric deviations,
//                  exit nonzero if the relative T deviation exceeds --check-rtol
//   --check-rtol=  pass/fail relative-T tolerance for --check (default 1e-6)
//   --check-atol=  pass/fail absolute-T tolerance in K for --check (default 1e-3)
//   --bench        report steps/s and wall time for both the CPU and GPU paths
//
// NOTE ON --check TOLERANCES: the field update has no reordered summation
// (each cell is an independent expression), so the ONLY CPU/GPU divergence is
// per-cell exp()/FMA rounding.  Over the ~1400 explicit-Euler steps of a full
// run, that rounding can accumulate at the sharp laser peak (T~2000 K) past
// BOTH the tight defaults (rtol 1e-6 == 2 mK, atol 1e-3 K), so a CORRECT port
// can report FAIL.  For a bit-comparable long-run check, build with
// -DHEDGEHOG_LPBF_STRICT_FMAD=ON (nvcc --fmad=false); otherwise relax the
// bound for long runs, e.g. --check-atol=1e-1.  A single-step check (small
// --tend, exp/FMA-only, no accumulation) isolates one-step rounding.
// ===========================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "params.hpp"                 // structs + host CFL functions (C++17-safe)
#include "lpbf_device_physics.cuh"    // __host__ __device__ mirror of in718::
#include "lpbf_step_const.hpp"        // shared POD StepConst / Partial + make_const
#include "lpbf_kernels.hpp"           // extern "C" launch wrappers (defined in kernels.cu)

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

// src/main.cpp:97-109
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

// StepConst / Partial / make_const now live in include/lpbf_step_const.hpp
// (shared with kernels.cu and the Hedgehog+CUDA main).

// ===========================================================================
// CPU reference stepper (host) — line-for-line equivalent of step_task.hpp,
// operating on the full grid with a global "ghost=centre" mirror BC (which is
// exactly what the CPU driver's per-tile halo mirror realises, contract §2).
// Used by --check and as the CPU baseline in --bench.
// ===========================================================================
static void cpu_step(const std::vector<double>& Tin, std::vector<double>& Tout,
                     const StepConst& k, Physics phys) {
    const int nx = k.nx, ny = k.ny;
    for (int y = 0; y < ny; ++y) {
        const double ypos = static_cast<double>(y) * k.dy;
        for (int x = 0; x < nx; ++x) {
            const double T = Tin[y * nx + x];
            const double xpos = static_cast<double>(x) * k.dx;
            const double r2 = (xpos - k.x_l) * (xpos - k.x_l)
                            + (ypos - k.y_l) * (ypos - k.y_l);
            if (phys == Physics::Baseline) {          // step_task.hpp:87-131
                double lap_x;
                if      (x == 0)      lap_x = (Tin[y*nx+x+1] - T) * k.inv_dx2;
                else if (x == nx - 1) lap_x = (Tin[y*nx+x-1] - T) * k.inv_dx2;
                else lap_x = (Tin[y*nx+x+1] - 2.0*T + Tin[y*nx+x-1]) * k.inv_dx2;
                const double Tym = (y == 0)      ? T : Tin[(y-1)*nx+x];
                const double Typ = (y == ny - 1) ? T : Tin[(y+1)*nx+x];
                const double lap_y = (Typ - 2.0*T + Tym) * k.inv_dy2;
                const double S = k.q_scale * k.peak_S
                               * std::exp(-2.0 * r2 * k.inv_r0_sq)
                               / (k.rho_cp * k.h);
                double q_top = k.hConv * (T - k.Tamb);
                if (k.radiation) {
                    const double T4 = T*T*T*T, Ta4 = k.Tamb*k.Tamb*k.Tamb*k.Tamb;
                    q_top += k.eps * lpbf_dev::sigma_sb() * (T4 - Ta4);
                }
                const double q_bot = (k.K / k.dSub) * (T - k.Tsub);
                const double loss = (q_top + q_bot) * k.inv_h / k.rho_cp;
                const double dTdt = k.alpha * (lap_x + lap_y) + S - loss;
                Tout[y*nx+x] = T + k.dt * dTdt;
            } else {                                  // step_task.hpp:132-187
                const double Txm = (x == 0)      ? T : Tin[y*nx+x-1];
                const double Txp = (x == nx - 1) ? T : Tin[y*nx+x+1];
                const double Tym = (y == 0)      ? T : Tin[(y-1)*nx+x];
                const double Typ = (y == ny - 1) ? T : Tin[(y+1)*nx+x];
                const double kCe = lpbf_dev::k_eff_host(T, k.kmult);
                const double kxm = 0.5 * (kCe + lpbf_dev::k_eff_host(Txm, k.kmult));
                const double kxp = 0.5 * (kCe + lpbf_dev::k_eff_host(Txp, k.kmult));
                const double kym = 0.5 * (kCe + lpbf_dev::k_eff_host(Tym, k.kmult));
                const double kyp = 0.5 * (kCe + lpbf_dev::k_eff_host(Typ, k.kmult));
                const double div_k_grad =
                      (kxp*(Txp-T) - kxm*(T-Txm)) * k.inv_dx2
                    + (kyp*(Typ-T) - kym*(T-Tym)) * k.inv_dy2;
                const double q_in = k.q_scale * k.peak_S
                                  * std::exp(-2.0 * r2 * k.inv_r0_sq);
                double q_top = k.hConv * (T - k.Tamb);
                if (k.radiation) {
                    const double T4 = T*T*T*T, Ta4 = k.Tamb*k.Tamb*k.Tamb*k.Tamb;
                    q_top += k.eps * lpbf_dev::sigma_sb() * (T4 - Ta4);
                }
                const double q_bot = (lpbf_dev::k_of_host(T) / k.dSub) * (T - k.Tsub);
                const double rhs = div_k_grad + (q_in - q_top - q_bot) * k.inv_h;
                Tout[y*nx+x] = T + k.dt * rhs / (k.rho * lpbf_dev::cp_eff_host(T));
            }
        }
    }
}

// ===========================================================================
// GPU kernels (step_kernel, reduce_kernel) now live in src/gpu/kernels.cu and
// are launched through the extern "C" wrappers in include/lpbf_kernels.hpp.
// ===========================================================================

// ---------------------------------------------------------------------------
#define CUDA_OK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(e_) << " at " << __FILE__ \
    << ":" << __LINE__ << "\n"; std::exit(2); } } while (0)

struct StepMetrics { double peak_T=0, W_um=0, W_sub_um=0; int x_hot=0; };

// Host measure() — identical to src/main.cpp:164-211 (used by CPU path).
static StepMetrics measure_host(const std::vector<double>& T,
                                const Domain& dom, double Tm) {
    StepMetrics m; int y_min = dom.ny, y_max = -1; double Tmax = 0.0; int x_hot = 0;
    for (int y = 0; y < dom.ny; ++y) {
        const double* row = &T[y * dom.nx]; bool any = false;
        for (int x = 0; x < dom.nx; ++x) {
            const double v = row[x];
            if (v > Tmax) { Tmax = v; x_hot = x; }
            if (v > Tm) any = true;
        }
        if (any) { if (y < y_min) y_min = y; if (y > y_max) y_max = y; }
    }
    m.peak_T = Tmax; m.x_hot = x_hot;
    m.W_um = (y_max < 0) ? 0.0 : (y_max - y_min + 1) * dom.dy() * 1e6;
    if (Tmax > Tm) {
        const double dy = dom.dy(); int a = -1, b = -1;
        for (int y = 0; y < dom.ny; ++y)
            if (T[y*dom.nx + x_hot] > Tm) { if (a < 0) a = y; b = y; }
        if (a >= 0) {
            double y_lo = a*dy, y_hi = b*dy;
            if (a > 0) { double T0=T[(a-1)*dom.nx+x_hot], T1=T[a*dom.nx+x_hot];
                         y_lo = ((a-1) + (Tm-T0)/(T1-T0)) * dy; }
            if (b < dom.ny-1) { double T0=T[b*dom.nx+x_hot], T1=T[(b+1)*dom.nx+x_hot];
                                y_hi = (b + (Tm-T0)/(T1-T0)) * dy; }
            m.W_sub_um = (y_hi - y_lo) * 1e6;
        }
    }
    return m;
}

// GPU metrics: device reduction for the O(N) part (Tmax, x_hot, ymin, ymax),
// then copy the single hottest column (ny doubles) to host for the sub-grid
// W_sub interpolation — same semantics as measure_host.
static StepMetrics measure_gpu(const double* d_T, Partial* d_part,
                               std::vector<Partial>& h_part, int nblocks,
                               const Domain& dom, double Tm) {
    lpbf_launch_reduce(d_T, dom.nx, dom.ny, Tm, d_part, nblocks, /*stream=*/0);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(h_part.data(), d_part, nblocks * sizeof(Partial),
                       cudaMemcpyDeviceToHost));
    const long long N = (long long)dom.nx * dom.ny;
    double tmax = -1.0; long long arg = N; int ymin = dom.ny, ymax = -1;
    for (int j = 0; j < nblocks; ++j) {
        if (h_part[j].tmax > tmax ||
            (h_part[j].tmax == tmax && h_part[j].arg < arg)) {
            tmax = h_part[j].tmax; arg = h_part[j].arg;
        }
        if (h_part[j].ymin < ymin) ymin = h_part[j].ymin;
        if (h_part[j].ymax > ymax) ymax = h_part[j].ymax;
    }
    StepMetrics m;
    m.peak_T = tmax;
    m.x_hot  = (int)(arg % dom.nx);
    m.W_um   = (ymax < 0) ? 0.0 : (ymax - ymin + 1) * dom.dy() * 1e6;
    if (tmax > Tm) {
        std::vector<double> col(dom.ny);
        // Grab column x_hot: strided copy (spitch = nx*8, width = 8, height = ny)
        CUDA_OK(cudaMemcpy2D(col.data(), sizeof(double),
                             d_T + m.x_hot, dom.nx * sizeof(double),
                             sizeof(double), dom.ny, cudaMemcpyDeviceToHost));
        const double dy = dom.dy(); int a = -1, b = -1;
        for (int y = 0; y < dom.ny; ++y)
            if (col[y] > Tm) { if (a < 0) a = y; b = y; }
        if (a >= 0) {
            double y_lo = a*dy, y_hi = b*dy;
            if (a > 0)          y_lo = ((a-1) + (Tm-col[a-1])/(col[a]-col[a-1])) * dy;
            if (b < dom.ny-1)   y_hi = (b + (Tm-col[b])/(col[b+1]-col[b])) * dy;
            m.W_sub_um = (y_hi - y_lo) * 1e6;
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// Run one full simulation on the GPU.  Returns the final field in T_out and
// running maxima in peak_T/peak_W/steady_W_sub.  If write_outputs, produces
// metrics.csv / T_step_*.bin / meta.json in snap_dir.
// ---------------------------------------------------------------------------
struct RunResult { double peak_T, peak_W, steady_W_sub; long long n_steps;
                   double wall_ms; std::vector<double> Tfinal; };

static RunResult run_gpu(Material mat, Process pr, Domain dom, double dt,
                         int n_steps, Physics phys, double kmult,
                         const CLI& cli, bool write_outputs,
                         const std::string& snap_dir) {
    const int nx = dom.nx, ny = dom.ny;
    const size_t N = (size_t)nx * ny;
    StepConst k = make_const(mat, pr, dom, dt);
    k.kmult = kmult;
    const int phys_i = (phys == Physics::Extended) ? 1 : 0;

    double *d_A, *d_B;
    CUDA_OK(cudaMalloc(&d_A, N * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_B, N * sizeof(double)));
    std::vector<double> T(N, pr.Tamb);
    CUDA_OK(cudaMemcpy(d_A, T.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    int nblocks = (int)std::min<long long>(256, ((long long)N + 255) / 256);
    Partial* d_part; CUDA_OK(cudaMalloc(&d_part, nblocks * sizeof(Partial)));
    std::vector<Partial> h_part(nblocks);

    std::ofstream log;
    if (write_outputs) {
        std::filesystem::create_directories(snap_dir);
        log.open(snap_dir + "/metrics.csv");
        log << "step,t_us,x_laser_um,peakT_K,W_um\n";
    }

    double x_laser = 0.20 * dom.Lx, y_laser = 0.50 * dom.Ly;
    const double x_start = x_laser;
    double peak_T = pr.Tamb, peak_W = 0.0, steady_W_sub = 0.0;
    bool wrapped = false;

    double* d_cur = d_A; double* d_nxt = d_B;
    const auto wall0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n_steps; ++s) {
        const double t = s * dt;
        k.x_l = x_laser; k.y_l = y_laser;   // laser position for this step
        lpbf_launch_step(d_cur, d_nxt, &k, phys_i, /*stream=*/0);
        CUDA_OK(cudaGetLastError());
        std::swap(d_cur, d_nxt);            // d_cur now holds the updated field

        // advance laser (src/main.cpp:351-352)
        x_laser += pr.V * dt;
        if (x_laser > 0.90 * dom.Lx) { x_laser = x_start; wrapped = true; }

        const StepMetrics m = measure_gpu(d_cur, d_part, h_part, nblocks, dom, mat.Tm);
        if (m.peak_T > peak_T) peak_T = m.peak_T;
        if (m.W_um   > peak_W) peak_W = m.W_um;
        const bool cruise = !wrapped && x_laser >= 0.45*dom.Lx && x_laser <= 0.85*dom.Lx;
        if (cruise && m.W_sub_um > steady_W_sub) steady_W_sub = m.W_sub_um;

        if (write_outputs) {
            const bool dump = (cli.snap_every > 0 && s % cli.snap_every == 0)
                           || (s == n_steps - 1);
            if (dump) {
                CUDA_OK(cudaMemcpy(T.data(), d_cur, N*sizeof(double),
                                   cudaMemcpyDeviceToHost));
                char buf[256];
                std::snprintf(buf, sizeof(buf), "%s/T_step_%06d.bin",
                              snap_dir.c_str(), s);
                std::ofstream out(buf, std::ios::binary);
                out.write(reinterpret_cast<const char*>(T.data()),
                          (std::streamsize)(T.size()*sizeof(double)));
                log << s << ',' << t*1e6 << ',' << x_laser*1e6
                    << ',' << m.peak_T << ',' << m.W_um << '\n';
                std::cout << "  step " << s << "  t=" << t*1e3 << " ms  Tmax="
                          << m.peak_T << " K  W=" << m.W_um << " um\n";
            }
        }
    }
    CUDA_OK(cudaDeviceSynchronize());
    const double wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - wall0).count();

    RunResult rr; rr.peak_T = peak_T; rr.peak_W = peak_W;
    rr.steady_W_sub = steady_W_sub; rr.n_steps = n_steps; rr.wall_ms = wall_ms;
    rr.Tfinal.resize(N);
    CUDA_OK(cudaMemcpy(rr.Tfinal.data(), d_cur, N*sizeof(double), cudaMemcpyDeviceToHost));

    if (write_outputs) {
        int dev; cudaGetDevice(&dev); cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
        std::ofstream meta(snap_dir + "/meta.json");
        meta << "{\n"
             << "  \"case\":     \"" << cli.case_id << "\",\n"
             << "  \"backend\":  \"cuda\",\n"
             << "  \"device\":   \"" << prop.name << "\",\n"
             << "  \"physics\":  \"" << cli.physics << "\",\n"
             << "  \"kmult\":    " << kmult << ",\n"
             << "  \"eta\":      " << mat.eta << ",\n"
             << "  \"dsub_m\":   " << pr.dSub << ",\n"
             << "  \"hconv\":    " << pr.hConv << ",\n"
             << "  \"P_W\":      " << pr.P << ",\n"
             << "  \"V_mps\":    " << pr.V << ",\n"
             << "  \"r0_m\":     " << pr.r0 << ",\n"
             << "  \"nx\":       " << dom.nx << ",\n"
             << "  \"ny\":       " << dom.ny << ",\n"
             << "  \"Lx_m\":     " << dom.Lx << ",\n"
             << "  \"Ly_m\":     " << dom.Ly << ",\n"
             << "  \"Tm_K\":     " << mat.Tm << ",\n"
             << "  \"dt_s\":     " << dt << ",\n"
             << "  \"n_steps\":  " << n_steps << ",\n"
             << "  \"peak_T_K\": " << peak_T << ",\n"
             << "  \"peak_W_um\":" << peak_W << ",\n"
             << "  \"steady_W_sub_um\": " << steady_W_sub << ",\n"
             << "  \"wall_ms\":  " << wall_ms << ",\n"
             << "  \"steps_per_s\": " << (wall_ms > 0 ? n_steps*1000.0/wall_ms : 0.0) << "\n"
             << "}\n";
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_part);
    return rr;
}

// ---------------------------------------------------------------------------
// CPU reference full run (double-buffered) — for --check / --bench.
// ---------------------------------------------------------------------------
static RunResult run_cpu(Material mat, Process pr, Domain dom, double dt,
                         int n_steps, Physics phys, double kmult) {
    const size_t N = (size_t)dom.nx * dom.ny;
    StepConst k = make_const(mat, pr, dom, dt); k.kmult = kmult;
    std::vector<double> A(N, pr.Tamb), B(N, pr.Tamb);
    std::vector<double>* cur = &A; std::vector<double>* nxt = &B;
    double x_laser = 0.20*dom.Lx, y_laser = 0.50*dom.Ly; const double x_start = x_laser;
    double peak_T = pr.Tamb, peak_W = 0.0, steady_W_sub = 0.0; bool wrapped = false;
    const auto wall0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n_steps; ++s) {
        k.x_l = x_laser; k.y_l = y_laser;
        cpu_step(*cur, *nxt, k, phys);
        std::swap(cur, nxt);
        x_laser += pr.V * dt;
        if (x_laser > 0.90*dom.Lx) { x_laser = x_start; wrapped = true; }
        const StepMetrics m = measure_host(*cur, dom, mat.Tm);
        if (m.peak_T > peak_T) peak_T = m.peak_T;
        if (m.W_um   > peak_W) peak_W = m.W_um;
        const bool cruise = !wrapped && x_laser >= 0.45*dom.Lx && x_laser <= 0.85*dom.Lx;
        if (cruise && m.W_sub_um > steady_W_sub) steady_W_sub = m.W_sub_um;
    }
    const double wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - wall0).count();
    RunResult rr; rr.peak_T=peak_T; rr.peak_W=peak_W; rr.steady_W_sub=steady_W_sub;
    rr.n_steps=n_steps; rr.wall_ms=wall_ms; rr.Tfinal=*cur; return rr;
}

// ===========================================================================
int main(int argc, char** argv) {
    CLI cli = parse_cli(argc, argv);
    Material mat; Process pr; Domain dom;
    apply_case(cli.case_id, pr);
    if (cli.P_ovr     > 0)  pr.P     = cli.P_ovr;
    if (cli.V_ovr     > 0)  pr.V     = cli.V_ovr;
    if (cli.r0_ovr    > 0)  pr.r0    = cli.r0_ovr;
    if (cli.eta_ovr   > 0)  mat.eta  = cli.eta_ovr;
    if (cli.dsub_ovr  > 0)  pr.dSub  = cli.dsub_ovr;
    if (cli.hconv_ovr >= 0) pr.hConv = cli.hconv_ovr;

    const Physics phys = (cli.physics == "ext") ? Physics::Extended : Physics::Baseline;
    const double dt = (phys == Physics::Extended) ? dt_cfl_ext(mat, dom, cli.kmult)
                                                  : dt_cfl(mat, dom);
    const int n_steps = static_cast<int>(cli.t_end / dt);

    int ndev = 0; cudaGetDeviceCount(&ndev);
    if (ndev == 0) { std::cerr << "No CUDA device found.\n"; return 3; }
    if (cli.device < 0 || cli.device >= ndev) {
        std::cerr << "Invalid --device=" << cli.device << " (have " << ndev << ")\n";
        return 3;
    }
    CUDA_OK(cudaSetDevice(cli.device));
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, cli.device);

    std::cout << "hedgehog_lpbf_gpu (CUDA 2D z-averaged)\n"
              << "  device    : " << cli.device << " " << prop.name << "\n"
              << "  case      : " << cli.case_id << "  (P=" << pr.P << " W, V="
              << pr.V*1e3 << " mm/s, r0=" << pr.r0*1e6 << " um)\n"
              << "  physics   : " << (phys==Physics::Extended?"extended":"baseline") << "\n"
              << "  grid      : " << dom.nx << " x " << dom.ny << "\n"
              << "  dt        : " << dt*1e6 << " us   n_steps=" << n_steps << "\n";

    // ---- --check -------------------------------------------------------
    if (cli.do_check) {
        std::cout << "\n[--check] running CPU and GPU for " << n_steps << " steps...\n";
        RunResult cpu = run_cpu(mat, pr, dom, dt, n_steps, phys, cli.kmult);
        RunResult gpu = run_gpu(mat, pr, dom, dt, n_steps, phys, cli.kmult,
                                cli, false, "");
        double max_abs = 0.0, max_rel = 0.0;
        for (size_t i = 0; i < cpu.Tfinal.size(); ++i) {
            const double a = std::fabs(gpu.Tfinal[i] - cpu.Tfinal[i]);
            const double denom = std::max(std::fabs(cpu.Tfinal[i]), 1.0);
            if (a > max_abs) max_abs = a;
            if (a/denom > max_rel) max_rel = a/denom;
        }
        std::cout << "  final-field  max|dT|      : " << max_abs << " K\n"
                  << "  final-field  max rel dT   : " << max_rel << "\n"
                  << "  peak_T_K   CPU/GPU/|d|    : " << cpu.peak_T << " / "
                  << gpu.peak_T << " / " << std::fabs(cpu.peak_T-gpu.peak_T) << "\n"
                  << "  peak_W_um  CPU/GPU/|d|    : " << cpu.peak_W << " / "
                  << gpu.peak_W << " / " << std::fabs(cpu.peak_W-gpu.peak_W) << "\n"
                  << "  steadyWsub CPU/GPU/|d|    : " << cpu.steady_W_sub << " / "
                  << gpu.steady_W_sub << " / "
                  << std::fabs(cpu.steady_W_sub-gpu.steady_W_sub) << "\n";
        // CPU and GPU differ only by exp()/FMA rounding accumulated over
        // n_steps; there is no on-device reduction in the field update, so the
        // per-cell op order is identical.  Pass when within tolerance.
        const bool pass = (max_rel <= cli.check_rtol) || (max_abs <= cli.check_atol);
        std::cout << "  tolerance    rtol=" << cli.check_rtol
                  << " atol=" << cli.check_atol << " K  -> "
                  << (pass ? "PASS" : "FAIL") << "\n";
        return pass ? 0 : 1;
    }

    // ---- --bench -------------------------------------------------------
    if (cli.do_bench) {
        std::cout << "\n[--bench] " << n_steps << " steps each path...\n";
        RunResult gpu = run_gpu(mat, pr, dom, dt, n_steps, phys, cli.kmult,
                                cli, false, "");
        RunResult cpu = run_cpu(mat, pr, dom, dt, n_steps, phys, cli.kmult);
        auto sps = [&](double ms){ return ms>0 ? n_steps*1000.0/ms : 0.0; };
        std::cout << "  GPU  wall " << gpu.wall_ms << " ms   " << sps(gpu.wall_ms)
                  << " steps/s   peak_T=" << gpu.peak_T << "\n"
                  << "  CPU  wall " << cpu.wall_ms << " ms   " << sps(cpu.wall_ms)
                  << " steps/s   peak_T=" << cpu.peak_T << "\n"
                  << "  speedup (CPU/GPU wall) : "
                  << (gpu.wall_ms>0 ? cpu.wall_ms/(double)gpu.wall_ms : 0.0) << "x\n";
        return 0;
    }

    // ---- normal run ----------------------------------------------------
    const std::string snap_dir = !cli.out_dir.empty() ? cli.out_dir
        : "snapshots_case" + cli.case_id + (phys==Physics::Extended?"_ext":"") + "_gpu";
    RunResult rr = run_gpu(mat, pr, dom, dt, n_steps, phys, cli.kmult,
                           cli, true, snap_dir);
    std::cout << "\nDone.\n  peak T : " << rr.peak_T << " K\n"
              << "  peak W : " << rr.peak_W << " um\n"
              << "  steady W (subgrid): " << rr.steady_W_sub << " um\n"
              << "  wall   : " << rr.wall_ms << " ms  ("
              << (rr.wall_ms>0 ? n_steps*1000.0/rr.wall_ms : 0.0) << " steps/s)\n"
              << "  output : " << snap_dir << "\n";
    return 0;
}
