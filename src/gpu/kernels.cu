// SPDX-License-Identifier: 0BSD
// ===========================================================================
// kernels.cu — the ONLY translation unit that contains __global__ kernel
// bodies for the 2D z-averaged LPBF solver.  Compiled by nvcc at C++17.
//
// It exposes the kernels to host code exclusively through the extern "C"
// stream-aware launch wrappers declared in include/lpbf_kernels.hpp, so both
// the standalone nvcc binary (src/gpu/lpbf_cuda.cu) and the g++/C++20 Hedgehog
// binary (src/gpu/hh_gpu_main.cpp) link ONE definition of every kernel.
//
// The kernel bodies are copied verbatim from the original monolithic
// lpbf_cuda.cu (step_kernel / reduce_kernel) so numerical results are
// unchanged — this file only adds the launch-config + stream plumbing.
// ===========================================================================
#include <cuda_runtime.h>
#include <algorithm>

#include "lpbf_step_const.hpp"          // POD StepConst / Partial (C++17-safe)
#include "lpbf_device_physics.cuh"      // __host__ __device__ mirror of in718::
#include "lpbf_kernels.hpp"             // extern "C" wrapper declarations

// ---------------------------------------------------------------------------
// Per-cell explicit-Euler update, factored out of step_kernel so that BOTH the
// plain step_kernel and the fused step+reduce_kernel (OPT #3) evaluate the EXACT
// same floating-point expressions.  __forceinline__ means the emitted arithmetic
// is byte-identical between the two callers, so fusing the reduction cannot alter
// the field or any metric.  Body copied verbatim from the original step_kernel
// (lpbf_cuda.cu:244-302).
// ---------------------------------------------------------------------------
__device__ __forceinline__ double lpbf_cell_update(
        const double* __restrict__ Tin, long long c, int x, int y,
        const StepConst& k, int phys) {
    const int nx = k.nx, ny = k.ny;
    const double T = Tin[c];
    const double xpos = static_cast<double>(x) * k.dx;
    const double ypos = static_cast<double>(y) * k.dy;
    const double r2 = (xpos - k.x_l) * (xpos - k.x_l)
                    + (ypos - k.y_l) * (ypos - k.y_l);

    if (phys == 0) {                                   // baseline
        double lap_x;
        if      (x == 0)      lap_x = (Tin[c+1] - T) * k.inv_dx2;
        else if (x == nx - 1) lap_x = (Tin[c-1] - T) * k.inv_dx2;
        else lap_x = (Tin[c+1] - 2.0*T + Tin[c-1]) * k.inv_dx2;
        const double Tym = (y == 0)      ? T : Tin[c - nx];
        const double Typ = (y == ny - 1) ? T : Tin[c + nx];
        const double lap_y = (Typ - 2.0*T + Tym) * k.inv_dy2;
        const double S = k.q_scale * k.peak_S
                       * exp(-2.0 * r2 * k.inv_r0_sq) / (k.rho_cp * k.h);
        double q_top = k.hConv * (T - k.Tamb);
        if (k.radiation) {
            const double T4 = T*T*T*T, Ta4 = k.Tamb*k.Tamb*k.Tamb*k.Tamb;
            q_top += k.eps * lpbf_dev::sigma_sb() * (T4 - Ta4);
        }
        const double q_bot = (k.K / k.dSub) * (T - k.Tsub);
        const double loss = (q_top + q_bot) * k.inv_h / k.rho_cp;
        const double dTdt = k.alpha * (lap_x + lap_y) + S - loss;
        return T + k.dt * dTdt;
    } else {                                           // extended
        const double Txm = (x == 0)      ? T : Tin[c-1];
        const double Txp = (x == nx - 1) ? T : Tin[c+1];
        const double Tym = (y == 0)      ? T : Tin[c - nx];
        const double Typ = (y == ny - 1) ? T : Tin[c + nx];
        const double kCe = lpbf_dev::k_eff_dev(T, k.kmult);
        const double kxm = 0.5 * (kCe + lpbf_dev::k_eff_dev(Txm, k.kmult));
        const double kxp = 0.5 * (kCe + lpbf_dev::k_eff_dev(Txp, k.kmult));
        const double kym = 0.5 * (kCe + lpbf_dev::k_eff_dev(Tym, k.kmult));
        const double kyp = 0.5 * (kCe + lpbf_dev::k_eff_dev(Typ, k.kmult));
        const double div_k_grad =
              (kxp*(Txp-T) - kxm*(T-Txm)) * k.inv_dx2
            + (kyp*(Typ-T) - kym*(T-Tym)) * k.inv_dy2;
        const double q_in = k.q_scale * k.peak_S * exp(-2.0 * r2 * k.inv_r0_sq);
        double q_top = k.hConv * (T - k.Tamb);
        if (k.radiation) {
            const double T4 = T*T*T*T, Ta4 = k.Tamb*k.Tamb*k.Tamb*k.Tamb;
            q_top += k.eps * lpbf_dev::sigma_sb() * (T4 - Ta4);
        }
        const double q_bot = (lpbf_dev::k_of_dev(T) / k.dSub) * (T - k.Tsub);
        const double rhs = div_k_grad + (q_in - q_top - q_bot) * k.inv_h;
        return T + k.dt * rhs / (k.rho * lpbf_dev::cp_eff_dev(T));
    }
}

// ---------------------------------------------------------------------------
// Field update: one CUDA thread per cell, in-kernel "ghost = centre" mirror BC
// at all four domain edges.  phys 0=baseline / 1=extended.
// (lpbf_cuda.cu:244-302, VERBATIM — kept inline, NOT delegated to
// lpbf_cell_update, because factoring it into a device function changed nvcc's
// FMA contraction and altered the field below print precision, breaking the
// bit-identical T-bin reference.  Confirmed on H100: inline body == reference.)
// ---------------------------------------------------------------------------
__global__ void step_kernel(const double* __restrict__ Tin,
                            double* __restrict__ Tout,
                            StepConst k, int phys /*0=base,1=ext*/) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= k.nx || y >= k.ny) return;
    const int nx = k.nx, ny = k.ny;
    const long long c = (long long)y * nx + x;
    const double T = Tin[c];
    const double xpos = static_cast<double>(x) * k.dx;
    const double ypos = static_cast<double>(y) * k.dy;
    const double r2 = (xpos - k.x_l) * (xpos - k.x_l)
                    + (ypos - k.y_l) * (ypos - k.y_l);

    if (phys == 0) {                                   // baseline
        double lap_x;
        if      (x == 0)      lap_x = (Tin[c+1] - T) * k.inv_dx2;
        else if (x == nx - 1) lap_x = (Tin[c-1] - T) * k.inv_dx2;
        else lap_x = (Tin[c+1] - 2.0*T + Tin[c-1]) * k.inv_dx2;
        const double Tym = (y == 0)      ? T : Tin[c - nx];
        const double Typ = (y == ny - 1) ? T : Tin[c + nx];
        const double lap_y = (Typ - 2.0*T + Tym) * k.inv_dy2;
        const double S = k.q_scale * k.peak_S
                       * exp(-2.0 * r2 * k.inv_r0_sq) / (k.rho_cp * k.h);
        double q_top = k.hConv * (T - k.Tamb);
        if (k.radiation) {
            const double T4 = T*T*T*T, Ta4 = k.Tamb*k.Tamb*k.Tamb*k.Tamb;
            q_top += k.eps * lpbf_dev::sigma_sb() * (T4 - Ta4);
        }
        const double q_bot = (k.K / k.dSub) * (T - k.Tsub);
        const double loss = (q_top + q_bot) * k.inv_h / k.rho_cp;
        const double dTdt = k.alpha * (lap_x + lap_y) + S - loss;
        Tout[c] = T + k.dt * dTdt;
    } else {                                           // extended
        const double Txm = (x == 0)      ? T : Tin[c-1];
        const double Txp = (x == nx - 1) ? T : Tin[c+1];
        const double Tym = (y == 0)      ? T : Tin[c - nx];
        const double Typ = (y == ny - 1) ? T : Tin[c + nx];
        const double kCe = lpbf_dev::k_eff_dev(T, k.kmult);
        const double kxm = 0.5 * (kCe + lpbf_dev::k_eff_dev(Txm, k.kmult));
        const double kxp = 0.5 * (kCe + lpbf_dev::k_eff_dev(Txp, k.kmult));
        const double kym = 0.5 * (kCe + lpbf_dev::k_eff_dev(Tym, k.kmult));
        const double kyp = 0.5 * (kCe + lpbf_dev::k_eff_dev(Typ, k.kmult));
        const double div_k_grad =
              (kxp*(Txp-T) - kxm*(T-Txm)) * k.inv_dx2
            + (kyp*(Typ-T) - kym*(T-Tym)) * k.inv_dy2;
        const double q_in = k.q_scale * k.peak_S * exp(-2.0 * r2 * k.inv_r0_sq);
        double q_top = k.hConv * (T - k.Tamb);
        if (k.radiation) {
            const double T4 = T*T*T*T, Ta4 = k.Tamb*k.Tamb*k.Tamb*k.Tamb;
            q_top += k.eps * lpbf_dev::sigma_sb() * (T4 - Ta4);
        }
        const double q_bot = (lpbf_dev::k_of_dev(T) / k.dSub) * (T - k.Tsub);
        const double rhs = div_k_grad + (q_in - q_top - q_bot) * k.inv_h;
        Tout[c] = T + k.dt * rhs / (k.rho * lpbf_dev::cp_eff_dev(T));
    }
}

// ---------------------------------------------------------------------------
// OPT #3 — FUSED step + reduce.  Each 2D thread computes its own updated cell
// (identical arithmetic to step_kernel via lpbf_cell_update), writes it to
// Tout, and reduces its OWN value into a per-block Partial via a shared-mem
// reduction using the SAME min-index argmax tie-break and ymin/ymax init as
// reduce_kernel.  Because max-with-min-index and min/max are associative &
// commutative, the finalized Partial is bit-identical regardless of the block
// decomposition (2D 16x16 here vs the 1D grid-stride of reduce_kernel).  This
// collapses the step->reduce dependency into a single grid launch.
// out holds gridDim.x*gridDim.y Partials.
// ---------------------------------------------------------------------------
__global__ void step_reduce_kernel(const double* __restrict__ Tin,
                                   double* __restrict__ Tout,
                                   StepConst k, int phys, double Tm,
                                   Partial* __restrict__ out) {
    const int x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int nx = k.nx, ny = k.ny;
    const long long N = (long long)nx * ny;

    double   myT   = -1.0;
    long long myArg = N;
    int      myYmin = ny, myYmax = -1;
    if (x < nx && y < ny) {
        const long long c = (long long)y * nx + x;
        const double v = lpbf_cell_update(Tin, c, x, y, k, phys);
        Tout[c] = v;
        myT = v; myArg = c;
        if (v > Tm) { myYmin = y; myYmax = y; }
    }

    __shared__ double   s_tmax[256];
    __shared__ long long s_arg[256];
    __shared__ int      s_ymin[256];
    __shared__ int      s_ymax[256];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    s_tmax[tid] = myT; s_arg[tid] = myArg; s_ymin[tid] = myYmin; s_ymax[tid] = myYmax;
    __syncthreads();
    if (tid == 0) {
        const int nthreads = blockDim.x * blockDim.y;
        double bt = -1.0; long long ba = N; int bmn = ny; int bmx = -1;
        for (int j = 0; j < nthreads; ++j) {
            if (s_tmax[j] > bt || (s_tmax[j] == bt && s_arg[j] < ba)) {
                bt = s_tmax[j]; ba = s_arg[j];
            }
            if (s_ymin[j] < bmn) bmn = s_ymin[j];
            if (s_ymax[j] > bmx) bmx = s_ymax[j];
        }
        out[blockIdx.y * gridDim.x + blockIdx.x] = { bt, ba, bmn, bmx };
    }
}

// ---------------------------------------------------------------------------
// Per-block partial reduction (lpbf_cuda.cu:308-341, unchanged).
// ---------------------------------------------------------------------------
__global__ void reduce_kernel(const double* __restrict__ T, int nx, int ny,
                              double Tm, Partial* __restrict__ out) {
    __shared__ double s_tmax[256];
    __shared__ long long s_arg[256];
    __shared__ int s_ymin[256];
    __shared__ int s_ymax[256];
    const int tid = threadIdx.x;
    const long long N = (long long)nx * ny;
    double tmax = -1.0; long long arg = N; int ymin = ny; int ymax = -1;
    for (long long i = blockIdx.x * blockDim.x + tid; i < N;
         i += (long long)blockDim.x * gridDim.x) {
        const double v = T[i];
        if (v > tmax) { tmax = v; arg = i; }
        if (v > Tm) {
            const int yy = (int)(i / nx);
            if (yy < ymin) ymin = yy;
            if (yy > ymax) ymax = yy;
        }
    }
    s_tmax[tid] = tmax; s_arg[tid] = arg; s_ymin[tid] = ymin; s_ymax[tid] = ymax;
    __syncthreads();
    if (tid == 0) {
        double bt = -1.0; long long ba = N; int bmn = ny; int bmx = -1;
        for (int j = 0; j < blockDim.x; ++j) {
            if (s_tmax[j] > bt || (s_tmax[j] == bt && s_arg[j] < ba)) {
                bt = s_tmax[j]; ba = s_arg[j];
            }
            if (s_ymin[j] < bmn) bmn = s_ymin[j];
            if (s_ymax[j] > bmx) bmx = s_ymax[j];
        }
        out[blockIdx.x] = { bt, ba, bmn, bmx };
    }
}

// ---------------------------------------------------------------------------
// Device-side cross-block finalize (OPT #2 tick batching).  Reproduces the
// host reduction loop of gpu_step_task.hpp::execute VERBATIM (same sequential
// order, same min-index argmax tie-break, same ymin/ymax init) so the emitted
// Partial is BIT-IDENTICAL to the host finalize.  Single-thread by design — the
// 113 per-block partials are few, and a serial reduction guarantees identical
// ordering/tie-break to the reference host loop.  Running this on-device lets
// the batch defer the Partial D2H + stream sync to once per batch.
// ---------------------------------------------------------------------------
__global__ void reduce_finalize_kernel(const Partial* __restrict__ parts,
                                       int nblocks, int nx, int ny,
                                       Partial* __restrict__ out) {
    const long long N = (long long)nx * ny;
    double tmax = -1.0; long long arg = N; int ymin = ny; int ymax = -1;
    for (int j = 0; j < nblocks; ++j) {
        const Partial p = parts[j];
        if (p.tmax > tmax || (p.tmax == tmax && p.arg < arg)) {
            tmax = p.tmax; arg = p.arg;
        }
        if (p.ymin < ymin) ymin = p.ymin;
        if (p.ymax > ymax) ymax = p.ymax;
    }
    out[0] = { tmax, arg, ymin, ymax };
}

// Extract the single hottest column T[:, x_hot] into out_col (x_hot = arg%nx of
// the finalized Partial).  Reads the SAME device bytes the strided 2D D2H read,
// so hot_col values are bit-identical; here they are staged device-side into a
// per-batch ring for one bulk D2H instead of a per-step strided copy + sync.
__global__ void extract_col_kernel(const double* __restrict__ T, int nx, int ny,
                                   const Partial* __restrict__ glob,
                                   double* __restrict__ out_col) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ny) return;
    const int x_hot = (int)(glob->arg % nx);
    out_col[y] = T[(long long)y * nx + x_hot];
}

// ---------------------------------------------------------------------------
// extern "C" launch wrappers — the g++<->nvcc seam.
// ---------------------------------------------------------------------------
extern "C" {

void lpbf_launch_step(const double* Tin, double* Tout,
                      const StepConst* k, int phys, cudaStream_t stream) {
    dim3 blk(16, 16);
    dim3 grd((k->nx + blk.x - 1) / blk.x, (k->ny + blk.y - 1) / blk.y);
    step_kernel<<<grd, blk, 0, stream>>>(Tin, Tout, *k, phys);
}

void lpbf_launch_reduce(const double* T, int nx, int ny, double Tm,
                        Partial* out, int nblocks, cudaStream_t stream) {
    reduce_kernel<<<nblocks, 256, 0, stream>>>(T, nx, ny, Tm, out);
}

// OPT #3: fused step+reduce.  Writes gridDim.x*gridDim.y Partials into `out`
// (== ((nx+15)/16)*((ny+15)/16)); the caller finalizes them with
// lpbf_launch_reduce_finalize exactly as for the two-kernel path.
void lpbf_launch_step_reduce(const double* Tin, double* Tout,
                             const StepConst* k, int phys, double Tm,
                             Partial* out, cudaStream_t stream) {
    dim3 blk(16, 16);
    dim3 grd((k->nx + blk.x - 1) / blk.x, (k->ny + blk.y - 1) / blk.y);
    step_reduce_kernel<<<grd, blk, 0, stream>>>(Tin, Tout, *k, phys, Tm, out);
}

void lpbf_launch_reduce_finalize(const Partial* parts, int nblocks,
                                 int nx, int ny, Partial* out,
                                 cudaStream_t stream) {
    reduce_finalize_kernel<<<1, 1, 0, stream>>>(parts, nblocks, nx, ny, out);
}

void lpbf_launch_extract_col(const double* T, int nx, int ny,
                             const Partial* glob, double* out_col,
                             cudaStream_t stream) {
    const int bs = 128;
    extract_col_kernel<<<(ny + bs - 1) / bs, bs, 0, stream>>>(
        T, nx, ny, glob, out_col);
}

}  // extern "C"
