// SPDX-License-Identifier: 0BSD
// ---------------------------------------------------------------------------
// Shared scalar bundle for the single-GPU LPBF 2D z-averaged step.
//
// This header is C++17-safe and contains NO CUDA tokens and NO Hedgehog
// dependency, so it can be #included by BOTH:
//   * nvcc translation units  (src/gpu/kernels.cu, src/gpu/lpbf_cuda.cu), and
//   * a plain g++ C++20 translation unit (src/gpu/hh_gpu_main.cpp).
//
// StepConst / Partial are POD and are the ABI passed across the g++<->nvcc
// seam (see include/lpbf_kernels.hpp).  make_const() is pure host arithmetic.
// ---------------------------------------------------------------------------
#pragma once
#include <cmath>
#include "params.hpp"

// All scalars the field-update kernel needs.  Provably identical constants for
// the CPU reference stepper and the GPU kernels (both consume StepConst).
struct StepConst {
    int    nx, ny;
    double dx, dy, inv_dx2, inv_dy2, inv_r0_sq;
    double alpha, K, rho, cp, rho_cp, inv_h, h, dt;
    double peak_S, eta, P;
    double hConv, Tamb, Tsub, dSub, eps;
    bool   radiation;
    double kmult;
    double x_l, y_l;   // laser position (updated every step)
    double q_scale;    // single-track: always 1
};

// Per-block partial reduction result: Tmax + argmax(min-index tie) + molten
// ymin/ymax.  Reproduces measure() semantics from src/main.cpp:164-211.
struct Partial { double tmax; long long arg; int ymin; int ymax; };

// Pure host arithmetic (line-for-line from the original lpbf_cuda.cu:157-175).
inline StepConst make_const(lpbf::Material const& mat, lpbf::Process const& pr,
                            lpbf::Domain const& dom, double dt) {
    StepConst k{};
    k.nx = dom.nx;  k.ny = dom.ny;
    k.dx = dom.dx(); k.dy = dom.dy();
    k.inv_dx2 = 1.0 / (k.dx * k.dx);
    k.inv_dy2 = 1.0 / (k.dy * k.dy);
    k.inv_r0_sq = 1.0 / (pr.r0 * pr.r0);
    k.alpha = mat.alpha(); k.K = mat.k; k.rho = mat.rho; k.cp = mat.cp;
    k.rho_cp = mat.rho * mat.cp; k.inv_h = 1.0 / pr.h; k.h = pr.h; k.dt = dt;
    k.peak_S = (2.0 * mat.eta * pr.P) / (M_PI * pr.r0 * pr.r0);  // step_task.hpp:68
    k.eta = mat.eta; k.P = pr.P;
    k.hConv = pr.hConv; k.Tamb = pr.Tamb; k.Tsub = pr.Tsub; k.dSub = pr.dSub;
    k.eps = mat.eps; k.radiation = pr.radiation;
    k.kmult = 1.0;   // overwritten by the caller with the real kmult
    k.q_scale = 1.0; // single-track driver: beam never jumps (q_scale == 1)
    k.x_l = 0.0; k.y_l = 0.0;
    return k;
}
