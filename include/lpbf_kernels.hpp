// SPDX-License-Identifier: 0BSD
// ---------------------------------------------------------------------------
// The single seam between host code and device code.
//
// These extern "C" launch wrappers hide all <<<>>> launch syntax and every
// __global__/__device__ token behind an ordinary C ABI.  They are DEFINED in
// the nvcc TU src/gpu/kernels.cu and DECLARED here for consumption by:
//   * the standalone nvcc binary   (src/gpu/lpbf_cuda.cu), and
//   * the g++/C++20 Hedgehog binary (src/gpu/hh_gpu_main.cpp).
//
// Only <cuda_runtime.h> (for cudaStream_t) and the POD StepConst/Partial are
// required — no Hedgehog, no device-physics header.  This is what makes the
// Hedgehog+CUDA main host-compilable by g++.
// ---------------------------------------------------------------------------
#pragma once
#include <cuda_runtime.h>
#include "lpbf_step_const.hpp"

extern "C" {

// Launch the full-grid explicit-Euler field update on `stream`.
//   phys: 0 = baseline (constant properties), 1 = extended (Mills k(T),cp(T)).
// The wrapper computes its own dim3 block/grid config internally.
void lpbf_launch_step(const double* Tin, double* Tout,
                      const StepConst* k, int phys, cudaStream_t stream);

// Launch the block-partial reduction (Tmax + argmax + molten ymin/ymax) on
// `stream`.  Writes `nblocks` Partial records to `out`; the caller copies them
// D2H and finishes the cross-block reduction on the host.
void lpbf_launch_reduce(const double* T, int nx, int ny, double Tm,
                        Partial* out, int nblocks, cudaStream_t stream);

// OPT #3: fused step + reduce in ONE grid launch.  Advances the field AND
// produces ((nx+15)/16)*((ny+15)/16) per-block Partials in `out`, bit-identical
// to running lpbf_launch_step followed by lpbf_launch_reduce.  Finalize with
// lpbf_launch_reduce_finalize (nblocks = that same product).
void lpbf_launch_step_reduce(const double* Tin, double* Tout,
                             const StepConst* k, int phys, double Tm,
                             Partial* out, cudaStream_t stream);

// OPT #2: finish the cross-block reduction ON THE DEVICE (single Partial in
// `out`), bit-identical to the host finalize loop.  Lets the batched step task
// defer the Partial D2H + stream sync to once per batch.
void lpbf_launch_reduce_finalize(const Partial* parts, int nblocks,
                                 int nx, int ny, Partial* out,
                                 cudaStream_t stream);

// OPT #2: stage the hottest column (x_hot = out->arg % nx of the finalized
// Partial) into `out_col` device-side, for one bulk D2H per batch instead of a
// per-step strided 2D copy + sync.
void lpbf_launch_extract_col(const double* T, int nx, int ny,
                             const Partial* glob, double* out_col,
                             cudaStream_t stream);

}  // extern "C"
