// SPDX-License-Identifier: 0BSD
// ---------------------------------------------------------------------------
// Device-side mirror of the IN718 extended-physics property model from
// include/params.hpp (namespace lpbf::in718).
//
// WHY A MIRROR INSTEAD OF ANNOTATING params.hpp:
//   params.hpp is shared verbatim with the CPU binaries, which are compiled
//   by a plain C++ compiler that does not define __host__/__device__.  To keep
//   the CPU code paths untouched (this repo has a bit-identity culture) we do
//   NOT edit params.hpp.  Instead every function below is a line-for-line copy
//   of the corresponding params.hpp function, annotated __host__ __device__.
//
//   The one real porting gotcha (see porting contract): params.hpp::interp
//   takes pointers to namespace-scope `inline constexpr` tables.  Taking the
//   address of a host constexpr array from device code is ill-formed, so the
//   9-point Mills tables are declared __device__ __constant__ here and the
//   device interp reads them directly.  A host copy of interp keeps the same
//   arithmetic for the CPU reference stepper used by --check.
//
//   Every constant and expression below is annotated with the params.hpp line
//   it reproduces so the two can be diffed by eye.
// ---------------------------------------------------------------------------
#pragma once
#include <cmath>

namespace lpbf_dev {

// params.hpp:11  Stefan-Boltzmann
// NOTE ON LINKAGE: every helper below carries the `inline` keyword (not merely
// __forceinline__, which is an inlining hint with no bearing on linkage) and
// the __constant__ tables are `static`.  This makes the header safe to #include
// from more than one translation unit without duplicate-symbol link errors on
// nvcc 12.x.  Today it is pulled in only by src/gpu/lpbf_cuda.cu.
__host__ __device__ inline double sigma_sb() { return 5.670374419e-8; }

// params.hpp:63-65
__host__ __device__ inline double T_SOL() { return 1533.0; }  // K
__host__ __device__ inline double T_LIQ() { return 1609.0; }  // K
__host__ __device__ inline double LF()    { return 210.0e3; } // J/kg

// params.hpp:67  N_PTS = 9
static constexpr int N_PTS = 9;

// params.hpp:68-73  Mills 9-point piecewise-linear tables.
// __constant__ so device code can index them; a mirrored host array (below)
// serves the host reference stepper.  `static` keeps them TU-local.
static __device__ __constant__ double d_T_PTS [N_PTS] = { 298,  473,  673,  873, 1073,
                                                  1273, 1473, 1533, 1609 };
static __device__ __constant__ double d_K_PTS [N_PTS] = { 8.9, 12.9, 15.8, 18.7, 21.9,
                                                  25.6, 28.6, 29.3, 29.6 };
static __device__ __constant__ double d_CP_PTS[N_PTS] = { 435,  479,  515,  558,  580,
                                                   628,  670,  685,  720 };

static const double h_T_PTS [N_PTS] = { 298,  473,  673,  873, 1073,
                                       1273, 1473, 1533, 1609 };
static const double h_K_PTS [N_PTS] = { 8.9, 12.9, 15.8, 18.7, 21.9,
                                       25.6, 28.6, 29.3, 29.6 };
static const double h_CP_PTS[N_PTS] = { 435,  479,  515,  558,  580,
                                        628,  670,  685,  720 };

// params.hpp:75-83  clamp-at-ends + linear-search piecewise-linear interp.
__host__ __device__ inline
double interp(double T, const double* xs, const double* ys, int n) {
    if (T <= xs[0])     return ys[0];
    if (T >= xs[n - 1]) return ys[n - 1];
    int i = 1;
    while (T > xs[i]) ++i;
    const double f = (T - xs[i - 1]) / (xs[i] - xs[i - 1]);
    return ys[i - 1] + f * (ys[i] - ys[i - 1]);
}

// params.hpp:85-86  k_of / cp_of.  Split host/device to pick the right table.
__device__ inline double k_of_dev (double T) { return interp(T, d_T_PTS, d_K_PTS,  N_PTS); }
__device__ inline double cp_of_dev(double T) { return interp(T, d_T_PTS, d_CP_PTS, N_PTS); }
__host__            inline double k_of_host (double T) { return interp(T, h_T_PTS, h_K_PTS,  N_PTS); }
__host__            inline double cp_of_host(double T) { return interp(T, h_T_PTS, h_CP_PTS, N_PTS); }

// params.hpp:93-100  Marangoni conductivity enhancement (kmult<=1 -> no-op).
__host__ __device__ inline double k_enh(double T, double kmult) {
    if (kmult <= 1.0) return 1.0;
    double s;
    if      (T <= T_SOL()) s = 0.0;
    else if (T >= T_LIQ()) s = 1.0;
    else                   s = (T - T_SOL()) / (T_LIQ() - T_SOL());
    return 1.0 + (kmult - 1.0) * s;
}

// params.hpp:101-103  k_eff = k_of * k_enh
__device__ inline double k_eff_dev(double T, double kmult) {
    return k_of_dev(T) * k_enh(T, kmult);
}
__host__ inline double k_eff_host(double T, double kmult) {
    return k_of_host(T) * k_enh(T, kmult);
}

// params.hpp:107-111  apparent-cp latent heat over the mushy interval.
__device__ inline double cp_eff_dev(double T) {
    double c = cp_of_dev(T);
    if (T >= T_SOL() && T <= T_LIQ()) c += LF() / (T_LIQ() - T_SOL());
    return c;
}
__host__ inline double cp_eff_host(double T) {
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
__device__ inline double k_eff_lat_dev(double T, double m_lat) {
    return k_of_dev(T) * k_enh(T, m_lat);
}
__device__ inline double k_eff_z_dev(double T, double m_z) {
    return k_of_dev(T) * k_enh(T, m_z);
}
__host__ inline double k_eff_lat_host(double T, double m_lat) {
    return k_of_host(T) * k_enh(T, m_lat);
}
__host__ inline double k_eff_z_host(double T, double m_z) {
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
__host__ __device__ inline double q_evap(double T) {
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
