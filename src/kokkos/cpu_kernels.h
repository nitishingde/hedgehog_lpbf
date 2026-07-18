#ifndef HEDGEHOG_LPBF_CPU_KERNELS_H
#define HEDGEHOG_LPBF_CPU_KERNELS_H

#include "utility.h"
#include "lpbf_step_const.hpp"

#include <chrono>

using namespace lpbf;

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

static void cpu_step(const std::vector<double>& Tin, std::vector<double>& Tout, const StepConst& k, Physics phys) {
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

// Host measure() — identical to src/main.cpp:164-211 (used by CPU path).
static StepMetrics measure_host(const std::vector<double>& T, const Domain& dom, double Tm) {
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

static RunResult run_cpu(const lpbf::Material mat, lpbf::Process pr, lpbf::Domain dom, double dt, int n_steps, lpbf::Physics phys, double kmult) {
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


#endif //HEDGEHOG_LPBF_CPU_KERNELS_H
