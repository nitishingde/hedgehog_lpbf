#ifndef HEDGEHOG_LPBF_COMMON_H
#define HEDGEHOG_LPBF_COMMON_H

#include <concepts>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

enum class Physics { Baseline, Extended };

template<std::floating_point Float>
struct Material {
    Float k    = 25.0;     // W/m/K
    Float rho  = 8190.0;   // kg/m^3
    Float cp   = 565.0;    // J/kg/K
    Float Tm   = 1609.0;   // K   (solidus/liquidus average)
    Float eps  = 0.40;     // emissivity
    Float eta  = 0.35;     // absorptivity at 1070 nm
    constexpr Float alpha() const noexcept { return k / (rho * cp); }
};

template<std::floating_point Float>
struct Process {
    Float P     = 285.0;      // W
    Float V     = 0.96;       // m/s
    Float r0    = 33.5e-6;    // m   (1/e^2 radius)
    Float h     = 40.0e-6;    // m   (layer thickness)
    Float hConv = 100.0;      // W/m^2/K
    Float dSub  = 30.0e-6;    // m
    Float Tamb  = 298.15;     // K
    Float Tsub  = 298.15;     // K
    bool  radiation = true;
};

template<std::floating_point Float>
struct Domain {
    Float   Lx = 1.6e-3;
    Float   Ly = 0.6e-3;
    int32_t nx = 240;
    int32_t ny = 120;
    constexpr Float dx() const noexcept { return Lx / (nx - 1); }
    constexpr Float dy() const noexcept { return Ly / (ny - 1); }
};

template<std::floating_point Float>
struct StepConst {
    int32_t nx, ny;
    Float   dx, dy, inv_dx2, inv_dy2, inv_r0_sq;
    Float   alpha, K, rho, cp, rho_cp, inv_h, h, dt;
    Float   peak_S, eta, P;
    Float   hConv, Tamb, Tsub, dSub, eps;
    bool    radiation;
    Float   kmult;
    Float   x_l, y_l;   // laser position (updated every step)
    Float   q_scale;    // single-track: always 1
};


template<std::floating_point Float>
inline StepConst<Float> makeConst(const Material<Float> &material, const Process<Float> &process, const Domain<Float> &domain, const Float dt) {
    auto stepConst = StepConst<Float>{};
    stepConst.nx = domain.nx;  stepConst.ny = domain.ny;
    stepConst.dx = domain.dx(); stepConst.dy = domain.dy();
    stepConst.inv_dx2 = 1.0 / (stepConst.dx * stepConst.dx);
    stepConst.inv_dy2 = 1.0 / (stepConst.dy * stepConst.dy);
    stepConst.inv_r0_sq = 1.0 / (process.r0 * process.r0);
    stepConst.alpha = material.alpha(); stepConst.K = material.k; stepConst.rho = material.rho; stepConst.cp = material.cp;
    stepConst.rho_cp = material.rho * material.cp; stepConst.inv_h = 1.0 / process.h; stepConst.h = process.h; stepConst.dt = dt;
    stepConst.peak_S = (2.0 * material.eta * process.P) / (M_PI * process.r0 * process.r0);  // step_task.hpp:68
    stepConst.eta = material.eta; stepConst.P = process.P;
    stepConst.hConv = process.hConv; stepConst.Tamb = process.Tamb; stepConst.Tsub = process.Tsub; stepConst.dSub = process.dSub;
    stepConst.eps = material.eps; stepConst.radiation = process.radiation;
    stepConst.kmult = 1.0;   // overwritten by the caller with the real kmult
    stepConst.q_scale = 1.0; // single-track driver: beam never jumps (q_scale == 1)
    stepConst.x_l = 0.0; stepConst.y_l = 0.0;
    return stepConst;
}

template<std::floating_point Float>
struct CLI {
    std::string case_id   = "0";
    int32_t     n_threads = 4;        // parsed & ignored on GPU (kept for parity)
    int32_t     ntiles    = 8;        // parsed & ignored on GPU (kept for parity)
    Float       t_end     = 1.5e-3;
    int32_t     snap_every = 100;
    std::string physics   = "base";
    std::string out_dir;
    Float P_ovr = -1, V_ovr = -1, r0_ovr = -1;
    Float eta_ovr = -1, dsub_ovr = -1, hconv_ovr = -1;
    Float kmult = 1.0;
    int32_t device      = 0;
    bool   do_check    = false;
    bool   do_bench    = false;
    Float check_rtol  = 1e-6;
    Float check_atol  = 1e-3;
};

template<std::floating_point Float>
static auto parse_cli(const int argc, char **argv) {
    auto cli = CLI<Float>();
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto starts = [&](char const* p) { return a.rfind(p, 0) == 0; };
        if      (starts("--case="))       cli.case_id    = a.substr(7);
        else if (starts("--threads="))    cli.n_threads  = std::atoi(a.c_str() + 10);
        else if (starts("--tiles="))      cli.ntiles     = std::atoi(a.c_str() +  8);
        else if (starts("--tend="))       cli.t_end      = std::atof(a.c_str() +  7);
        else if (starts("--snap-every=")) cli.snap_every = std::atoi(a.c_str() + 13);
        else if (starts("--physics="))    cli.physics    = a.substr(10);
        else if (starts("--out="))        cli.out_dir    = a.substr(6);
        else if (starts("--P="))          cli.P_ovr      = std::atof(a.c_str() +  4);
        else if (starts("--V="))          cli.V_ovr      = std::atof(a.c_str() +  4);
        else if (starts("--r0="))         cli.r0_ovr     = std::atof(a.c_str() +  5);
        else if (starts("--eta="))        cli.eta_ovr    = std::atof(a.c_str() +  6);
        else if (starts("--dsub="))       cli.dsub_ovr   = std::atof(a.c_str() +  7);
        else if (starts("--hconv="))      cli.hconv_ovr  = std::atof(a.c_str() +  8);
        else if (starts("--kmult="))      cli.kmult      = std::atof(a.c_str() +  8);
        else if (starts("--device="))     cli.device     = std::atoi(a.c_str() +  9);
        else if (starts("--check-rtol=")) cli.check_rtol = std::atof(a.c_str() + 13);
        else if (starts("--check-atol=")) cli.check_atol = std::atof(a.c_str() + 13);
        else if (a == "--check")          cli.do_check   = true;
        else if (a == "--bench")          cli.do_bench   = true;
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
    return cli;
}

template<std::floating_point Float>
static void apply_case(const std::string &id, Process<Float> &pr) {
    if      (id == "0")   { pr.P = 285.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else if (id == "1.1") { pr.P = 285.0; pr.V = 0.96; pr.r0 = 24.5e-6; }
    else if (id == "1.2") { pr.P = 285.0; pr.V = 0.96; pr.r0 = 41.0e-6; }
    else if (id == "2.1") { pr.P = 285.0; pr.V = 1.20; pr.r0 = 33.5e-6; }
    else if (id == "2.2") { pr.P = 285.0; pr.V = 0.80; pr.r0 = 33.5e-6; }
    else if (id == "3.1") { pr.P = 325.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else if (id == "3.2") { pr.P = 245.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else { std::cerr << "unknown case '" << id << "'\n"; std::exit(1); }
}

template<std::floating_point Float>
struct StepMetrics {
    Float peak_T=0;
    Float W_um=0;
    Float W_sub_um=0;
    int32_t x_hot=0;
};

template<std::floating_point Float>
struct RunResult {
    Float peak_T;
    Float peak_W;
    Float steady_W_sub;
    long long n_steps;
    Float wall_ms;
    std::vector<Float> Tfinal;
};

template<std::floating_point Float>
inline Float dt_cfl(Material<Float> const& m, Domain<Float> const& d) noexcept {
    const Float a = m.alpha();
    return 0.35 / (a * (1.0 / (d.dx() * d.dx()) + 1.0 / (d.dy() * d.dy())));
}

namespace in718 {
    inline constexpr double T_SOL = 1533.0;   // K
    inline constexpr double T_LIQ = 1609.0;   // K
    inline constexpr double LF    = 210.0e3;  // J/kg

    inline constexpr int    N_PTS          = 9;
    inline constexpr double T_PTS [N_PTS]  = {  298,  473,  673,  873, 1073,
                                               1273, 1473, 1533, 1609 };
    inline constexpr double K_PTS [N_PTS]  = {  8.9, 12.9, 15.8, 18.7, 21.9,
                                               25.6, 28.6, 29.3, 29.6 };
    inline constexpr double CP_PTS[N_PTS]  = {  435,  479,  515,  558,  580,
                                                628,  670,  685,  720 };

    inline double interp(double T, double const* xs, double const* ys, int n)
        noexcept {
        if (T <= xs[0])     return ys[0];
        if (T >= xs[n - 1]) return ys[n - 1];
        int i = 1;
        while (T > xs[i]) ++i;
        const double f = (T - xs[i - 1]) / (xs[i] - xs[i - 1]);
        return ys[i - 1] + f * (ys[i] - ys[i - 1]);
    }

    inline double k_of (double T) noexcept { return interp(T, T_PTS, K_PTS,  N_PTS); }
    inline double cp_of(double T) noexcept { return interp(T, T_PTS, CP_PTS, N_PTS); }

    // Learned melt-pool conductivity enhancement (Marangoni-convection proxy).
    // The effective in-plane conductivity is the Mills value times a factor that
    // ramps from 1 (solid, below the solidus) to `kmult` (fully molten, above the
    // liquidus).  kmult is the single learned parameter; kmult = 1 recovers the
    // pure-conduction extended physics exactly.
    inline double k_enh(double T, double kmult) noexcept {
        if (kmult <= 1.0) return 1.0;
        double s;
        if      (T <= T_SOL) s = 0.0;
        else if (T >= T_LIQ) s = 1.0;
        else                 s = (T - T_SOL) / (T_LIQ - T_SOL);
        return 1.0 + (kmult - 1.0) * s;
    }
    inline double k_eff(double T, double kmult) noexcept {
        return k_of(T) * k_enh(T, kmult);
    }

    // Apparent heat capacity: cp(T) plus the latent heat spread uniformly
    // over the mushy interval.
    inline double cp_eff(double T) noexcept {
        double c = cp_of(T);
        if (T >= T_SOL && T <= T_LIQ) c += LF / (T_LIQ - T_SOL);
        return c;
    }

    // Largest effective diffusivity, including the conductivity enhancement
    // (scanned on a fine grid since k_eff is no longer piecewise-linear).  cp_eff
    // >= cp, so using cp is conservative for the CFL bound.
    inline double alpha_eff_max(double rho, double kmult) noexcept {
        double m = 0.0;
        for (double T = 298.0; T <= 2200.0; T += 5.0)
            m = std::max(m, k_eff(T, kmult) / (rho * cp_of(T)));
        return m;
    }
}  // namespace in718

// CFL-stable step for the extended (variable-property) physics: 0.35 of the
// diffusive limit at the largest effective diffusivity (which grows with the
// conductivity-enhancement factor kmult; kmult = 1 is the unenhanced default).
template<std::floating_point Float>
inline Float dt_cfl_ext(Material<Float> const& m, Domain<Float> const& d, Float kmult = 1.0) noexcept {
    const double a = in718::alpha_eff_max(m.rho, kmult);
    return 0.35 / (a * (1.0 / (d.dx() * d.dx()) + 1.0 / (d.dy() * d.dy())));
}

#endif //HEDGEHOG_LPBF_COMMON_H
