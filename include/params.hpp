// SPDX-License-Identifier: 0BSD
// IN718 material constants and process / domain parameters.
#pragma once
#include <algorithm>
#include <cmath>

namespace lpbf {

enum class Physics { Baseline, Extended };

inline constexpr double SIGMA_SB = 5.670374419e-8;  // Stefan-Boltzmann

struct Material {
    double k    = 25.0;     // W/m/K
    double rho  = 8190.0;   // kg/m^3
    double cp   = 565.0;    // J/kg/K
    double Tm   = 1609.0;   // K   (solidus/liquidus average)
    double eps  = 0.40;     // emissivity
    double eta  = 0.35;     // absorptivity at 1070 nm
    constexpr double alpha() const noexcept { return k / (rho * cp); }
};

struct Process {
    double P     = 285.0;      // W
    double V     = 0.96;       // m/s
    double r0    = 33.5e-6;    // m   (1/e^2 radius)
    double h     = 40.0e-6;    // m   (layer thickness)
    double hConv = 100.0;      // W/m^2/K
    double dSub  = 30.0e-6;    // m
    double Tamb  = 298.15;     // K
    double Tsub  = 298.15;     // K
    bool   radiation = true;
};

struct Domain {
    double Lx = 1.6e-3, Ly = 0.6e-3;
    int    nx = 240, ny = 120;
    constexpr double dx() const noexcept { return Lx / (nx - 1); }
    constexpr double dy() const noexcept { return Ly / (ny - 1); }
};

// Stable explicit-Euler step from the 2D diffusion CFL with a 0.35 safety
// factor; same expression the Python simulator uses.
inline double dt_cfl(Material const& m, Domain const& d) noexcept {
    const double a = m.alpha();
    return 0.35 / (a * (1.0 / (d.dx() * d.dx()) + 1.0 / (d.dy() * d.dy())));
}

// ---------------------------------------------------------------------------
// Extended physics: temperature-dependent IN718 properties.
//
// Piecewise-linear representation of the recommended values of Mills,
// "Recommended Values of Thermophysical Properties for Selected Commercial
// Alloys" (Woodhead, 2002): solid k rises from 8.9 W/m/K at room
// temperature roughly linearly to 29.3 at the solidus (liquid 29.6);
// cp rises from 435 to 685 J/kg/K (liquid 720).  The latent heat of
// fusion (210 kJ/kg) is folded into an apparent heat capacity over the
// solidus-liquidus interval [1533, 1609] K.  Density is kept constant
// (varies < 8 % to the liquidus).
// ---------------------------------------------------------------------------
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
inline double dt_cfl_ext(Material const& m, Domain const& d,
                         double kmult = 1.0) noexcept {
    const double a = in718::alpha_eff_max(m.rho, kmult);
    return 0.35 / (a * (1.0 / (d.dx() * d.dx()) + 1.0 / (d.dy() * d.dy())));
}

}  // namespace lpbf
