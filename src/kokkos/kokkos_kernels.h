#ifndef HEDGEHOG_LPBF_KOKKOS_KERNELS_H
#define HEDGEHOG_LPBF_KOKKOS_KERNELS_H

#include "utility.h"
#include "lpbf_step_const.hpp"

#include <Kokkos_Core.hpp>

template<typename T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto square(const T val) {
    return val*val;
}

template<typename T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto quad(const T val) {
    return val*val*val*val;
}
consteval double sigma_sb() { return 5.670374419e-8; }
consteval double T_SOL()    { return 1533.0;         }  // K
consteval double T_LIQ()    { return 1609.0;         }  // K
consteval double LF()       { return 210.0e3;        } // J/kg

template<int32_t N = 9>
KOKKOS_INLINE_FUNCTION double interp(const double T, const Kokkos::Array<double, N> &xs, const Kokkos::Array<double, N> &ys) {
    if(T <= xs[0])     return ys[0];
    if(T >= xs[N - 1]) return ys[N - 1];

    int i = 1;
    while (T > xs[i]) ++i;
    const double f = (T - xs[i - 1]) / (xs[i] - xs[i - 1]);
    return ys[i - 1] + f * (ys[i] - ys[i - 1]);
}

KOKKOS_INLINE_FUNCTION double thermalConductivity(const double T) {
    constexpr auto TEMPERATURE          = Kokkos::Array<double, 9>{298,  473,  673,  873, 1073, 1273, 1473, 1533, 1609};
    constexpr auto THERMAL_CONDUCTIVITY = Kokkos::Array<double, 9>{8.9, 12.9, 15.8, 18.7, 21.9, 25.6, 28.6, 29.3, 29.6};
    return interp<9>(T, TEMPERATURE, THERMAL_CONDUCTIVITY);
}

KOKKOS_INLINE_FUNCTION double specificHeatCapacity(const double T) {
    constexpr auto TEMPERATURE            = Kokkos::Array<double, 9>{298,  473,  673,  873, 1073, 1273, 1473, 1533, 1609};
    constexpr auto SPECIFIC_HEAT_CAPACITY = Kokkos::Array<double, 9>{435,  479,  515,  558,  580,  628,  670,  685,  720};
    return interp<9>(T, TEMPERATURE, SPECIFIC_HEAT_CAPACITY);
}

// params.hpp:93-100  Marangoni conductivity enhancement (kmult<=1 -> no-op).
KOKKOS_INLINE_FUNCTION double thermalConductivityEnhancement(const double T, const double kmult) {
    if (kmult <= 1.0) return 1.0;
    double s;
    if      (T <= T_SOL()) s = 0.0;
    else if (T >= T_LIQ()) s = 1.0;
    else                   s = (T - T_SOL()) / (T_LIQ() - T_SOL());
    return 1.0 + (kmult - 1.0) * s;
}

KOKKOS_INLINE_FUNCTION double effectiveThermalConductivity(const double T, const double kmult) {
    return thermalConductivity(T) * thermalConductivityEnhancement(T, kmult);
}

// params.hpp:107-111  apparent-cp latent heat over the mushy interval.
KOKKOS_INLINE_FUNCTION double effectiveSpecificHeatCapacity(const double T) {
    const int32_t factor = (T_SOL() <= T and T <= T_LIQ());
    return specificHeatCapacity(T) + factor*(LF()/(T_LIQ()-T_SOL()));
}

template<class ExecutionSpace = Kokkos::Serial, class ScalarField = Kokkos::View<double**>, class KokkosView = Kokkos::View<double*>>
void fdm_boundary(const ExecutionSpace &executionSpace, const ScalarField &field) {
    const auto NX = field.extent(0);
    const auto NY = field.extent(1);
    Kokkos::parallel_for("BND_VERTICAL", Kokkos::RangePolicy(executionSpace, 0, NY), KOKKOS_LAMBDA(const int32_t j) {
        field(   0, j) = field(   1, j);
        field(NX-1, j) = field(NX-2, j);
    });

    Kokkos::parallel_for("BND_HORIZONTAL", Kokkos::RangePolicy(executionSpace, 0, NX), KOKKOS_LAMBDA(const int32_t i) {
        const int32_t isLeft  = (i == 0);
        const int32_t isRight = (i == (NX - 1));
        const auto    ri      = isLeft*1 + isRight*(NX-2) + (1-isLeft)*(1-isRight)*i;

        field(i, 0)    = field(ri, 1);
        field(i, NY-1) = field(ri, NY-2);
    });
}

// 1. Custom struct to hold all 2D reduction variables
struct ThermalReduction {
    double  Tmax;
    int32_t x_hot;
    int32_t y_min;
    int32_t y_max;

    KOKKOS_INLINE_FUNCTION ThermalReduction() {
        Tmax  = 0.0;
        x_hot = 0;
        y_min = INT32_MAX;
        y_max = INT32_MIN;
    }
};

struct ThermalReducer {
    using reducer = ThermalReducer;
    using value_type = ThermalReduction;
    using result_view_type = Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    KOKKOS_INLINE_FUNCTION explicit ThermalReducer(value_type& val) : value(val) {}

    KOKKOS_INLINE_FUNCTION static void join(value_type& dest, const value_type& src) {
        if (src.Tmax > dest.Tmax) {
            dest.Tmax = src.Tmax;
            dest.x_hot = src.x_hot;
        }
        dest.y_min = Kokkos::min(dest.y_min, src.y_min);
        dest.y_max = Kokkos::max(dest.y_max, src.y_max);
    }

    static KOKKOS_INLINE_FUNCTION void init(value_type& val) {
        val = value_type();
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION value_type& reference() const {
        return value;
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION result_view_type view() const {
        return result_view_type(&value);
    }

private:
    value_type& value;
};

// Host measure() — identical to src/main.cpp:164-211 (used by CPU path).
template<KokkosView Field = Kokkos::View<double**>, KokkosExecutionSpace ExecutionSpace = Kokkos::Serial>
static StepMetrics measure(const Field &tempField, const Domain& dom, double Tm, const ExecutionSpace &executionSpace = ExecutionSpace()) {
    ThermalReduction results;

    const auto NX = tempField.extent(0);
    const auto NY = tempField.extent(1);
    Kokkos::parallel_reduce("ThermalAnalysis", Kokkos::MDRangePolicy({0, 0}, {NX, NY}),
        KOKKOS_LAMBDA(const int x, const int y, ThermalReduction& local) {
            const auto temperature = tempField(x, y);

            if(local.Tmax < temperature) {
                local.Tmax  = temperature;
                local.x_hot = x;
            }
            if(Tm < temperature) {
                local.y_min = Kokkos::min(local.y_min, y);
                local.y_max = Kokkos::max(local.y_max, y);
            }
        },
        ThermalReducer(results)
    );

    return StepMetrics {
        .peak_T = results.Tmax,
        .W_um = (results.y_max < 0 or results.y_min == 1e9) ? 0.0 : (results.y_max - results.y_min + 1) * dom.dy() * 1e6,
        .W_sub_um = 0,//TODO
        .x_hot = results.x_hot
    };
}

[[nodiscard]] static RunResult runKokkos(const lpbf::Material &mat, lpbf::Process pr, lpbf::Domain dom, double dt, const int32_t n_steps, lpbf::Physics phys, const double kmult) {
    using MemorySpace = Kokkos::SharedSpace;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;

    const auto NX = dom.nx + 2;
    const auto NY = dom.ny + 2;

    auto currentT = Kokkos::View<double**, MemorySpace>("TemperatureField", NX, NY);
    auto nextT    = Kokkos::View<double**, MemorySpace>("TemperatureField", NX, NY);

    const auto expX = Kokkos::View<double*, MemorySpace>("ExpX", NX);
    const auto expY = Kokkos::View<double*, MemorySpace>("ExpY", NY);

    auto stepConst = make_const(mat, pr, dom, dt);
    stepConst.kmult = kmult;

    double x_laser       = 0.20*dom.Lx, y_laser = 0.50*dom.Ly;
    const double x_start = x_laser;
    double peak_T        = pr.Tamb, peak_W = 0.0, steady_W_sub = 0.0;
    bool wrapped         = false;

    const auto executionSpace = createExecutionSpace<ExecutionSpace>(0);
    Kokkos::parallel_for("expYTable", Kokkos::RangePolicy(executionSpace, 0, NY-2), KOKKOS_LAMBDA(const int j) {
        const auto yPos = static_cast<double>(j) * stepConst.dy;
        expY(j+1) = Kokkos::exp(-2.0 * square(yPos - y_laser) * stepConst.inv_r0_sq);
    });
    executionSpace.fence();

    const auto wall0 = std::chrono::steady_clock::now();
    for(int s = 0; s < n_steps; ++s) {
        stepConst.x_l = x_laser;
        stepConst.y_l = y_laser;

        Kokkos::parallel_for("expXTable", Kokkos::RangePolicy(executionSpace, 0, NX-2), KOKKOS_LAMBDA(const int i) {
            const auto xPos = static_cast<double>(i) * stepConst.dx;
            expX(i+1) = Kokkos::exp(-2.0 * square(xPos - x_laser) * stepConst.inv_r0_sq);
        });
        // Kokkos::parallel_for("expYTable", Kokkos::RangePolicy(executionSpace, 0, NY-2), KOKKOS_LAMBDA(const int j) {
        //     const auto yPos = static_cast<double>(j) * stepConst.dy;
        //     expY(j+1) = Kokkos::exp(-2.0 * square(yPos - y_laser) * stepConst.inv_r0_sq);
        // });

        const auto inpT = Kokkos::View<const double**, MemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>(currentT);
        const auto outT = nextT;

        switch(phys) {
            case Physics::Extended:
                Kokkos::parallel_for("FDM.2", Kokkos::MDRangePolicy(executionSpace, {1, 1}, {NX-1, NY-1}), KOKKOS_LAMBDA(const int32_t i, const int32_t j) {
                    const auto T   = inpT(i,   j);
                    const auto Txm = inpT(i-1, j);
                    const auto Txp = inpT(i+1, j);
                    const auto Tym = inpT(i, j-1);
                    const auto Typ = inpT(i, j+1);
                    const auto kCe = effectiveThermalConductivity(T, stepConst.kmult);
                    const auto kxm = 0.5 * (kCe + effectiveThermalConductivity(Txm, stepConst.kmult));
                    const auto kxp = 0.5 * (kCe + effectiveThermalConductivity(Txp, stepConst.kmult));
                    const auto kym = 0.5 * (kCe + effectiveThermalConductivity(Tym, stepConst.kmult));
                    const auto kyp = 0.5 * (kCe + effectiveThermalConductivity(Typ, stepConst.kmult));
                    const auto div_k_grad = 0.0
                        + (kxp*(Txp-T) - kxm*(T-Txm)) * stepConst.inv_dx2
                        + (kyp*(Typ-T) - kym*(T-Tym)) * stepConst.inv_dy2;
                    const auto q_in = stepConst.q_scale * stepConst.peak_S * expX(i) * expY(j);

                    auto q_rad = 0.0;
                    if(stepConst.radiation) {
                        q_rad = stepConst.eps * sigma_sb() * (quad(T) - quad(stepConst.Tamb));
                    }
                    const auto q_top = stepConst.hConv * (T - stepConst.Tamb) + q_rad;
                    const auto q_bot = (thermalConductivity(T) / stepConst.dSub) * (T - stepConst.Tsub);
                    const auto rhs   = div_k_grad + (q_in - q_top - q_bot) * stepConst.inv_h;

                    outT(i, j) = T + stepConst.dt * rhs / (stepConst.rho * effectiveSpecificHeatCapacity(T));
                });
                break;

            default:
            case Physics::Baseline:
                Kokkos::parallel_for("FDM.1", Kokkos::MDRangePolicy(executionSpace, {1, 1}, {NX-1, NY-1}), KOKKOS_LAMBDA(const int32_t i, const int32_t j) {
                    const auto T     = inpT(i, j);
                    const auto lap_x = (inpT(i+1, j) - 2.0*T + inpT(i-1, j)) * stepConst.inv_dx2;
                    const auto lap_y = (inpT(i, j+1) - 2.0*T + inpT(i, j-1)) * stepConst.inv_dy2;
                    const auto S     = stepConst.q_scale * stepConst.peak_S * expX(i) * expY(j) / (stepConst.rho_cp * stepConst.h);
                    auto       q_rad = 0.0;
                    if(stepConst.radiation) {
                        q_rad = stepConst.eps * sigma_sb() * (quad(T) - quad(stepConst.Tamb));
                    }
                    const auto q_top = stepConst.hConv * (T - stepConst.Tamb) + q_rad;
                    const auto q_bot = (stepConst.K / stepConst.dSub) * (T - stepConst.Tsub);
                    const auto loss  = (q_top + q_bot) * stepConst.inv_h / stepConst.rho_cp;
                    const auto dTdt  = stepConst.alpha * (lap_x + lap_y) + S - loss;

                    outT(i, j) = T + stepConst.dt * dTdt;
                });
                break;
        }
        fdm_boundary(executionSpace, outT);
        executionSpace.fence();
        std::swap(currentT, nextT);

        x_laser += pr.V * dt;
        if(0.90*dom.Lx < x_laser) {
            x_laser = x_start;
            wrapped = true;
        }

        const auto stepMetric = measure(inpT, dom, mat.Tm);
        peak_T = std::max(peak_T, stepMetric.peak_T);
        peak_W = std::max(peak_W, stepMetric.W_um);
        if(!wrapped and x_laser <= 0.85*dom.Lx and 0.45*dom.Lx <= x_laser) {
            steady_W_sub = std::max(steady_W_sub, stepMetric.W_sub_um);
        }
    }
    const auto wall_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - wall0).count());

    return RunResult {
        .peak_T = peak_T,
        .peak_W = peak_W,
        .steady_W_sub=steady_W_sub,//TODO
        .n_steps=n_steps,
        .wall_ms=wall_ms,
        .Tfinal={}//TODO
    };
}

#endif //HEDGEHOG_LPBF_KOKKOS_KERNELS_H
