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
consteval double T_SOL()    { return 1533.0;         } // K
consteval double T_LIQ()    { return 1609.0;         } // K
consteval double LF()       { return 210.0e3;        } // J/kg

template<int32_t N = 9>
requires (N == 9)
KOKKOS_INLINE_FUNCTION double interp(const double T, const Kokkos::Array<double, N> &xs, const Kokkos::Array<double, N> &ys) {
    if(T <= xs[0])     return ys[0];
    if(T >= xs[N - 1]) return ys[N - 1];

    int32_t idx = 1;
    idx += (T > xs[1]);
    idx += (T > xs[2]);
    idx += (T > xs[3]);
    idx += (T > xs[4]);
    idx += (T > xs[5]);
    idx += (T > xs[6]);
    idx += (T > xs[7]);

    const auto x0 = xs[idx - 1];
    const auto x1 = xs[idx];
    const auto y0 = ys[idx - 1];
    const auto y1 = ys[idx];

    const auto f = (T - x0) / (x1 - x0);
    return y0 + f * (y1 - y0);
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
    const auto m = Kokkos::max(kmult - 1.0, 0.0);
    auto s = (T - T_SOL()) / (T_LIQ() - T_SOL());
    s =  Kokkos::max(0.0, Kokkos::min(s, 1.0));
    return 1.0 + m * s;
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
        if(src.Tmax > dest.Tmax) {
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
static StepMetrics measure(const ExecutionSpace &executionSpace, const Field &tempField, const Domain& dom, double Tm) {
    ThermalReduction results;

    const auto NX = tempField.extent(0);
    const auto NY = tempField.extent(1);
    Kokkos::parallel_reduce("ThermalAnalysis", Kokkos::MDRangePolicy(executionSpace, {0, 0}, {NX, NY}),
        KOKKOS_LAMBDA(const int32_t x, const int32_t y, ThermalReduction& local) {
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
    executionSpace.fence();

    return StepMetrics {
        .peak_T = results.Tmax,
        .W_um = (results.y_max < 0 or results.y_min == 1e9) ? 0.0 : (results.y_max - results.y_min + 1) * dom.dy() * 1e6,
        .W_sub_um = 0,//TODO
        .x_hot = results.x_hot
    };
}

#endif //HEDGEHOG_LPBF_KOKKOS_KERNELS_H
