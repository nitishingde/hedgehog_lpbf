#ifndef HEDGEHOG_LPBF_KOKKOS_KERNELS_H
#define HEDGEHOG_LPBF_KOKKOS_KERNELS_H

#include "utility.h"

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

template<std::floating_point Float, int32_t N = 9>
requires (N == 9)
KOKKOS_INLINE_FUNCTION Float interp(const Float T, const Kokkos::Array<Float, N> &xs, const Kokkos::Array<Float, N> &ys) {
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

template<std::floating_point Float>
KOKKOS_INLINE_FUNCTION Float thermalConductivity(const Float T) {
    constexpr auto TEMPERATURE          = Kokkos::Array<Float, 9>{298,  473,  673,  873, 1073, 1273, 1473, 1533, 1609};
    constexpr auto THERMAL_CONDUCTIVITY = Kokkos::Array<Float, 9>{8.9, 12.9, 15.8, 18.7, 21.9, 25.6, 28.6, 29.3, 29.6};
    return interp<Float, 9>(T, TEMPERATURE, THERMAL_CONDUCTIVITY);
}

template<std::floating_point Float>
KOKKOS_INLINE_FUNCTION Float specificHeatCapacity(const Float T) {
    constexpr auto TEMPERATURE            = Kokkos::Array<Float, 9>{298,  473,  673,  873, 1073, 1273, 1473, 1533, 1609};
    constexpr auto SPECIFIC_HEAT_CAPACITY = Kokkos::Array<Float, 9>{435,  479,  515,  558,  580,  628,  670,  685,  720};
    return interp<Float, 9>(T, TEMPERATURE, SPECIFIC_HEAT_CAPACITY);
}

// params.hpp:93-100  Marangoni conductivity enhancement (kmult<=1 -> no-op).
template<std::floating_point Float>
KOKKOS_INLINE_FUNCTION Float thermalConductivityEnhancement(const Float T, const Float kmult) {
    const auto m = Kokkos::max(kmult - Float{1}, Float{0});
    Float s = (T - T_SOL()) / (T_LIQ() - T_SOL());
    s = Kokkos::max(Float{0}, Kokkos::min(s, Float{1}));
    return Float{1} + m * s;
}

template<std::floating_point Float>
KOKKOS_INLINE_FUNCTION Float effectiveThermalConductivity(const Float T, const Float kmult) {
    return thermalConductivity(T) * thermalConductivityEnhancement(T, kmult);
}

// params.hpp:107-111  apparent-cp latent heat over the mushy interval.
template<std::floating_point Float>
KOKKOS_INLINE_FUNCTION Float effectiveSpecificHeatCapacity(const Float T) {
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
template<std::floating_point Float>
struct ThermalReduction {
    Float   Tmax;
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

template<std::floating_point Float>
struct ThermalReducer {
    using reducer = ThermalReducer;
    using value_type = ThermalReduction<Float>;
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
template<std::floating_point Float, KokkosView Field = Kokkos::View<Float**>, KokkosExecutionSpace ExecutionSpace = Kokkos::Serial>
static StepMetrics<Float> measure(const ExecutionSpace &executionSpace, const Field &tempField, const Domain<Float> &dom, Float Tm) {
    auto results = ThermalReduction<Float>{};

    const auto NX = tempField.extent(0);
    const auto NY = tempField.extent(1);
    Kokkos::parallel_reduce("ThermalAnalysis", Kokkos::MDRangePolicy(executionSpace, {0, 0}, {NX, NY}),
        KOKKOS_LAMBDA(const int32_t x, const int32_t y, ThermalReduction<Float> &local) {
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

    return StepMetrics<Float> {
        .peak_T = results.Tmax,
        .W_um = (results.y_max < 0 or results.y_min == 1e9) ? Float{0} : Float((results.y_max - results.y_min + 1) * dom.dy() * 1e6),
        .W_sub_um = 0,//TODO
        .x_hot = results.x_hot
    };
}

#endif //HEDGEHOG_LPBF_KOKKOS_KERNELS_H
