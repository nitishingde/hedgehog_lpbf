#ifndef HEDGEHOG_LPBF_KOKKOS_KERNELS_H
#define HEDGEHOG_LPBF_KOKKOS_KERNELS_H

#include <Kokkos_Core.hpp>

template<typename T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto square(const T val) {
    return val*val;
}

template<typename T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto quad(const T val) {
    return val*val*val*val;
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
                    const auto kCe = lpbf_dev::k_eff_host(T, stepConst.kmult);
                    const auto kxm = 0.5 * (kCe + lpbf_dev::k_eff_host(Txm, stepConst.kmult));
                    const auto kxp = 0.5 * (kCe + lpbf_dev::k_eff_host(Txp, stepConst.kmult));
                    const auto kym = 0.5 * (kCe + lpbf_dev::k_eff_host(Tym, stepConst.kmult));
                    const auto kyp = 0.5 * (kCe + lpbf_dev::k_eff_host(Typ, stepConst.kmult));
                    const auto div_k_grad = 0.0
                        + (kxp*(Txp-T) - kxm*(T-Txm)) * stepConst.inv_dx2
                        + (kyp*(Typ-T) - kym*(T-Tym)) * stepConst.inv_dy2;
                    const auto q_in = stepConst.q_scale * stepConst.peak_S * expX(i) * expY(j);

                    auto q_rad = 0.0;
                    if(stepConst.radiation) {
                        q_rad = stepConst.eps * lpbf_dev::sigma_sb() * (quad(T) - quad(stepConst.Tamb));
                    }
                    const auto q_top = stepConst.hConv * (T - stepConst.Tamb) + q_rad;
                    const auto q_bot = (lpbf_dev::k_of_host(T) / stepConst.dSub) * (T - stepConst.Tsub);
                    const auto rhs   = div_k_grad + (q_in - q_top - q_bot) * stepConst.inv_h;

                    outT(i, j) = T + stepConst.dt * rhs / (stepConst.rho * lpbf_dev::cp_eff_host(T));
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
                        q_rad = stepConst.eps * lpbf_dev::sigma_sb() * (quad(T) - quad(stepConst.Tamb));
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
