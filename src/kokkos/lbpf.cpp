#include "../common.h"
#include "kokkos_kernels.h"

int main(const int argc, char *argv[]) {
    using Float          = float;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace    = ExecutionSpace::memory_space;

    const auto kokkosSG = Kokkos::ScopeGuard();

    const auto     cli = parse_cli<Float>(argc, argv);
    auto           mat = Material<Float>();
    auto           pr  = Process<Float>();
    constexpr auto dom = Domain<Float>();

    apply_case(cli.case_id, pr);
    if (cli.P_ovr     > 0)  pr.P     = cli.P_ovr;
    if (cli.V_ovr     > 0)  pr.V     = cli.V_ovr;
    if (cli.r0_ovr    > 0)  pr.r0    = cli.r0_ovr;
    if (cli.eta_ovr   > 0)  mat.eta  = cli.eta_ovr;
    if (cli.dsub_ovr  > 0)  pr.dSub  = cli.dsub_ovr;
    if (cli.hconv_ovr >= 0) pr.hConv = cli.hconv_ovr;

    const Physics phys = (cli.physics == "ext") ? Physics::Extended : Physics::Baseline;
    const Float dt = (phys == Physics::Extended) ? dt_cfl_ext(mat, dom, cli.kmult)
                                                  : dt_cfl(mat, dom);
    const int n_steps = static_cast<int>(cli.t_end / dt);

    std::cout << "kokkos_lpbf\n"
              << "  device    : " << cli.device << "\n"
              << "  case      : " << cli.case_id << "  (P=" << pr.P << " W, V="
              << pr.V*1e3 << " mm/s, r0=" << pr.r0*1e6 << " um)\n"
              << "  physics   : " << (phys==Physics::Extended?"extended":"baseline") << "\n"
              << "  grid      : " << dom.nx << " x " << dom.ny << "\n"
              << "  dt        : " << dt*1e6 << " us   n_steps=" << n_steps << "\n";


    constexpr auto NX = dom.nx + 2;
    constexpr auto NY = dom.ny + 2;

    auto currentT = Kokkos::View<Float**, MemorySpace>("TemperatureField", NX, NY);
    auto nextT    = Kokkos::View<Float**, MemorySpace>("TemperatureField", NX, NY);

    const auto expX = Kokkos::View<Float*, MemorySpace>("ExpX", NX);
    const auto expY = Kokkos::View<Float*, MemorySpace>("ExpY", NY);

    auto stepConst  = makeConst(mat, pr, dom, dt);
    stepConst.kmult = cli.kmult;

    constexpr auto x_start = 0.20*dom.Lx;
    auto           x_laser = x_start;
    constexpr auto y_laser = 0.50*dom.Ly;
    auto           peak_T  = pr.Tamb, peak_W = Float{0}, steady_W_sub = Float{0};
    bool           wrapped = false;

    const auto executionSpace = createExecutionSpace<ExecutionSpace>(0);
    Kokkos::deep_copy(executionSpace, currentT, pr.Tamb);
    Kokkos::deep_copy(executionSpace, nextT, pr.Tamb);
    Kokkos::parallel_for("expYTable", Kokkos::RangePolicy(executionSpace, 0, NY-2), KOKKOS_LAMBDA(const int j) {
        const auto yPos = static_cast<Float>(j) * stepConst.dy;
        expY(j+1) = Kokkos::exp(-2.0 * square(yPos - y_laser) * stepConst.inv_r0_sq);
    });
    executionSpace.fence();

    const auto wall0 = std::chrono::steady_clock::now();
    for(int s = 0; s < n_steps; ++s) {
        stepConst.x_l = x_laser;
        stepConst.y_l = y_laser;

        Kokkos::parallel_for("expXTable", Kokkos::RangePolicy(executionSpace, 0, NX-2), KOKKOS_LAMBDA(const int i) {
            const auto xPos = static_cast<Float>(i) * stepConst.dx;
            expX(i+1) = Kokkos::exp(-2.0 * square(xPos - x_laser) * stepConst.inv_r0_sq);
        });

        const auto inpT = Kokkos::View<const Float**, MemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>(currentT);
        const auto outT = nextT;

        switch(phys) {
            case Physics::Extended:
                Kokkos::parallel_for("FDM.2", Kokkos::MDRangePolicy(executionSpace, {1, 1}, {NX-1, NY-1}), KOKKOS_LAMBDA(const int32_t i, const int32_t j) {
                    const auto T   = inpT(i,   j);
                    const auto Txm = inpT(i-1, j);
                    const auto Txp = inpT(i+1, j);
                    const auto Tym = inpT(i,   j-1);
                    const auto Typ = inpT(i,   j+1);
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
                    const auto T   = inpT(i,   j);
                    const auto Txm = inpT(i-1, j);
                    const auto Txp = inpT(i+1, j);
                    const auto Tym = inpT(i,   j-1);
                    const auto Typ = inpT(i,   j+1);
                    const auto lap_x = (Txp - 2.0*T + Txm) * stepConst.inv_dx2;
                    const auto lap_y = (Typ - 2.0*T + Tym) * stepConst.inv_dy2;
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

        const auto stepMetric = measure(executionSpace, outT, dom, mat.Tm);

        peak_T = std::max(peak_T, stepMetric.peak_T);
        peak_W = std::max(peak_W, stepMetric.W_um);
        std::swap(currentT, nextT);

        x_laser += pr.V * dt;
        if(0.90*dom.Lx < x_laser) {
            x_laser = x_start;
            wrapped = true;
        }

        if(!wrapped and x_laser <= 0.85*dom.Lx and 0.45*dom.Lx <= x_laser) {
            steady_W_sub = std::max(steady_W_sub, stepMetric.W_sub_um);
        }
    }
    const auto wall_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - wall0).count());


    std::cout << "\n[--bench] " << n_steps << " steps each path...\n";
    auto sps = [&](const double ms){ return ms>0 ? n_steps*1000.0/ms : 0.0; };
    std::cout << "  Kokkos  wall " << wall_ms << " ms   " << sps(wall_ms)
              << " steps/s   peak_T=" << peak_T << "    peak_W=" << peak_W << "\n";

    return 0;
}
