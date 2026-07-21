#include "../common.h"
#include "kokkos_kernels.h"
#include "tasks.h"

#include <stdfloat>

using namespace std::string_literals;

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

    auto sps = [&](const double ms){ return ms>0 ? n_steps*1000.0/ms : 0.0; };

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
        expY(j+1) = Kokkos::exp(Float{-2} * square(yPos - y_laser) * stepConst.inv_r0_sq);
    });
    executionSpace.fence();

    if(cli.hedgehog) {
        using ScalarFieldView = Kokkos::View<Float**, MemorySpace>;

        const auto pool        = AutoReleaseMemoryPool<ScalarFieldView>::create();
        pool->fill(cli.n_threads, "field", NX, NY);
        const auto fdmTask     = std::make_shared<lpbf::FdmTask<Float, ExecutionSpace>>(pool);
        const auto reducerTask = std::make_shared<lpbf::ReducerTask<ExecutionSpace, ScalarFieldView>>(mat.Tm, cli.n_threads);
        const auto fmdSM       = std::make_shared<lpbf::FdmStateTask<Float>>(n_steps);
        
        using InputData = lpbf::InputData<Float, MemorySpace>;
        auto graph = hh::Graph<1, InputData, StepMetrics<Float>>("LPBF");
        graph.input<InputData>(fdmTask);
        graph.edge<ScalarFieldView>(fdmTask, reducerTask);
        graph.output<StepMetrics<Float>>(reducerTask);
        graph.executeGraph();
        
        auto start = std::chrono::steady_clock::now();
        graph.pushData(std::make_shared<InputData>(
            expX,
            expY,
            currentT,
            nextT,
            dt,
            dom,
            pr,
            stepConst,
            x_laser,
            y_laser,
            x_laser,
            y_laser,
            phys,
            n_steps
        ));
        graph.finishPushingData();
        Float peakT = -1;
        Float peakW = -1;
        while(auto result = graph.getBlockingResult()) {
            auto stepMetric = std::get<std::shared_ptr<StepMetrics<Float>>>(*result);
            peakT = std::max(peakT, stepMetric->peak_T);
            peakW = std::max(peakW, stepMetric->W_um);
        }
        graph.waitForTermination();
        const auto time = toMilliSeconds(std::chrono::steady_clock::now() - start);
        std::printf("[Hedgehog][Time %.3fms][Steps/s %.3f][Peak Temp %.3fK][Peak Width %.3fum]\n", time, sps(time), peakT, peakW);
        const auto dotFile = "lbpf_"s + (phys == Physics::Baseline? "base": "ext") + ".dot";
        graph.createDotFile(dotFile, hh::ColorScheme::EXECUTION, hh::StructureOptions::NONE);

        return 0;
    }

    const auto wall0 = std::chrono::steady_clock::now();
    for(int s = 0; s < n_steps; ++s) {
        stepConst.x_l = x_laser;
        stepConst.y_l = y_laser;

        Kokkos::parallel_for("expXTable", Kokkos::RangePolicy(executionSpace, 0, NX-2), KOKKOS_LAMBDA(const int32_t i) {
            const auto xPos = static_cast<Float>(i) * stepConst.dx;
            expX(i+1) = Kokkos::exp(Float{-2} * square(xPos - x_laser) * stepConst.inv_r0_sq);
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
                    const auto kxm = Float{0.5} * (kCe + effectiveThermalConductivity(Txm, stepConst.kmult));
                    const auto kxp = Float{0.5} * (kCe + effectiveThermalConductivity(Txp, stepConst.kmult));
                    const auto kym = Float{0.5} * (kCe + effectiveThermalConductivity(Tym, stepConst.kmult));
                    const auto kyp = Float{0.5} * (kCe + effectiveThermalConductivity(Typ, stepConst.kmult));
                    const auto div_k_grad = Float{0}
                        + (kxp*(Txp-T) - kxm*(T-Txm)) * stepConst.inv_dx2
                        + (kyp*(Typ-T) - kym*(T-Tym)) * stepConst.inv_dy2;
                    const auto q_in = stepConst.q_scale * stepConst.peak_S * expX(i) * expY(j);

                    auto q_rad = Float{0};
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
                    const auto lap_x = (Txp - Float{2}*T + Txm) * stepConst.inv_dx2;
                    const auto lap_y = (Typ - Float{2}*T + Tym) * stepConst.inv_dy2;
                    const auto S     = stepConst.q_scale * stepConst.peak_S * expX(i) * expY(j) / (stepConst.rho_cp * stepConst.h);
                    auto       q_rad = Float{0};
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


    std::printf("[Time %.3fms][Steps/s %.3f][Peak Temp %.3f][Peak Width %.3f]\n", wall_ms, sps(wall_ms), static_cast<double>(peak_T), static_cast<double>(peak_W));

    return 0;
}
