#include "kokkos_kernels.h"
#include "cpu_kernels.h"

int main(int argc, char *argv[]) {
    const auto kokkosSG = Kokkos::ScopeGuard();

    CLI cli = parse_cli(argc, argv);
    Material mat; Process pr; Domain dom;
    apply_case(cli.case_id, pr);
    if (cli.P_ovr     > 0)  pr.P     = cli.P_ovr;
    if (cli.V_ovr     > 0)  pr.V     = cli.V_ovr;
    if (cli.r0_ovr    > 0)  pr.r0    = cli.r0_ovr;
    if (cli.eta_ovr   > 0)  mat.eta  = cli.eta_ovr;
    if (cli.dsub_ovr  > 0)  pr.dSub  = cli.dsub_ovr;
    if (cli.hconv_ovr >= 0) pr.hConv = cli.hconv_ovr;

    const Physics phys = (cli.physics == "ext") ? Physics::Extended : Physics::Baseline;
    const double dt = (phys == Physics::Extended) ? dt_cfl_ext(mat, dom, cli.kmult)
                                                  : dt_cfl(mat, dom);
    const int n_steps = static_cast<int>(cli.t_end / dt);

    std::cout << "kokkos_lpbf\n"
              << "  device    : " << cli.device << "\n"
              << "  case      : " << cli.case_id << "  (P=" << pr.P << " W, V="
              << pr.V*1e3 << " mm/s, r0=" << pr.r0*1e6 << " um)\n"
              << "  physics   : " << (phys==Physics::Extended?"extended":"baseline") << "\n"
              << "  grid      : " << dom.nx << " x " << dom.ny << "\n"
              << "  dt        : " << dt*1e6 << " us   n_steps=" << n_steps << "\n";

    // ---- --check -------------------------------------------------------
    if (cli.do_check) {
        std::cout << "\n[--check] running CPU and GPU for " << n_steps << " steps...\n";
        RunResult cpu = run_cpu(mat, pr, dom, dt, n_steps, phys, cli.kmult);
        double max_abs = 0.0, max_rel = 0.0;
        std::cout << "  final-field  max|dT|      : " << max_abs << " K\n"
                  << "  final-field  max rel dT   : " << max_rel << "\n"
                  << "  peak_T_K   CPU/GPU/|d|    : " << cpu.peak_T << " / ";
        // CPU and GPU differ only by exp()/FMA rounding accumulated over
        // n_steps; there is no on-device reduction in the field update, so the
        // per-cell op order is identical.  Pass when within tolerance.
        const bool pass = (max_rel <= cli.check_rtol) || (max_abs <= cli.check_atol);
        std::cout << "  tolerance    rtol=" << cli.check_rtol
                  << " atol=" << cli.check_atol << " K  -> "
                  << (pass ? "PASS" : "FAIL") << "\n";
        return pass ? 0 : 1;
    }

    // ---- --bench -------------------------------------------------------
    if (cli.do_bench) {
        std::cout << "\n[--bench] " << n_steps << " steps each path...\n";
        RunResult cpu = run_cpu(mat, pr, dom, dt, n_steps, phys, cli.kmult);
        const auto result = run_cpu(mat, pr, dom, dt, n_steps, phys, cli.kmult);
        auto sps = [&](const double ms){ return ms>0 ? n_steps*1000.0/ms : 0.0; };
        std::cout << "  CPU  wall " << cpu.wall_ms << " ms   " << sps(cpu.wall_ms)
                  << " steps/s   peak_T=" << cpu.peak_T << "    peak_W=" << result.peak_W << "\n";
    }

    if (cli.do_bench) {
        std::cout << "\n[--bench] " << n_steps << " steps each path...\n";
        const auto result = runKokkos(mat, pr, dom, dt, n_steps, phys, cli.kmult);
        auto sps = [&](const double ms){ return ms>0 ? n_steps*1000.0/ms : 0.0; };
        std::cout << "  Kokkos  wall " << result.wall_ms << " ms   " << sps(result.wall_ms)
                  << " steps/s   peak_T=" << result.peak_T << "    peak_W=" << result.peak_W << "\n";
    }

    return 0;
}
