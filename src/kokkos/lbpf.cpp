#include "kokkos_kernels.h"

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

    // ---- --bench -------------------------------------------------------
    if (cli.do_bench) {
        std::cout << "\n[--bench] " << n_steps << " steps each path...\n";
        const auto result = runKokkos(mat, pr, dom, dt, n_steps, phys, cli.kmult);
        auto sps = [&](const double ms){ return ms>0 ? n_steps*1000.0/ms : 0.0; };
        std::cout << "  Kokkos  wall " << result.wall_ms << " ms   " << sps(result.wall_ms)
                  << " steps/s   peak_T=" << result.peak_T << "    peak_W=" << result.peak_W << "\n";
    }

    return 0;
}
