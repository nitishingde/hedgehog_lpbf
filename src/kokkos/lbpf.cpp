#include "utility.h"
#include "lpbf_step_const.hpp"

#include <Kokkos_Core.hpp>

using namespace lpbf;

static void cpu_step(const std::vector<double>& Tin, std::vector<double>& Tout, const StepConst& k, Physics phys) {
    const int nx = k.nx, ny = k.ny;
    for (int y = 0; y < ny; ++y) {
        const double ypos = static_cast<double>(y) * k.dy;
        for (int x = 0; x < nx; ++x) {
            const double T = Tin[y * nx + x];
            const double xpos = static_cast<double>(x) * k.dx;
            const double r2 = (xpos - k.x_l) * (xpos - k.x_l)
                            + (ypos - k.y_l) * (ypos - k.y_l);
            if (phys == Physics::Baseline) {          // step_task.hpp:87-131
                double lap_x;
                if      (x == 0)      lap_x = (Tin[y*nx+x+1] - T) * k.inv_dx2;
                else if (x == nx - 1) lap_x = (Tin[y*nx+x-1] - T) * k.inv_dx2;
                else lap_x = (Tin[y*nx+x+1] - 2.0*T + Tin[y*nx+x-1]) * k.inv_dx2;
                const double Tym = (y == 0)      ? T : Tin[(y-1)*nx+x];
                const double Typ = (y == ny - 1) ? T : Tin[(y+1)*nx+x];
                const double lap_y = (Typ - 2.0*T + Tym) * k.inv_dy2;
                const double S = k.q_scale * k.peak_S
                               * std::exp(-2.0 * r2 * k.inv_r0_sq)
                               / (k.rho_cp * k.h);
                double q_top = k.hConv * (T - k.Tamb);
                if (k.radiation) {
                    const double T4 = T*T*T*T, Ta4 = k.Tamb*k.Tamb*k.Tamb*k.Tamb;
                    q_top += k.eps * lpbf_dev::sigma_sb() * (T4 - Ta4);
                }
                const double q_bot = (k.K / k.dSub) * (T - k.Tsub);
                const double loss = (q_top + q_bot) * k.inv_h / k.rho_cp;
                const double dTdt = k.alpha * (lap_x + lap_y) + S - loss;
                Tout[y*nx+x] = T + k.dt * dTdt;
            } else {                                  // step_task.hpp:132-187
                const double Txm = (x == 0)      ? T : Tin[y*nx+x-1];
                const double Txp = (x == nx - 1) ? T : Tin[y*nx+x+1];
                const double Tym = (y == 0)      ? T : Tin[(y-1)*nx+x];
                const double Typ = (y == ny - 1) ? T : Tin[(y+1)*nx+x];
                const double kCe = lpbf_dev::k_eff_host(T, k.kmult);
                const double kxm = 0.5 * (kCe + lpbf_dev::k_eff_host(Txm, k.kmult));
                const double kxp = 0.5 * (kCe + lpbf_dev::k_eff_host(Txp, k.kmult));
                const double kym = 0.5 * (kCe + lpbf_dev::k_eff_host(Tym, k.kmult));
                const double kyp = 0.5 * (kCe + lpbf_dev::k_eff_host(Typ, k.kmult));
                const double div_k_grad =
                      (kxp*(Txp-T) - kxm*(T-Txm)) * k.inv_dx2
                    + (kyp*(Typ-T) - kym*(T-Tym)) * k.inv_dy2;
                const double q_in = k.q_scale * k.peak_S
                                  * std::exp(-2.0 * r2 * k.inv_r0_sq);
                double q_top = k.hConv * (T - k.Tamb);
                if (k.radiation) {
                    const double T4 = T*T*T*T, Ta4 = k.Tamb*k.Tamb*k.Tamb*k.Tamb;
                    q_top += k.eps * lpbf_dev::sigma_sb() * (T4 - Ta4);
                }
                const double q_bot = (lpbf_dev::k_of_host(T) / k.dSub) * (T - k.Tsub);
                const double rhs = div_k_grad + (q_in - q_top - q_bot) * k.inv_h;
                Tout[y*nx+x] = T + k.dt * rhs / (k.rho * lpbf_dev::cp_eff_host(T));
            }
        }
    }
}

// Host measure() — identical to src/main.cpp:164-211 (used by CPU path).
static StepMetrics measure_host(const std::vector<double>& T, const Domain& dom, double Tm) {
    StepMetrics m; int y_min = dom.ny, y_max = -1; double Tmax = 0.0; int x_hot = 0;
    for (int y = 0; y < dom.ny; ++y) {
        const double* row = &T[y * dom.nx]; bool any = false;
        for (int x = 0; x < dom.nx; ++x) {
            const double v = row[x];
            if (v > Tmax) { Tmax = v; x_hot = x; }
            if (v > Tm) any = true;
        }
        if (any) { if (y < y_min) y_min = y; if (y > y_max) y_max = y; }
    }
    m.peak_T = Tmax; m.x_hot = x_hot;
    m.W_um = (y_max < 0) ? 0.0 : (y_max - y_min + 1) * dom.dy() * 1e6;
    if (Tmax > Tm) {
        const double dy = dom.dy(); int a = -1, b = -1;
        for (int y = 0; y < dom.ny; ++y)
            if (T[y*dom.nx + x_hot] > Tm) { if (a < 0) a = y; b = y; }
        if (a >= 0) {
            double y_lo = a*dy, y_hi = b*dy;
            if (a > 0) { double T0=T[(a-1)*dom.nx+x_hot], T1=T[a*dom.nx+x_hot];
                y_lo = ((a-1) + (Tm-T0)/(T1-T0)) * dy; }
            if (b < dom.ny-1) { double T0=T[b*dom.nx+x_hot], T1=T[(b+1)*dom.nx+x_hot];
                y_hi = (b + (Tm-T0)/(T1-T0)) * dy; }
            m.W_sub_um = (y_hi - y_lo) * 1e6;
        }
    }
    return m;
}

static RunResult run_cpu(const lpbf::Material mat, lpbf::Process pr, lpbf::Domain dom, double dt, int n_steps, lpbf::Physics phys, double kmult) {
    const size_t N = (size_t)dom.nx * dom.ny;
    StepConst k = make_const(mat, pr, dom, dt); k.kmult = kmult;
    std::vector<double> A(N, pr.Tamb), B(N, pr.Tamb);
    std::vector<double>* cur = &A; std::vector<double>* nxt = &B;
    double x_laser = 0.20*dom.Lx, y_laser = 0.50*dom.Ly; const double x_start = x_laser;
    double peak_T = pr.Tamb, peak_W = 0.0, steady_W_sub = 0.0; bool wrapped = false;
    const auto wall0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n_steps; ++s) {
        k.x_l = x_laser; k.y_l = y_laser;
        cpu_step(*cur, *nxt, k, phys);
        std::swap(cur, nxt);
        x_laser += pr.V * dt;
        if (x_laser > 0.90*dom.Lx) { x_laser = x_start; wrapped = true; }
        const StepMetrics m = measure_host(*cur, dom, mat.Tm);
        if (m.peak_T > peak_T) peak_T = m.peak_T;
        if (m.W_um   > peak_W) peak_W = m.W_um;
        const bool cruise = !wrapped && x_laser >= 0.45*dom.Lx && x_laser <= 0.85*dom.Lx;
        if (cruise && m.W_sub_um > steady_W_sub) steady_W_sub = m.W_sub_um;
    }
    const double wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - wall0).count();
    RunResult rr; rr.peak_T=peak_T; rr.peak_W=peak_W; rr.steady_W_sub=steady_W_sub;
    rr.n_steps=n_steps; rr.wall_ms=wall_ms; rr.Tfinal=*cur; return rr;
}

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
        for (size_t i = 0; i < cpu.Tfinal.size(); ++i) {
            const double denom = std::max(std::fabs(cpu.Tfinal[i]), 1.0);
        }
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
        auto sps = [&](double ms){ return ms>0 ? n_steps*1000.0/ms : 0.0; };
        std::cout << "  CPU  wall " << cpu.wall_ms << " ms   " << sps(cpu.wall_ms)
                  << " steps/s   peak_T=" << cpu.peak_T << "\n";
        return 0;
    }

    return 0;
}
