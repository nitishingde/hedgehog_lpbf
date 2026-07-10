// SPDX-License-Identifier: 0BSD
// LPBF 2D z-averaged heat-equation simulator driven by a Hedgehog
// dataflow graph.  The graph contains a single multi-threaded task that
// advances one horizontal row-strip ("tile") of the field by one
// explicit-Euler step; the driver decomposes the domain into N tiles,
// pushes them all into the graph each step, collects the N updated
// tiles, then refreshes halos before the next step.
//
// Build:    see CMakeLists.txt; needs the hedgehog header library.
// Run:      ./hedgehog_lpbf --case=0 --threads=4 --tiles=8 --tend=1.5e-3
//
// Outputs (in snapshots_case<id>/):
//     T_step_NNNNNN.bin   raw nx*ny double field snapshots
//     metrics.csv         step, t [us], x_laser [um], peak T [K], W [um]

#include <hedgehog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "params.hpp"
#include "tile.hpp"
#include "step_task.hpp"

using namespace lpbf;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct CLI {
    std::string case_id  = "0";
    int         n_threads = 4;
    int         ntiles    = 8;
    double      t_end     = 1.5e-3;   // s
    int         snap_every = 100;     // 0: final-step snapshot only
    std::string physics   = "base";   // base | ext
    std::string out_dir;              // empty: snapshots_case<id>[_ext]
    // Optional overrides (negative = keep case/default value).
    double P_ovr = -1, V_ovr = -1, r0_ovr = -1;
    double eta_ovr = -1, dsub_ovr = -1, hconv_ovr = -1;
    double kmult = 1.0;   // learned melt-pool conductivity enhancement
};

static CLI parse_cli(int argc, char** argv) {
    CLI c;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto starts = [&](char const* p) {
            return a.rfind(p, 0) == 0;
        };
        if      (starts("--case="))      c.case_id    = a.substr(7);
        else if (starts("--threads="))   c.n_threads   = std::atoi(a.c_str() + 10);
        else if (starts("--tiles="))     c.ntiles      = std::atoi(a.c_str() +  8);
        else if (starts("--tend="))      c.t_end       = std::atof(a.c_str() +  7);
        else if (starts("--snap-every=")) c.snap_every = std::atoi(a.c_str() + 13);
        else if (starts("--physics="))   c.physics     = a.substr(10);
        else if (starts("--out="))       c.out_dir     = a.substr(6);
        else if (starts("--P="))         c.P_ovr       = std::atof(a.c_str() +  4);
        else if (starts("--V="))         c.V_ovr       = std::atof(a.c_str() +  4);
        else if (starts("--r0="))        c.r0_ovr      = std::atof(a.c_str() +  5);
        else if (starts("--eta="))       c.eta_ovr     = std::atof(a.c_str() +  6);
        else if (starts("--dsub="))      c.dsub_ovr    = std::atof(a.c_str() +  7);
        else if (starts("--hconv="))     c.hconv_ovr   = std::atof(a.c_str() +  8);
        else if (starts("--kmult=")) c.kmult = std::atof(a.c_str() + 8);
        else if (a == "--help" || a == "-h") {
            std::cout
                << "Usage: hedgehog_lpbf [options]\n"
                   "  --case=ID            AMB2022-03 case: 0, 1.1, 1.2, 2.1,\n"
                   "                       2.2, 3.1, 3.2 (default 0)\n"
                   "  --physics=base|ext   constant props (base) or Mills k(T),\n"
                   "                       cp(T) + latent heat (ext)\n"
                   "  --P=W --V=M/S --r0=M   process overrides (SI units)\n"
                   "  --eta=F --dsub=M --hconv=W/M2K   model-parameter overrides\n"
                   "  --out=DIR            output directory override\n"
                   "  --threads=N          worker threads in the step task\n"
                   "  --tiles=N            row-strip tiles\n"
                   "  --tend=SECS          physical end-time, e.g. 1.5e-3\n"
                   "  --snap-every=N       snapshot stride (0: final only)\n";
            std::exit(0);
        }
    }
    return c;
}

// All seven AMB2022-03 IN718 single-track cases (data/ambench.json).
static void apply_case(std::string const& id, Process& pr) {
    if      (id == "0")   { pr.P = 285.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else if (id == "1.1") { pr.P = 285.0; pr.V = 0.96; pr.r0 = 24.5e-6; }
    else if (id == "1.2") { pr.P = 285.0; pr.V = 0.96; pr.r0 = 41.0e-6; }
    else if (id == "2.1") { pr.P = 285.0; pr.V = 1.20; pr.r0 = 33.5e-6; }
    else if (id == "2.2") { pr.P = 285.0; pr.V = 0.80; pr.r0 = 33.5e-6; }
    else if (id == "3.1") { pr.P = 325.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else if (id == "3.2") { pr.P = 245.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else {
        std::cerr << "unknown case '" << id << "'\n";
        std::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Halo / tile helpers
// ---------------------------------------------------------------------------

static std::pair<std::vector<int>, std::vector<int>>
build_tile_bounds(int ny, int ntiles) {
    std::vector<int> y0(ntiles), y1(ntiles);
    for (int i = 0; i < ntiles; ++i) {
        y0[i] =  (i      * ny) / ntiles;
        y1[i] = ((i + 1) * ny) / ntiles - 1;
    }
    return {y0, y1};
}

static std::shared_ptr<HaloTile>
make_tile(int idx, int y0, int y1, Domain const& dom,
          std::vector<double> const& T_global,
          int step, double t,
          double x_laser, double y_laser)
{
    auto tile     = std::make_shared<HaloTile>();
    tile->idx     = idx;
    tile->y0      = y0;
    tile->y1      = y1;
    tile->nx      = dom.nx;
    tile->step    = step;
    tile->t       = t;
    tile->x_laser = x_laser;
    tile->y_laser = y_laser;

    const int rows = tile->total_rows();
    tile->data.resize(static_cast<std::size_t>(rows) * dom.nx);

    // Mirror at top/bottom domain boundary -> zero-flux y-BC.
    const int top_src = (y0 == 0)            ? y0 : (y0 - 1);
    const int bot_src = (y1 == dom.ny - 1)   ? y1 : (y1 + 1);

    std::copy_n(&T_global[top_src * dom.nx], dom.nx, &tile->at(0, 0));
    for (int r = 0; r < tile->owned_rows(); ++r) {
        std::copy_n(&T_global[(y0 + r) * dom.nx], dom.nx,
                    &tile->at(r + 1, 0));
    }
    std::copy_n(&T_global[bot_src * dom.nx], dom.nx, &tile->at(rows - 1, 0));
    return tile;
}

// Compute peak T and y-extent above T_m on the global field.
struct StepMetrics {
    double peak_T   = 0.0;
    double W_um     = 0.0;   // row-count width (grid-quantised)
    double W_sub_um = 0.0;   // sub-grid width through the hottest column
};

static StepMetrics measure(std::vector<double> const& T,
                           Domain const& dom, double Tm) {
    StepMetrics m;
    int y_min = dom.ny, y_max = -1;
    double Tmax = 0.0;
    int x_hot = 0;
    for (int y = 0; y < dom.ny; ++y) {
        const double* row = &T[y * dom.nx];
        bool any_molten = false;
        for (int x = 0; x < dom.nx; ++x) {
            const double v = row[x];
            if (v > Tmax) { Tmax = v; x_hot = x; }
            if (v > Tm) any_molten = true;
        }
        if (any_molten) {
            if (y < y_min) y_min = y;
            if (y > y_max) y_max = y;
        }
    }
    m.peak_T = Tmax;
    m.W_um   = (y_max < 0) ? 0.0
                           : (y_max - y_min + 1) * dom.dy() * 1e6;

    // Sub-grid width: linear interpolation of the T = Tm crossings of the
    // transverse profile through the hottest column.
    if (Tmax > Tm) {
        const double dy = dom.dy();
        int a = -1, b = -1;
        for (int y = 0; y < dom.ny; ++y) {
            if (T[y * dom.nx + x_hot] > Tm) { if (a < 0) a = y; b = y; }
        }
        if (a >= 0) {
            double y_lo = a * dy, y_hi = b * dy;
            if (a > 0) {
                const double T0 = T[(a - 1) * dom.nx + x_hot];
                const double T1 = T[a       * dom.nx + x_hot];
                y_lo = ((a - 1) + (Tm - T0) / (T1 - T0)) * dy;
            }
            if (b < dom.ny - 1) {
                const double T0 = T[b       * dom.nx + x_hot];
                const double T1 = T[(b + 1) * dom.nx + x_hot];
                y_hi = (b + (Tm - T0) / (T1 - T0)) * dy;
            }
            m.W_sub_um = (y_hi - y_lo) * 1e6;
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    CLI cli = parse_cli(argc, argv);

    Material mat;
    Process  pr;
    Domain   dom;
    apply_case(cli.case_id, pr);
    if (cli.P_ovr     > 0) pr.P     = cli.P_ovr;
    if (cli.V_ovr     > 0) pr.V     = cli.V_ovr;
    if (cli.r0_ovr    > 0) pr.r0    = cli.r0_ovr;
    if (cli.eta_ovr   > 0) mat.eta  = cli.eta_ovr;
    if (cli.dsub_ovr  > 0) pr.dSub  = cli.dsub_ovr;
    if (cli.hconv_ovr >= 0) pr.hConv = cli.hconv_ovr;

    const Physics phys = (cli.physics == "ext") ? Physics::Extended
                                                : Physics::Baseline;
    const double dt      = (phys == Physics::Extended)
                               ? dt_cfl_ext(mat, dom, cli.kmult)
                               : dt_cfl(mat, dom);
    const int    n_steps = static_cast<int>(cli.t_end / dt);
    if (cli.ntiles > dom.ny) cli.ntiles = dom.ny;

    std::cout
        << "Hedgehog LPBF 2D simulator\n"
        << "  case      : " << cli.case_id  << "  (P=" << pr.P << " W, "
                                            "V=" << pr.V * 1e3 << " mm/s, "
                                            "r0=" << pr.r0 * 1e6 << " um)\n"
        << "  physics   : " << (phys == Physics::Extended
                                ? "extended (Mills k(T), cp(T) + latent heat)"
                                : "baseline (constant properties)") << "\n"
        << "  eta/dsub/hconv : " << mat.eta << " / " << pr.dSub * 1e6
                                 << " um / " << pr.hConv << " W/m2K\n"
        << "  grid      : " << dom.nx << " x " << dom.ny
        <<                "  (dx=" << dom.dx() * 1e6 << " um, "
                                  "dy=" << dom.dy() * 1e6 << " um)\n"
        << "  dt        : " << dt * 1e6 << " us  (CFL-limited)\n"
        << "  n_steps   : " << n_steps  << "\n"
        << "  tiles     : " << cli.ntiles << "\n"
        << "  threads   : " << cli.n_threads << "\n";

    // -------------------------------------------------------------------
    // Build the Hedgehog dataflow graph.
    // -------------------------------------------------------------------
    auto prof = std::make_shared<StepProfile>();
    hh::Graph<1, HaloTile, HaloTile> graph("LPBF-2D");
    auto step_task = std::make_shared<StepTileTask>(
        "Step", cli.n_threads, mat, pr, dom, dt, prof, phys, cli.kmult);
    graph.inputs(step_task);
    graph.outputs(step_task);
    graph.executeGraph();

    // -------------------------------------------------------------------
    // Global field + bookkeeping
    // -------------------------------------------------------------------
    std::vector<double> T(static_cast<std::size_t>(dom.nx) * dom.ny, pr.Tamb);
    auto [y0_, y1_] = build_tile_bounds(dom.ny, cli.ntiles);

    const std::string snap_dir =
        !cli.out_dir.empty()
            ? cli.out_dir
            : "snapshots_case" + cli.case_id
                  + (phys == Physics::Extended ? "_ext" : "");
    std::filesystem::create_directories(snap_dir);

    std::ofstream log(snap_dir + "/metrics.csv");
    log << "step,t_us,x_laser_um,peakT_K,W_um\n";

    double x_laser = 0.20 * dom.Lx;
    double y_laser = 0.50 * dom.Ly;
    const double x_start = x_laser;

    double peak_T   = pr.Tamb;
    double peak_W   = 0.0;
    // Steady-cruise width: sub-grid width, running max gated to the first
    // traverse with the beam in the central window (excludes the start-up
    // transient and anything after the wrap) -- this is what an ex-situ
    // mid-track cross-section samples.
    double steady_W_sub = 0.0;
    bool   wrapped      = false;
    const auto wall_t0 = std::chrono::steady_clock::now();

    // Phase accumulators (main thread).
    using clk = std::chrono::steady_clock;
    using ns_t = std::chrono::nanoseconds;
    ns_t ns_build{0}, ns_push{0}, ns_wait{0}, ns_stitch{0},
         ns_advance{0}, ns_metrics{0}, ns_snapshot{0};
    auto as_ns = [](auto a, auto b) {
        return std::chrono::duration_cast<ns_t>(b - a);
    };

    // Per-step time series (microseconds).
    std::ofstream perstep(snap_dir + "/perstep.csv");
    perstep << "step,build_us,push_us,wait_us,stitch_us,"
               "advance_us,metrics_us,snapshot_us,total_us\n";

    // -------------------------------------------------------------------
    // Time-stepping loop.  Each iteration: build N tiles -> push -> wait
    // for N outputs -> stitch interiors back into T -> advance laser.
    // -------------------------------------------------------------------
    for (int s = 0; s < n_steps; ++s) {
        const double t = s * dt;

        const auto t_b0 = clk::now();
        std::vector<std::shared_ptr<HaloTile>> tiles;
        tiles.reserve(cli.ntiles);
        for (int i = 0; i < cli.ntiles; ++i) {
            tiles.push_back(make_tile(i, y0_[i], y1_[i], dom, T,
                                      s, t, x_laser, y_laser));
        }
        const auto t_b1 = clk::now();

        for (auto& tile : tiles) graph.pushData(tile);
        const auto t_p1 = clk::now();

        ns_t ns_wait_step{0}, ns_stitch_step{0};
        int collected = 0;
        while (collected < cli.ntiles) {
            const auto t_w0 = clk::now();
            auto opt = graph.getBlockingResult();
            const auto t_w1 = clk::now();
            ns_wait_step += as_ns(t_w0, t_w1);
            if (!opt) break;
            auto updated = std::get<std::shared_ptr<HaloTile>>(*opt);
            for (int r = 0; r < updated->owned_rows(); ++r) {
                const int yg = updated->y0 + r;
                std::copy_n(&updated->at(r + 1, 0), dom.nx,
                            &T[yg * dom.nx]);
            }
            const auto t_s1 = clk::now();
            ns_stitch_step += as_ns(t_w1, t_s1);
            ++collected;
        }
        const auto t_c1 = clk::now();

        x_laser += pr.V * dt;
        if (x_laser > 0.90 * dom.Lx) { x_laser = x_start; wrapped = true; }
        const auto t_a1 = clk::now();

        const auto m = measure(T, dom, mat.Tm);
        if (m.peak_T > peak_T) peak_T = m.peak_T;
        if (m.W_um   > peak_W) peak_W = m.W_um;
        const bool cruise = !wrapped
                         && x_laser >= 0.45 * dom.Lx
                         && x_laser <= 0.85 * dom.Lx;
        if (cruise && m.W_sub_um > steady_W_sub) steady_W_sub = m.W_sub_um;
        const auto t_m1 = clk::now();

        const bool dump = (cli.snap_every > 0 && s % cli.snap_every == 0)
                       || (s == n_steps - 1);
        if (dump) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "%s/T_step_%06d.bin", snap_dir.c_str(), s);
            std::ofstream out(buf, std::ios::binary);
            out.write(reinterpret_cast<const char*>(T.data()),
                      static_cast<std::streamsize>(T.size() * sizeof(double)));
            std::cout << "  step " << s
                      << "  t=" << t * 1e3 << " ms"
                      << "  Tmax=" << m.peak_T << " K"
                      << "  W="    << m.W_um   << " um\n";
            log << s << ',' << t * 1e6 << ',' << x_laser * 1e6
                << ',' << m.peak_T << ',' << m.W_um << '\n';
        }
        const auto t_d1 = clk::now();

        const auto build_us    = as_ns(t_b0, t_b1).count() / 1000.0;
        const auto push_us     = as_ns(t_b1, t_p1).count() / 1000.0;
        const auto wait_us     = ns_wait_step.count()      / 1000.0;
        const auto stitch_us   = ns_stitch_step.count()    / 1000.0;
        const auto advance_us  = as_ns(t_c1, t_a1).count() / 1000.0;
        const auto metrics_us  = as_ns(t_a1, t_m1).count() / 1000.0;
        const auto snapshot_us = as_ns(t_m1, t_d1).count() / 1000.0;
        const auto total_us    = as_ns(t_b0, t_d1).count() / 1000.0;
        perstep << s << ',' << build_us << ',' << push_us << ','
                << wait_us << ',' << stitch_us << ',' << advance_us
                << ',' << metrics_us << ',' << snapshot_us << ','
                << total_us << '\n';

        ns_build    += as_ns(t_b0, t_b1);
        ns_push     += as_ns(t_b1, t_p1);
        ns_wait     += ns_wait_step;
        ns_stitch   += ns_stitch_step;
        ns_advance  += as_ns(t_c1, t_a1);
        ns_metrics  += as_ns(t_a1, t_m1);
        ns_snapshot += as_ns(t_m1, t_d1);
    }

    graph.finishPushingData();
    graph.waitForTermination();

    // Dump the Hedgehog DOT graph (post-execution, EXECUTION colouring +
    // per-thread structure) so `dot -Tpng` can render it.
    {
        const std::string dot_path = snap_dir + "/graph.dot";
        graph.createDotFile(dot_path,
                            hh::ColorScheme::EXECUTION,
                            hh::StructureOptions::ALL);
        std::cout << "  dot graph         : " << dot_path << '\n';
    }

    const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - wall_t0).count();

    // Also dump a small JSON with grid/process metadata + the full
    // profiling roll-up for the Python post-processor.
    auto ms = [](ns_t x) { return x.count() / 1.0e6; };
    const double compute_ms_total = prof->compute_ns.load() / 1.0e6;
    const double alloc_ms_total   = prof->alloc_ns  .load() / 1.0e6;
    const double stencil_ms_total = prof->stencil_ns.load() / 1.0e6;
    const std::int64_t n_calls    = prof->n_calls   .load();

    std::ofstream meta(snap_dir + "/meta.json");
    meta << "{\n"
         << "  \"case\":     \"" << cli.case_id << "\",\n"
         << "  \"physics\":  \"" << cli.physics << "\",\n"
         << "  \"kmult\":    " << cli.kmult      << ",\n"
         << "  \"eta\":      " << mat.eta        << ",\n"
         << "  \"dsub_m\":   " << pr.dSub        << ",\n"
         << "  \"hconv\":    " << pr.hConv       << ",\n"
         << "  \"P_W\":      " << pr.P           << ",\n"
         << "  \"V_mps\":    " << pr.V           << ",\n"
         << "  \"r0_m\":     " << pr.r0          << ",\n"
         << "  \"nx\":       " << dom.nx         << ",\n"
         << "  \"ny\":       " << dom.ny         << ",\n"
         << "  \"Lx_m\":     " << dom.Lx         << ",\n"
         << "  \"Ly_m\":     " << dom.Ly         << ",\n"
         << "  \"Tm_K\":     " << mat.Tm         << ",\n"
         << "  \"dt_s\":     " << dt             << ",\n"
         << "  \"n_steps\":  " << n_steps        << ",\n"
         << "  \"ntiles\":   " << cli.ntiles     << ",\n"
         << "  \"threads\":  " << cli.n_threads  << ",\n"
         << "  \"peak_T_K\": " << peak_T         << ",\n"
         << "  \"peak_W_um\":" << peak_W         << ",\n"
         << "  \"steady_W_sub_um\": " << steady_W_sub << ",\n"
         << "  \"wall_ms\":  " << wall_ms        << ",\n"
         << "  \"phases_ms\": {\n"
         << "    \"build\":    "  << ms(ns_build)    << ",\n"
         << "    \"push\":     "  << ms(ns_push)     << ",\n"
         << "    \"wait\":     "  << ms(ns_wait)     << ",\n"
         << "    \"stitch\":   "  << ms(ns_stitch)   << ",\n"
         << "    \"advance\":  "  << ms(ns_advance)  << ",\n"
         << "    \"metrics\":  "  << ms(ns_metrics)  << ",\n"
         << "    \"snapshot\": "  << ms(ns_snapshot) << "\n"
         << "  },\n"
         << "  \"task_ms\": {\n"
         << "    \"compute_total_cpu\":  " << compute_ms_total << ",\n"
         << "    \"alloc_total_cpu\":    " << alloc_ms_total   << ",\n"
         << "    \"stencil_total_cpu\":  " << stencil_ms_total << ",\n"
         << "    \"n_calls\":             " << n_calls          << "\n"
         << "  }\n"
         << "}\n";

    std::cout << "\nDone.\n"
              << "  wall time         : " << wall_ms << " ms\n"
              << "  peak T            : " << peak_T  << " K\n"
              << "  peak W            : " << peak_W  << " um\n"
              << "  steady W (subgrid): " << steady_W_sub << " um\n"
              << "  output dir        : " << snap_dir << '\n'
              << "  --- timing roll-up (ms) ---\n"
              << "  build tiles       : " << ms(ns_build)    << '\n'
              << "  push to graph     : " << ms(ns_push)     << '\n'
              << "  wait on graph     : " << ms(ns_wait)     << '\n'
              << "  stitch back       : " << ms(ns_stitch)   << '\n'
              << "  advance laser     : " << ms(ns_advance)  << '\n'
              << "  global metrics    : " << ms(ns_metrics)  << '\n'
              << "  snapshot I/O      : " << ms(ns_snapshot) << '\n'
              << "  (worker total cpu): " << compute_ms_total
              << "   (" << n_calls << " calls, "
              << (n_calls ? compute_ms_total*1000.0/n_calls : 0.0)
              << " us / call)\n"
              << "  (worker alloc cpu): " << alloc_ms_total   << '\n'
              << "  (worker stencil  ): " << stencil_ms_total << '\n';
    return 0;
}
