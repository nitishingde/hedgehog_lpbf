// SPDX-License-Identifier: 0BSD
// ===========================================================================
// hedgehog_lpbf_hhcuda — a REAL Hedgehog dataflow graph whose stencil compute
// runs on the GPU.  Single-GPU port of the 2D z-averaged explicit-FD LPBF
// heat-equation step (src/main.cpp + include/step_task.hpp), reusing the CUDA
// kernels of the standalone port (now shared via src/gpu/kernels.cu).
//
// COMPILATION: this TU is compiled by g++ at C++20 (NOT nvcc).  It includes
// Hedgehog (with HH_USE_CUDA -> hh::AbstractCUDATask) and <cuda_runtime.h>, but
// never writes <<<>>> and never includes the device-physics header: all kernel
// work goes through the extern "C" launch wrappers in lpbf_kernels.hpp.  Link:
// Hedgehog (header-only) + Threads + CUDA::cudart + the kernels.cu object.
//
// GRAPH TOPOLOGY (see include/gpu_step_task.hpp):
//
//     seed Tick(step 0) ─► [ GpuStepTask ]  (green CUDA node, 1 thread)
//                              │  ▲ self-cycle: emits Tick(step s+1) to itself
//                              │  └────────────────────────────────────┐
//                              ├── MetricsMsg (every step) ─► [ MetricsTask ] ─► StepResult
//                              └── SnapshotMsg (dump steps) ─► [ SnapshotWriterTask ] ─► StepResult
//
// The GPU task advances the field + reduces on its per-task stream; the two CPU
// sink tasks finalise metrics / write .bin snapshots off the critical path.  The
// exported graph.dot therefore shows genuine producer-consumer structure with
// per-edge queue occupancy and per-node execution-time colouring.
//
// Numerics are bit-for-bit identical to the standalone GPU solver: the same
// kernels, the same StepConst, the same laser law and metric reductions.
//
// CLI mirrors the existing binaries (superset shared with lpbf_cuda.cu):
//   --case --physics --tend --snap-every --threads (CPU sink threads)
//   --P --V --r0 --eta --dsub --hconv --kmult --out --device --bench
// ===========================================================================
#include <hedgehog.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "params.hpp"
#include "lpbf_step_const.hpp"
#include "gpu_messages.hpp"
#include "gpu_step_task.hpp"

using namespace lpbf;

// ---------------------------------------------------------------------------
struct CLI {
    std::string case_id   = "0";
    int         n_threads = 4;        // threads for the CPU sink tasks
    double      t_end     = 1.5e-3;
    int         snap_every = 100;
    std::string physics   = "base";
    std::string out_dir;
    double P_ovr = -1, V_ovr = -1, r0_ovr = -1;
    double eta_ovr = -1, dsub_ovr = -1, hconv_ovr = -1;
    double kmult = 1.0;
    int    device   = 0;
    int    batch    = 64;     // OPT #2: logical steps advanced per task activation
    int    mode     = 2;      // 0 legacy baseline, 1 opt1-only, 2 opt1+opt2 batched
    bool   fused    = false;  // OPT #3: fuse step+reduce
    bool   pinned   = false;  // OPT #4: pinned per-batch metric staging
    bool   do_bench = false;
    bool   quiet    = false;  // ENS: emit one compact RESULT csv line, no disk output
    std::string manifest;     // ENS: CSV of runs looped in-process (one CUDA ctx)
};

static CLI parse_cli(int argc, char** argv) {
    CLI c;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto starts = [&](char const* p) { return a.rfind(p, 0) == 0; };
        if      (starts("--case="))       c.case_id    = a.substr(7);
        else if (starts("--threads="))    c.n_threads  = std::atoi(a.c_str() + 10);
        else if (starts("--tend="))       c.t_end      = std::atof(a.c_str() +  7);
        else if (starts("--snap-every=")) c.snap_every = std::atoi(a.c_str() + 13);
        else if (starts("--physics="))    c.physics    = a.substr(10);
        else if (starts("--out="))        c.out_dir    = a.substr(6);
        else if (starts("--P="))          c.P_ovr      = std::atof(a.c_str() +  4);
        else if (starts("--V="))          c.V_ovr      = std::atof(a.c_str() +  4);
        else if (starts("--r0="))         c.r0_ovr     = std::atof(a.c_str() +  5);
        else if (starts("--eta="))        c.eta_ovr    = std::atof(a.c_str() +  6);
        else if (starts("--dsub="))       c.dsub_ovr   = std::atof(a.c_str() +  7);
        else if (starts("--hconv="))      c.hconv_ovr  = std::atof(a.c_str() +  8);
        else if (starts("--kmult="))      c.kmult      = std::atof(a.c_str() +  8);
        else if (starts("--device="))     c.device     = std::atoi(a.c_str() +  9);
        else if (starts("--batch="))      c.batch      = std::atoi(a.c_str() +  8);
        else if (starts("--mode="))       c.mode       = std::atoi(a.c_str() +  7);
        else if (a == "--fused")          c.fused      = true;
        else if (a == "--pinned")         c.pinned     = true;
        else if (a == "--bench")          c.do_bench   = true;
        else if (a == "--quiet")          c.quiet      = true;
        else if (starts("--manifest="))   c.manifest   = a.substr(11);
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "Usage: hedgehog_lpbf_hhcuda [options]   (Hedgehog+CUDA 2D solver)\n"
                "  --case=ID            0,1.1,1.2,2.1,2.2,3.1,3.2 (default 0)\n"
                "  --physics=base|ext   constant props | Mills k(T),cp(T)+latent\n"
                "  --P=W --V=M/S --r0=M     process overrides (SI)\n"
                "  --eta=F --dsub=M --hconv=W/M2K   model overrides\n"
                "  --kmult=F            melt-pool conductivity enhancement\n"
                "  --out=DIR            output directory override\n"
                "  --tend=SECS          physical end-time (default 1.5e-3)\n"
                "  --snap-every=N       snapshot stride (0: final only)\n"
                "  --threads=N          worker threads for the CPU sink tasks\n"
                "  --batch=K            logical steps advanced per GPU task activation\n"
                "  --mode=0|1|2         0 legacy baseline, 1 opt1-only, 2 opt1+opt2 (default)\n"
                "  --fused              OPT #3: fuse step+reduce into one grid launch\n"
                "  --pinned             OPT #4: pinned per-batch metric staging\n"
                "  --device=N           CUDA device (default 0)\n"
                "  --bench              print wall ms + steps/s (no file output)\n"
                "  --quiet              emit one compact RESULT csv line (no disk output)\n"
                "  --manifest=FILE      loop a runs CSV in-process (one CUDA context)\n";
            std::exit(0);
        }
    }
    return c;
}

static void apply_case(std::string const& id, Process& pr) {
    if      (id == "0")   { pr.P = 285.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else if (id == "1.1") { pr.P = 285.0; pr.V = 0.96; pr.r0 = 24.5e-6; }
    else if (id == "1.2") { pr.P = 285.0; pr.V = 0.96; pr.r0 = 41.0e-6; }
    else if (id == "2.1") { pr.P = 285.0; pr.V = 1.20; pr.r0 = 33.5e-6; }
    else if (id == "2.2") { pr.P = 285.0; pr.V = 0.80; pr.r0 = 33.5e-6; }
    else if (id == "3.1") { pr.P = 325.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else if (id == "3.2") { pr.P = 245.0; pr.V = 0.96; pr.r0 = 33.5e-6; }
    else { std::cerr << "unknown case '" << id << "'\n"; std::exit(1); }
}

// Pre-compute the per-step Tick plan using the exact laser law of the CPU /
// standalone-GPU solvers (kernel uses the PRE-advance position; csv/cruise use
// the POST-advance position + wrap state).
static std::vector<Tick> build_plan(const Process& pr, const Domain& dom,
                                    double dt, int n_steps, int snap_every) {
    std::vector<Tick> plan(n_steps);
    double x_laser = 0.20 * dom.Lx, y_laser = 0.50 * dom.Ly;
    const double x_start = x_laser;
    bool wrapped = false;
    for (int s = 0; s < n_steps; ++s) {
        Tick& tk = plan[s];
        tk.step = s; tk.t = s * dt;
        tk.x_l = x_laser; tk.y_l = y_laser; tk.q_scale = 1.0;
        x_laser += pr.V * dt;
        if (x_laser > 0.90 * dom.Lx) { x_laser = x_start; wrapped = true; }
        tk.x_laser_csv = x_laser; tk.wrapped = wrapped;
        tk.dump = (snap_every > 0 && s % snap_every == 0) || (s == n_steps - 1);
    }
    return plan;
}

// ===========================================================================
// ENS additions (additive; not reachable on any default-flag path) --------
// One fully-resolved run.  physics="base"|"ext"; all fields are SI.
struct RunSpec {
    std::string run_id, cond_id, draw_id, config, physics;
    double P = 0, V = 0, r0 = 0, eta = 0, dsub_m = 0, kmult = 1.0;
};
struct RowResult { double steady_W = 0, peak_T = 0, wall_ms = 0; long n_steps = 0; };

// Execute one run with NO disk output (write_outputs=false), reusing the exact
// plan/graph/kernels of the single-run path.  The CUDA context/device is already
// selected by the caller; this only rebuilds the (cheap) per-run graph + buffers.
static RowResult run_one_row(const RunSpec& rs, double t_end, const CLI& cli) {
    Material mat; Process pr; Domain dom;
    pr.P = rs.P; pr.V = rs.V; pr.r0 = rs.r0;
    mat.eta = rs.eta; pr.dSub = rs.dsub_m;

    const Physics phys = (rs.physics == "ext") ? Physics::Extended : Physics::Baseline;
    const double dt = (phys == Physics::Extended) ? dt_cfl_ext(mat, dom, rs.kmult)
                                                  : dt_cfl(mat, dom);
    const int n_steps = static_cast<int>(t_end / dt);

    std::vector<Tick> plan = build_plan(pr, dom, dt, n_steps, /*snap_every=*/0);
    StepConst k = make_const(mat, pr, dom, dt);  k.kmult = rs.kmult;
    const int phys_i = (phys == Physics::Extended) ? 1 : 0;

    auto acc = std::make_shared<RunAccum>();
    hh::Graph<1, Tick, StepResult> graph("LPBF-2D-HHCUDA");
    auto gpu = std::make_shared<GpuStepTask>(dom, k, phys_i, mat.Tm, n_steps,
                                             plan, pr.Tamb, /*write_outputs=*/false,
                                             cli.batch, cli.mode, cli.fused, cli.pinned);
    auto metrics = std::make_shared<MetricsTask>(dom, mat.Tm, acc, false, std::string());
    auto snap = std::make_shared<SnapshotWriterTask>(1, std::string());
    graph.inputs(gpu);
    graph.edges(gpu, gpu);
    graph.edges(gpu, metrics);
    graph.edges(gpu, snap);
    graph.outputs(metrics);
    graph.outputs(snap);

    graph.executeGraph();
    const auto wall0 = std::chrono::steady_clock::now();
    graph.pushData(std::make_shared<Tick>(plan[0]));
    graph.finishPushingData();
    while (auto res = graph.getBlockingResult()) { (void)res; }
    graph.waitForTermination();
    const double wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - wall0).count();

    return RowResult{acc->steady_W_sub, acc->peak_T, wall_ms,
                     static_cast<long>(n_steps)};
}

// Emit exactly one compact CSV line for a completed run (crash-safe: flushed).
static void emit_result_line(const RunSpec& rs, const RowResult& r) {
    std::printf("RESULT,%s,%s,%s,%s,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.3f,%ld\n",
                rs.run_id.c_str(), rs.cond_id.c_str(), rs.draw_id.c_str(),
                rs.config.c_str(), rs.physics.c_str(),
                rs.P, rs.V, rs.r0, rs.eta, rs.dsub_m, rs.kmult,
                r.steady_W, r.peak_T, r.wall_ms, r.n_steps);
    std::fflush(stdout);
}

// Split a CSV line on commas (no quoting in our manifests).
static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out; std::string cur;
    for (char ch : line) {
        if (ch == ',') { out.push_back(cur); cur.clear(); }
        else if (ch != '\r') cur.push_back(ch);
    }
    out.push_back(cur);
    return out;
}

// Loop a manifest CSV in-process; header names are matched by column so column
// order is not load-bearing.  Returns 0 on success.
static int run_manifest(const CLI& cli) {
    std::ifstream f(cli.manifest);
    if (!f) { std::cerr << "cannot open manifest '" << cli.manifest << "'\n"; return 4; }
    std::string header;
    if (!std::getline(f, header)) { std::cerr << "empty manifest\n"; return 4; }
    std::vector<std::string> cols = split_csv(header);
    auto col = [&](const char* name) -> int {
        for (std::size_t i = 0; i < cols.size(); ++i) if (cols[i] == name) return (int)i;
        std::cerr << "manifest missing column '" << name << "'\n"; std::exit(4);
    };
    const int i_run = col("run_id"),   i_cond = col("cond_id"), i_draw = col("draw_id"),
              i_cfg = col("config"),   i_P = col("P_W"),        i_V = col("V_mps"),
              i_r0  = col("r0_m"),     i_eta = col("eta"),      i_dsub = col("dsub_m"),
              i_km  = col("kmult"),    i_phys = col("physics");

    // Header for downstream parsers (comment line; RESULT rows follow).
    std::printf("# RESULT,run_id,cond_id,draw_id,config,physics,P_W,V_mps,r0_m,eta,"
                "dsub_m,kmult,steady_W_sub_um,peak_T_K,wall_ms,n_steps\n");
    std::fflush(stdout);

    std::string line; long done = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> v = split_csv(line);
        RunSpec rs;
        rs.run_id  = v[i_run];  rs.cond_id = v[i_cond]; rs.draw_id = v[i_draw];
        rs.config  = v[i_cfg];  rs.physics = v[i_phys];
        rs.P  = std::atof(v[i_P].c_str());   rs.V = std::atof(v[i_V].c_str());
        rs.r0 = std::atof(v[i_r0].c_str());  rs.eta = std::atof(v[i_eta].c_str());
        rs.dsub_m = std::atof(v[i_dsub].c_str()); rs.kmult = std::atof(v[i_km].c_str());
        emit_result_line(rs, run_one_row(rs, cli.t_end, cli));
        ++done;
    }
    std::fprintf(stderr, "[manifest] completed %ld runs\n", done);
    return 0;
}

// ===========================================================================
int main(int argc, char** argv) {
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

    int ndev = 0; cudaGetDeviceCount(&ndev);
    if (ndev == 0) { std::cerr << "No CUDA device found.\n"; return 3; }
    if (cli.device < 0 || cli.device >= ndev) {
        std::cerr << "Invalid --device=" << cli.device << " (have " << ndev << ")\n";
        return 3;
    }
    cudaSetDevice(cli.device);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, cli.device);

    // ENS: manifest mode -- loop many runs in this one process/CUDA context.
    // Gated behind --manifest; the default single-run path below is untouched.
    if (!cli.manifest.empty()) return run_manifest(cli);

    // ENS: single-run compact-CSV mode (gated behind --quiet).  Reuses the same
    // resolved (pr,mat) overrides above via a RunSpec, emits one RESULT line.
    if (cli.quiet) {
        RunSpec rs;
        rs.run_id = cli.case_id; rs.cond_id = cli.case_id; rs.draw_id = "0";
        rs.config = "single";    rs.physics = cli.physics;
        rs.P = pr.P; rs.V = pr.V; rs.r0 = pr.r0;
        rs.eta = mat.eta; rs.dsub_m = pr.dSub; rs.kmult = cli.kmult;
        emit_result_line(rs, run_one_row(rs, cli.t_end, cli));
        return 0;
    }

    const bool write_outputs = !cli.do_bench;
    const std::string snap_dir = !cli.out_dir.empty() ? cli.out_dir
        : "snapshots_case" + cli.case_id
          + (phys == Physics::Extended ? "_ext" : "") + "_hhcuda";

    std::cout << "hedgehog_lpbf_hhcuda (Hedgehog + CUDA, 2D z-averaged)\n"
              << "  device    : " << cli.device << " " << prop.name << "\n"
              << "  case      : " << cli.case_id << "  (P=" << pr.P << " W, V="
              << pr.V*1e3 << " mm/s, r0=" << pr.r0*1e6 << " um)\n"
              << "  physics   : " << (phys==Physics::Extended?"extended":"baseline") << "\n"
              << "  grid      : " << dom.nx << " x " << dom.ny << "\n"
              << "  dt        : " << dt*1e6 << " us   n_steps=" << n_steps << "\n"
              << "  threads   : " << cli.n_threads << " (CPU sink tasks)\n"
              << "  batch     : " << cli.batch << " steps/activation\n"
              << "  opt       : mode=" << cli.mode
              << (cli.fused ? " +fused" : "") << (cli.pinned ? " +pinned" : "") << "\n"
              << "  mode      : " << (cli.do_bench ? "bench (no output)" : "run") << "\n";

    // -------------------------------------------------------------------
    // Build the plan and the Hedgehog graph.
    // -------------------------------------------------------------------
    std::vector<Tick> plan = build_plan(pr, dom, dt, n_steps, cli.snap_every);
    StepConst k = make_const(mat, pr, dom, dt);  k.kmult = cli.kmult;
    const int phys_i = (phys == Physics::Extended) ? 1 : 0;

    auto acc = std::make_shared<RunAccum>();
    hh::Graph<1, Tick, StepResult> graph("LPBF-2D-HHCUDA");
    auto gpu = std::make_shared<GpuStepTask>(dom, k, phys_i, mat.Tm, n_steps,
                                             plan, pr.Tamb, write_outputs,
                                             cli.batch, cli.mode, cli.fused,
                                             cli.pinned);
    auto metrics = std::make_shared<MetricsTask>(dom, mat.Tm, acc, write_outputs, snap_dir);
    auto snap = std::make_shared<SnapshotWriterTask>(
        static_cast<std::size_t>(cli.n_threads < 1 ? 1 : cli.n_threads), snap_dir);

    graph.inputs(gpu);
    graph.edges(gpu, gpu);        // self-cycle: recirculate the next Tick
    graph.edges(gpu, metrics);    // MetricsMsg -> MetricsTask
    graph.edges(gpu, snap);       // SnapshotMsg -> SnapshotWriterTask
    graph.outputs(metrics);
    graph.outputs(snap);

    if (write_outputs) std::filesystem::create_directories(snap_dir);

    graph.executeGraph();
    const auto wall0 = std::chrono::steady_clock::now();
    graph.pushData(std::make_shared<Tick>(plan[0]));  // seed step 0
    graph.finishPushingData();                        // no more EXTERNAL input
    // Drain the terminal StepResult tokens; getBlockingResult returns null when
    // the whole graph (including the self-cycle) has terminated.
    while (auto res = graph.getBlockingResult()) { (void)res; }
    graph.waitForTermination();
    const double wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - wall0).count();

    const double steps_per_s = wall_ms > 0 ? n_steps * 1000.0 / wall_ms : 0.0;

    if (write_outputs) {
        // Profiled dataflow graph: EXECUTION colouring + THREADING + QUEUE
        // (StructureOptions::ALL -> max & along-edge queue counts).
        graph.createDotFile(snap_dir + "/graph.dot",
                            hh::ColorScheme::EXECUTION,
                            hh::StructureOptions::ALL);

        std::ofstream meta(snap_dir + "/meta.json");
        meta << "{\n"
             << "  \"case\":     \"" << cli.case_id << "\",\n"
             << "  \"backend\":  \"hedgehog+cuda\",\n"
             << "  \"device\":   \"" << prop.name << "\",\n"
             << "  \"physics\":  \"" << cli.physics << "\",\n"
             << "  \"kmult\":    " << cli.kmult << ",\n"
             << "  \"eta\":      " << mat.eta << ",\n"
             << "  \"dsub_m\":   " << pr.dSub << ",\n"
             << "  \"hconv\":    " << pr.hConv << ",\n"
             << "  \"P_W\":      " << pr.P << ",\n"
             << "  \"V_mps\":    " << pr.V << ",\n"
             << "  \"r0_m\":     " << pr.r0 << ",\n"
             << "  \"nx\":       " << dom.nx << ",\n"
             << "  \"ny\":       " << dom.ny << ",\n"
             << "  \"Lx_m\":     " << dom.Lx << ",\n"
             << "  \"Ly_m\":     " << dom.Ly << ",\n"
             << "  \"Tm_K\":     " << mat.Tm << ",\n"
             << "  \"dt_s\":     " << dt << ",\n"
             << "  \"n_steps\":  " << n_steps << ",\n"
             << "  \"threads\":  " << cli.n_threads << ",\n"
             << "  \"batch\":    " << cli.batch << ",\n"
             << "  \"mode\":     " << cli.mode << ",\n"
             << "  \"fused\":    " << (cli.fused ? "true" : "false") << ",\n"
             << "  \"pinned\":   " << (cli.pinned ? "true" : "false") << ",\n"
             << "  \"peak_T_K\": " << acc->peak_T << ",\n"
             << "  \"peak_W_um\":" << acc->peak_W << ",\n"
             << "  \"steady_W_sub_um\": " << acc->steady_W_sub << ",\n"
             << "  \"wall_ms\":  " << wall_ms << ",\n"
             << "  \"steps_per_s\": " << steps_per_s << "\n"
             << "}\n";
    }

    if (cli.do_bench) {
        std::cout << "\n[--bench] Hedgehog+CUDA graph\n"
                  << "  wall      : " << wall_ms << " ms\n"
                  << "  steps/s   : " << steps_per_s << "\n"
                  << "  peak_T    : " << acc->peak_T << " K\n"
                  << "  steps     : " << n_steps << "\n";
        return 0;
    }

    std::cout << "\nDone.\n"
              << "  peak T : " << acc->peak_T << " K\n"
              << "  peak W : " << acc->peak_W << " um\n"
              << "  steady W (subgrid): " << acc->steady_W_sub << " um\n"
              << "  wall   : " << wall_ms << " ms  (" << steps_per_s << " steps/s)\n"
              << "  output : " << snap_dir << "\n"
              << "  dot    : " << snap_dir << "/graph.dot\n";
    return 0;
}
