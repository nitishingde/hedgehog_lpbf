// SPDX-License-Identifier: 0BSD
// ---------------------------------------------------------------------------
// POD / host-only message types that flow along the Hedgehog dataflow edges of
// the Hedgehog+CUDA LPBF graph (src/gpu/hh_gpu_main.cpp).  No CUDA tokens, no
// device code — plain C++.
// ---------------------------------------------------------------------------
#pragma once
#include <memory>
#include <vector>

namespace lpbf {

// One time step's worth of work.  Recirculated through the GPU task: after
// advancing step s the GPU task emits the Tick for step s+1 back to itself
// (a genuine self-cycle), until the final step.  All laser bookkeeping is
// pre-computed by the host driver via the exact laser law of the CPU/standalone
// solvers, so the GPU task stays a pure "advance + reduce" node.
struct Tick {
    int    step;          // step index s (0-based)
    double t;             // physical time  = s * dt   [s]
    double x_l, y_l;      // laser position used by step_kernel for step s
    double q_scale;       // single-track: always 1
    double x_laser_csv;   // POST-advance laser x (logged to metrics.csv)
    bool   wrapped;       // POST-advance wrap state (for the cruise gate)
    bool   dump;          // snapshot / csv-row on this step?
};

// Raw reduce output for one step, produced by the GPU task and finalised by the
// (single-threaded, ordered) MetricsTask into peak-T / melt-width metrics.
struct MetricsMsg {
    int    step;
    double t;             // physical time [s]
    double x_laser_csv;   // POST-advance laser x [m]
    bool   wrapped;
    bool   dump;
    double tmax;          // peak T over the field [K]
    int    x_hot;         // column index of the peak
    int    ymin, ymax;    // molten row extent (ymax < 0 => none molten)
    std::vector<double> hot_col;   // T along the hottest column (empty if none)
};

// Full-field snapshot for one dump step: host copy of the device field, written
// to T_step_%06d.bin by the (parallel) SnapshotWriterTask off the GPU's path.
struct SnapshotMsg {
    int    step;
    std::shared_ptr<std::vector<double>> field;  // nx*ny doubles, row-major
};

// Terminal token drained at the graph output (one per processed step).
struct StepResult { int step; };

}  // namespace lpbf
