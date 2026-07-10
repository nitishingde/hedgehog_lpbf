// SPDX-License-Identifier: 0BSD
// One horizontal row-strip of the global 2D field with halo rows above
// and below.  Owned rows live at indices 1 .. (owned_rows()) inclusive;
// row 0 is the top halo, row total_rows()-1 is the bottom halo.
#pragma once
#include <vector>

namespace lpbf {

struct HaloTile {
    // Per-tile geometry (constant across the run for a given tile idx)
    int idx   = 0;
    int y0    = 0;     // first owned row in the global grid
    int y1    = 0;     // last  owned row in the global grid (inclusive)
    int nx    = 0;     // number of columns (global)

    // Time-dependent context (refreshed by the driver each step)
    int    step    = 0;
    double t       = 0.0;
    double x_laser = 0.0;
    double y_laser = 0.0;
    double q_scale = 1.0;   // source multiplier (0 while the laser jumps
                            // between hatch lines with the beam off)

    // Storage (row-major)
    std::vector<double> data;

    int owned_rows() const noexcept { return y1 - y0 + 1; }
    int total_rows() const noexcept { return owned_rows() + 2; }

    double&       at(int r, int c)       noexcept { return data[r * nx + c]; }
    double        at(int r, int c) const noexcept { return data[r * nx + c]; }
};

}  // namespace lpbf
