// SPDX-License-Identifier: 0BSD
// Hedgehog task that advances one row-strip of the 2D field by one
// explicit-Euler step of the z-averaged heat equation
//
//     dT_bar/dt = K (Txx + Tyy) + S_bar + (K/h)(J_top - J_bot)
//
// realised through physical heat fluxes
//
//     -k dT/dz|_top = h_conv (T - Tamb) + eps sigma (T^4 - Tamb^4)
//     -k dT/dz|_bot = - (k / d_sub)(T - Tsub).
//
// The tile arrives with up-to-date halo rows; this task does NOT touch
// the halo rows (they're rebuilt by the driver before the next step).
#pragma once

#include <hedgehog/hedgehog.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include "params.hpp"
#include "tile.hpp"

namespace lpbf {

// Profiling counters shared across all worker threads of one StepTileTask
// (one shared atomic per counter, written to with relaxed memory order).
struct StepProfile {
    std::atomic<std::int64_t> compute_ns{0};   // wall ns in execute() bodies
    std::atomic<std::int64_t> alloc_ns  {0};   // allocating T_new
    std::atomic<std::int64_t> stencil_ns{0};   // Laplacian + source + loss
    std::atomic<std::int64_t> n_calls   {0};   // total execute() calls
};

class StepTileTask
    : public hh::AbstractTask<1, HaloTile, HaloTile>
{
public:
    StepTileTask(std::string const& name,
                 std::size_t        n_threads,
                 Material           mat,
                 Process            pr,
                 Domain             dom,
                 double             dt,
                 std::shared_ptr<StepProfile> prof,
                 Physics            phys   = Physics::Baseline,
                 double             kmult  = 1.0) noexcept
        : hh::AbstractTask<1, HaloTile, HaloTile>(name, n_threads),
          mat_(mat), pr_(pr), dom_(dom), dt_(dt), prof_(std::move(prof)),
          phys_(phys), kmult_(kmult) {}

    void execute(std::shared_ptr<HaloTile> tile) override {
        using clk = std::chrono::steady_clock;
        const auto t_call0 = clk::now();
        const int    nx    = tile->nx;
        const int    rows  = tile->total_rows();
        const double dx    = dom_.dx();
        const double dy    = dom_.dy();
        const double alpha = mat_.alpha();
        const double K     = mat_.k;
        const double rho   = mat_.rho;
        const double cp    = mat_.cp;
        const double inv_h = 1.0 / pr_.h;

        const double inv_r0_sq    = 1.0 / (pr_.r0 * pr_.r0);
        const double peak_S       = (2.0 * mat_.eta * pr_.P)
                                  / (M_PI * pr_.r0 * pr_.r0);
        const double rho_cp       = rho * cp;
        const double x_l          = tile->x_laser;
        const double y_l          = tile->y_laser;
        const double q_scale      = tile->q_scale;   // 0 when beam jumps
        const double inv_dx2      = 1.0 / (dx * dx);
        const double inv_dy2      = 1.0 / (dy * dy);

        const auto t_alloc0 = clk::now();
        std::vector<double> T_new(rows * nx);
        // Copy halo rows through unchanged so the tile object stays self-
        // consistent if the driver wants to inspect them.
        std::copy_n(tile->data.data(),           nx, T_new.data());
        std::copy_n(tile->data.data() + (rows - 1) * nx,
                    nx,
                    T_new.data() + (rows - 1) * nx);
        const auto t_alloc1 = clk::now();

        if (phys_ == Physics::Baseline) {
        for (int r = 1; r < rows - 1; ++r) {
            const int   yg = tile->y0 + (r - 1);
            const double y  = static_cast<double>(yg) * dy;

            for (int c = 0; c < nx; ++c) {
                const double T = tile->at(r, c);

                // Laplacian (insulated x-boundary by mirror).
                double lap_x;
                if (c == 0) {
                    lap_x = (tile->at(r, c + 1) - T) * inv_dx2;
                } else if (c == nx - 1) {
                    lap_x = (tile->at(r, c - 1) - T) * inv_dx2;
                } else {
                    lap_x = (tile->at(r, c + 1) - 2.0 * T + tile->at(r, c - 1))
                          * inv_dx2;
                }
                const double lap_y =
                    (tile->at(r + 1, c) - 2.0 * T + tile->at(r - 1, c))
                    * inv_dy2;

                // Moving-Gaussian source averaged through the layer.
                const double x  = static_cast<double>(c) * dx;
                const double r2 = (x - x_l) * (x - x_l) + (y - y_l) * (y - y_l);
                const double S  = q_scale * peak_S
                                  * std::exp(-2.0 * r2 * inv_r0_sq)
                                  / (rho_cp * pr_.h);

                // Top loss (convection + optional radiation).
                double q_top = pr_.hConv * (T - pr_.Tamb);
                if (pr_.radiation) {
                    const double T4    = T * T * T * T;
                    const double Ta4   = pr_.Tamb * pr_.Tamb
                                       * pr_.Tamb * pr_.Tamb;
                    q_top += mat_.eps * SIGMA_SB * (T4 - Ta4);
                }
                // Bottom loss (conduction to substrate, linear gradient).
                const double q_bot = (K / pr_.dSub) * (T - pr_.Tsub);
                const double loss  = (q_top + q_bot) * inv_h / rho_cp;

                const double dTdt = alpha * (lap_x + lap_y) + S - loss;
                T_new[r * nx + c] = T + dt_ * dTdt;
            }
        }
        } else {
        // Extended physics: temperature-dependent k(T), cp(T) (Mills) and
        // apparent-cp latent heat.  Conservative form div(k grad T) with
        // arithmetic-mean face conductivities; the copy-ghost adiabatic
        // x-BC is realised by zeroing the face flux at the domain edge
        // (ghost == centre).  y-BCs arrive through the halo rows exactly
        // as in the baseline path.
        for (int r = 1; r < rows - 1; ++r) {
            const int   yg = tile->y0 + (r - 1);
            const double y  = static_cast<double>(yg) * dy;

            for (int c = 0; c < nx; ++c) {
                const double T   = tile->at(r, c);
                const double Txm = (c == 0)      ? T : tile->at(r, c - 1);
                const double Txp = (c == nx - 1) ? T : tile->at(r, c + 1);
                const double Tym = tile->at(r - 1, c);
                const double Typ = tile->at(r + 1, c);

                // In-plane transport uses the enhanced effective conductivity
                // (Marangoni proxy); the vertical substrate sink below uses
                // the base conductivity (the solid substrate does not convect).
                const double kCe = in718::k_eff(T, kmult_);
                const double kxm = 0.5 * (kCe + in718::k_eff(Txm, kmult_));
                const double kxp = 0.5 * (kCe + in718::k_eff(Txp, kmult_));
                const double kym = 0.5 * (kCe + in718::k_eff(Tym, kmult_));
                const double kyp = 0.5 * (kCe + in718::k_eff(Typ, kmult_));

                const double div_k_grad =
                      (kxp * (Txp - T) - kxm * (T - Txm)) * inv_dx2
                    + (kyp * (Typ - T) - kym * (T - Tym)) * inv_dy2;

                // Absorbed surface flux (W/m^2) and losses (W/m^2).
                const double x  = static_cast<double>(c) * dx;
                const double r2 = (x - x_l) * (x - x_l)
                                + (y - y_l) * (y - y_l);
                const double q_in = q_scale * peak_S
                                  * std::exp(-2.0 * r2 * inv_r0_sq);

                double q_top = pr_.hConv * (T - pr_.Tamb);
                if (pr_.radiation) {
                    const double T4  = T * T * T * T;
                    const double Ta4 = pr_.Tamb * pr_.Tamb
                                     * pr_.Tamb * pr_.Tamb;
                    q_top += mat_.eps * SIGMA_SB * (T4 - Ta4);
                }
                // Substrate conduction with the base (unenhanced) conductivity.
                const double q_bot = (in718::k_of(T) / pr_.dSub) * (T - pr_.Tsub);

                // rho * cp_eff(T) * dT/dt = div(k grad T) + (q_in - losses)/h
                const double rhs_wm3 =
                    div_k_grad + (q_in - q_top - q_bot) * inv_h;
                T_new[r * nx + c] =
                    T + dt_ * rhs_wm3 / (rho * in718::cp_eff(T));
            }
        }
        }
        const auto t_stencil1 = clk::now();

        tile->data = std::move(T_new);
        this->addResult(tile);

        // Bookkeeping ------------------------------------------------------
        const auto t_call1 = clk::now();
        auto as_ns = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
                .count();
        };
        prof_->alloc_ns  .fetch_add(as_ns(t_alloc0, t_alloc1),
                                    std::memory_order_relaxed);
        prof_->stencil_ns.fetch_add(as_ns(t_alloc1, t_stencil1),
                                    std::memory_order_relaxed);
        prof_->compute_ns.fetch_add(as_ns(t_call0,  t_call1),
                                    std::memory_order_relaxed);
        prof_->n_calls   .fetch_add(1, std::memory_order_relaxed);
    }

    std::shared_ptr<hh::AbstractTask<1, HaloTile, HaloTile>>
    copy() override {
        return std::make_shared<StepTileTask>(
            this->name(), this->numberThreads(),
            mat_, pr_, dom_, dt_, prof_, phys_, kmult_);
    }

private:
    Material                     mat_;
    Process                      pr_;
    Domain                       dom_;
    double                       dt_;
    std::shared_ptr<StepProfile> prof_;
    Physics                      phys_;
    double                       kmult_;
};

}  // namespace lpbf
