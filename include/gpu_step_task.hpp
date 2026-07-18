// SPDX-License-Identifier: 0BSD
// ---------------------------------------------------------------------------
// The three Hedgehog nodes of the Hedgehog+CUDA LPBF graph.  Compiled by g++
// at C++20 (this header is included only from src/gpu/hh_gpu_main.cpp).
//
//   GpuStepTask        : hh::AbstractCUDATask<1, Tick, Tick, MetricsMsg, SnapshotMsg>
//                        Owns the device double-buffer + reduction scratch and
//                        the per-task CUDA stream.  Advances one explicit-Euler
//                        step on the GPU, reduces for peak-T/melt metrics, and
//                        RECIRCULATES the next Tick back to itself (self-cycle),
//                        emitting a MetricsMsg every step and a SnapshotMsg on
//                        dump steps.
//   MetricsTask        : hh::AbstractTask<1, MetricsMsg, StepResult>   (1 thread)
//                        Finalises melt-width, keeps running maxima in deterministic
//                        step order, owns metrics.csv.
//   SnapshotWriterTask : hh::AbstractTask<1, SnapshotMsg, StepResult>  (N threads)
//                        Writes T_step_%06d.bin off the GPU's critical path.
//
// No device tokens appear here: all kernel work goes through the extern "C"
// launch wrappers in lpbf_kernels.hpp (defined in the nvcc TU kernels.cu).
// ---------------------------------------------------------------------------
#pragma once

#include <hedgehog.h>   // needs HH_USE_CUDA defined -> AbstractCUDATask
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "params.hpp"
#include "lpbf_step_const.hpp"
#include "lpbf_kernels.hpp"
#include "gpu_messages.hpp"

namespace lpbf {

// Running maxima shared with main() (read after waitForTermination); written
// only by the single-threaded MetricsTask, so no synchronisation is required.
struct RunAccum {
    double peak_T = 0.0;
    double peak_W = 0.0;
    double steady_W_sub = 0.0;
    std::int64_t steps_seen = 0;
};

// ===========================================================================
// GPU step task — the recirculating device node.
// ===========================================================================
class GpuStepTask
    : public hh::AbstractCUDATask<1, Tick, Tick, MetricsMsg, SnapshotMsg> {
public:
    // mode: 0 = LEGACY per-step baseline (host finalize, ungated per-step hot-col
    //           strided D2H + dedicated syncs — reproduces the pre-optimization
    //           reference generator);
    //       1 = OPT #1 only (per-step, host finalize, cruise-GATED hot-col);
    //       2 = OPT #1 + #2 batched device-ring path (default; --batch=K).
    // fused: OPT #3 — fuse step+reduce into one grid launch (mode 2 only).
    // pinned: OPT #4 — pinned host staging for the per-batch metric D2H.
    GpuStepTask(Domain dom, StepConst k, int phys_i, double Tm,
                int n_steps, std::vector<Tick> plan, double Tamb,
                bool write_outputs, int batch,
                int mode = 2, bool fused = false, bool pinned = false)
        : hh::AbstractCUDATask<1, Tick, Tick, MetricsMsg, SnapshotMsg>(
              "GPU-Step", /*numberThreads=*/1),
          dom_(dom), k_(k), phys_i_(phys_i), Tm_(Tm), n_steps_(n_steps),
          plan_(std::move(plan)), Tamb_(Tamb), write_outputs_(write_outputs),
          batch_(batch < 1 ? 1 : batch),
          mode_(mode), fused_(fused), pinned_(pinned) {
        nx_ = dom_.nx; ny_ = dom_.ny;
        N_  = static_cast<std::size_t>(nx_) * ny_;
        nblocks_ = static_cast<int>(
            std::min<long long>(256, ((long long)N_ + 255) / 256));
        // OPT #3 fused kernel emits one Partial per 16x16 block.
        nblocks_fused_ = ((nx_ + 15) / 16) * ((ny_ + 15) / 16);
    }

    // Device allocation + initial field upload (device is already set + stream
    // created by AbstractCUDATask::initialize()).
    void initializeCuda() override {
        checkCudaErrors(cudaMalloc(&d_A_, N_ * sizeof(double)));
        checkCudaErrors(cudaMalloc(&d_B_, N_ * sizeof(double)));
        // Sized for the larger of the two-kernel (nblocks_) and fused
        // (nblocks_fused_) per-block Partial counts.
        checkCudaErrors(cudaMalloc(&d_part_,
                        std::max(nblocks_, nblocks_fused_) * sizeof(Partial)));
        // OPT #2 per-batch device rings: one finalized Partial and one hot
        // column per logical step in the batch, drained with a single D2H.
        checkCudaErrors(cudaMalloc(&d_glob_ring_, batch_ * sizeof(Partial)));
        checkCudaErrors(cudaMalloc(&d_col_ring_,
                                   (std::size_t)batch_ * ny_ * sizeof(double)));
        // Host staging for the per-batch metric D2H.  OPT #4: pin it so the
        // ~60 KB column ring copy uses a single DMA-mapped transfer.
        if (pinned_) {
            checkCudaErrors(cudaHostAlloc(&h_glob_p_, batch_ * sizeof(Partial),
                                          cudaHostAllocDefault));
            checkCudaErrors(cudaHostAlloc(&h_col_p_,
                            (std::size_t)batch_ * ny_ * sizeof(double),
                            cudaHostAllocDefault));
        } else {
            h_glob_.resize(batch_);
            h_col_.resize((std::size_t)batch_ * ny_);
            h_glob_p_ = h_glob_.data();
            h_col_p_  = h_col_.data();
        }
        // Legacy per-step path stages the raw per-block partials here (pageable,
        // matching the pre-optimization reference generator).
        h_part_.resize(nblocks_);
        std::vector<double> init(N_, Tamb_);
        checkCudaErrors(cudaMemcpy(d_A_, init.data(), N_ * sizeof(double),
                                   cudaMemcpyHostToDevice));
        d_cur_ = d_A_; d_nxt_ = d_B_;
    }

    void shutdownCuda() override {
        cudaFree(d_A_); cudaFree(d_B_); cudaFree(d_part_);
        cudaFree(d_glob_ring_); cudaFree(d_col_ring_);
        if (pinned_) { cudaFreeHost(h_glob_p_); cudaFreeHost(h_col_p_); }
    }

    // OPT #2 — TICK BATCHING.  One incoming Tick triggers a batch of up to
    // batch_ logical steps: the incoming tick->step is the batch's first step;
    // all per-step laser params come from plan_.  Within the batch the GPU runs
    // step+reduce+device-finalize (and, on cruise steps, hot-column extraction)
    // back to back on the private stream with NO host synchronization, staging
    // each step's finalized Partial + hot column into per-step device rings.
    // The batch is drained with ONE bulk D2H + ONE stream sync, after which a
    // MetricsMsg is emitted PER LOGICAL STEP (in step order) so metrics.csv /
    // meta.json are byte-for-byte unchanged.  The self-cycle recirculates the
    // NEXT batch's first Tick, so the graph now processes ceil(n_steps/batch_)
    // elements instead of n_steps.
    void execute(std::shared_ptr<Tick> tick) override {
        if (mode_ == 0) { executePerStep(tick, /*gated=*/false); return; }
        if (mode_ == 1) { executePerStep(tick, /*gated=*/true);  return; }
        // mode 2: OPT #1 + #2 batched device-ring path (below).
        const int start = tick->step;
        const int end   = std::min(start + batch_, n_steps_);  // exclusive
        const int bn    = end - start;

        // Snapshots captured mid-batch (dump steps are rare: <=16 total).  Each
        // gets its own host buffer so the async copies never alias; all complete
        // at the single batch-end sync.
        struct Pend { int step; std::shared_ptr<std::vector<double>> field; };
        std::vector<Pend> pend;

        for (int i = 0; i < bn; ++i) {
            const Tick& tk = plan_[start + i];
            // --- advance one explicit-Euler step on the device -------------
            k_.x_l = tk.x_l; k_.y_l = tk.y_l; k_.q_scale = tk.q_scale;
            if (fused_) {
                // OPT #3: one fused grid produces the field AND the per-block
                // partials; then a device finalize -> ring[i].
                lpbf_launch_step_reduce(d_cur_, d_nxt_, &k_, phys_i_, Tm_,
                                        d_part_, this->stream());
                checkCudaErrors(cudaGetLastError());
                std::swap(d_cur_, d_nxt_);   // d_cur_ now holds the updated field
                lpbf_launch_reduce_finalize(d_part_, nblocks_fused_, nx_, ny_,
                                            d_glob_ring_ + i, this->stream());
                checkCudaErrors(cudaGetLastError());
            } else {
                lpbf_launch_step(d_cur_, d_nxt_, &k_, phys_i_, this->stream());
                checkCudaErrors(cudaGetLastError());
                std::swap(d_cur_, d_nxt_);   // d_cur_ now holds the updated field

                // --- reduce + device-side cross-block finalize -> ring[i] ---
                lpbf_launch_reduce(d_cur_, nx_, ny_, Tm_, d_part_, nblocks_,
                                   this->stream());
                checkCudaErrors(cudaGetLastError());
                lpbf_launch_reduce_finalize(d_part_, nblocks_, nx_, ny_,
                                            d_glob_ring_ + i, this->stream());
                checkCudaErrors(cudaGetLastError());
            }

            // --- OPT #1 cruise gate: extract hot column only when consumed --
            const bool cruise = !tk.wrapped
                             && tk.x_laser_csv >= 0.45 * dom_.Lx
                             && tk.x_laser_csv <= 0.85 * dom_.Lx;
            if (cruise) {
                lpbf_launch_extract_col(d_cur_, nx_, ny_, d_glob_ring_ + i,
                                        d_col_ring_ + (std::size_t)i * ny_,
                                        this->stream());
                checkCudaErrors(cudaGetLastError());
            }

            // --- full-field snapshot on dump steps (off the critical path) --
            if (write_outputs_ && tk.dump) {
                auto field = std::make_shared<std::vector<double>>(N_);
                checkCudaErrors(cudaMemcpyAsync(field->data(), d_cur_,
                                                N_ * sizeof(double),
                                                cudaMemcpyDeviceToHost,
                                                this->stream()));
                pend.push_back({tk.step, std::move(field)});
            }
        }

        // --- ONE bulk D2H of the batch metric rings + ONE sync -------------
        checkCudaErrors(cudaMemcpyAsync(h_glob_p_, d_glob_ring_,
                                        bn * sizeof(Partial),
                                        cudaMemcpyDeviceToHost, this->stream()));
        checkCudaErrors(cudaMemcpyAsync(h_col_p_, d_col_ring_,
                                        (std::size_t)bn * ny_ * sizeof(double),
                                        cudaMemcpyDeviceToHost, this->stream()));
        checkCudaErrors(cudaStreamSynchronize(this->stream()));

        // --- emit one MetricsMsg per logical step, in step order -----------
        for (int i = 0; i < bn; ++i) {
            const Tick& tk = plan_[start + i];
            const Partial& g = h_glob_p_[i];
            auto mm = std::make_shared<MetricsMsg>();
            mm->step = tk.step; mm->t = tk.t;
            mm->x_laser_csv = tk.x_laser_csv; mm->wrapped = tk.wrapped;
            mm->dump = tk.dump;
            mm->tmax = g.tmax; mm->x_hot = (int)(g.arg % nx_);
            mm->ymin = g.ymin; mm->ymax = g.ymax;
            const bool cruise = !tk.wrapped
                             && tk.x_laser_csv >= 0.45 * dom_.Lx
                             && tk.x_laser_csv <= 0.85 * dom_.Lx;
            if (g.tmax > Tm_ && cruise) {
                const double* src = h_col_p_ + (std::size_t)i * ny_;
                mm->hot_col.assign(src, src + ny_);
            }
            this->addResult(mm);
        }
        for (auto& p : pend) {
            auto sm = std::make_shared<SnapshotMsg>();
            sm->step = p.step; sm->field = std::move(p.field);
            this->addResult(sm);
        }

        // --- recirculate the NEXT batch's first Tick back to ourselves -----
        if (end < n_steps_) {
            this->addResult(std::make_shared<Tick>(plan_[end]));
        } else {
            done_.store(true, std::memory_order_release);
        }
    }

    // -----------------------------------------------------------------------
    // LEGACY per-step path (ledger baseline / OPT #1 isolation).  Reproduces the
    // pre-optimization reference generator EXACTLY: step -> reduce -> pageable
    // Partial D2H -> host cross-block finalize -> per-step MetricsMsg, with a
    // per-step strided hot-column 2D D2H + a dedicated stream sync.
    //   gated == false : hot column copied whenever molten (tmax>Tm)  -> BASELINE
    //   gated == true  : hot column copied only on cruise steps        -> OPT #1
    // Same kernels + same reductions as every other mode, so metrics.csv and the
    // final T field are byte-identical across modes.
    // -----------------------------------------------------------------------
    void executePerStep(std::shared_ptr<Tick> tick, bool gated) {
        const Tick& tk = plan_[tick->step];
        k_.x_l = tk.x_l; k_.y_l = tk.y_l; k_.q_scale = tk.q_scale;
        lpbf_launch_step(d_cur_, d_nxt_, &k_, phys_i_, this->stream());
        checkCudaErrors(cudaGetLastError());
        std::swap(d_cur_, d_nxt_);

        lpbf_launch_reduce(d_cur_, nx_, ny_, Tm_, d_part_, nblocks_,
                           this->stream());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpyAsync(h_part_.data(), d_part_,
                                        nblocks_ * sizeof(Partial),
                                        cudaMemcpyDeviceToHost, this->stream()));
        checkCudaErrors(cudaStreamSynchronize(this->stream()));   // sync #1

        // Host cross-block finalize (verbatim reduction order + tie-break).
        double    tmax = -1.0;
        long long arg  = (long long)nx_ * ny_;
        int       ymin = ny_, ymax = -1;
        for (int j = 0; j < nblocks_; ++j) {
            const Partial& p = h_part_[j];
            if (p.tmax > tmax || (p.tmax == tmax && p.arg < arg)) {
                tmax = p.tmax; arg = p.arg;
            }
            if (p.ymin < ymin) ymin = p.ymin;
            if (p.ymax > ymax) ymax = p.ymax;
        }

        auto mm = std::make_shared<MetricsMsg>();
        mm->step = tk.step; mm->t = tk.t;
        mm->x_laser_csv = tk.x_laser_csv; mm->wrapped = tk.wrapped;
        mm->dump = tk.dump;
        mm->tmax = tmax; mm->x_hot = (int)(arg % nx_);
        mm->ymin = ymin; mm->ymax = ymax;

        const bool cruise = !tk.wrapped
                         && tk.x_laser_csv >= 0.45 * dom_.Lx
                         && tk.x_laser_csv <= 0.85 * dom_.Lx;
        if (tmax > Tm_ && (!gated || cruise)) {
            mm->hot_col.resize(ny_);
            const int x_hot = (int)(arg % nx_);
            checkCudaErrors(cudaMemcpy2DAsync(
                mm->hot_col.data(), sizeof(double),
                d_cur_ + x_hot, (std::size_t)nx_ * sizeof(double),
                sizeof(double), ny_,
                cudaMemcpyDeviceToHost, this->stream()));
            checkCudaErrors(cudaStreamSynchronize(this->stream()));  // sync #2
        }
        this->addResult(mm);

        if (write_outputs_ && tk.dump) {
            auto field = std::make_shared<std::vector<double>>(N_);
            checkCudaErrors(cudaMemcpyAsync(field->data(), d_cur_,
                                            N_ * sizeof(double),
                                            cudaMemcpyDeviceToHost,
                                            this->stream()));
            checkCudaErrors(cudaStreamSynchronize(this->stream()));  // sync #3
            auto sm = std::make_shared<SnapshotMsg>();
            sm->step = tk.step; sm->field = std::move(field);
            this->addResult(sm);
        }

        if (tk.step + 1 < n_steps_)
            this->addResult(std::make_shared<Tick>(plan_[tk.step + 1]));
        else
            done_.store(true, std::memory_order_release);
    }

    // A self-cycle deadlocks under the default rule (the self-edge is always a
    // connected notifier), so terminate explicitly once the final step is done
    // and our input queue has drained.
    [[nodiscard]] bool canTerminate() const override {
        return done_.load(std::memory_order_acquire)
            && this->coreTask()->receiversEmpty();
    }

private:
    Domain      dom_;
    StepConst   k_;
    int         phys_i_;
    double      Tm_;
    int         n_steps_;
    std::vector<Tick> plan_;
    double      Tamb_;
    bool        write_outputs_;
    int         batch_ = 1;
    int         mode_  = 2;        // 0 legacy baseline, 1 opt1, 2 batched
    bool        fused_ = false;    // OPT #3
    bool        pinned_ = false;   // OPT #4

    int         nx_ = 0, ny_ = 0, nblocks_ = 0, nblocks_fused_ = 0;
    std::size_t N_  = 0;
    double     *d_A_ = nullptr, *d_B_ = nullptr;
    double     *d_cur_ = nullptr, *d_nxt_ = nullptr;
    Partial    *d_part_ = nullptr;
    std::vector<Partial> h_part_;         // legacy per-step pageable staging
    // OPT #2 per-batch rings (device) + their host staging (drained per batch).
    Partial    *d_glob_ring_ = nullptr;   // one finalized Partial per step
    double     *d_col_ring_  = nullptr;   // one hot column per step
    std::vector<Partial> h_glob_;         // pageable staging (when !pinned_)
    std::vector<double>  h_col_;
    Partial    *h_glob_p_ = nullptr;      // -> h_glob_.data() or pinned alloc
    double     *h_col_p_  = nullptr;      // -> h_col_.data()  or pinned alloc
    std::atomic<bool> done_{false};
};

// ===========================================================================
// Metrics task — ordered finalisation + metrics.csv (single thread).
// ===========================================================================
class MetricsTask : public hh::AbstractTask<1, MetricsMsg, StepResult> {
public:
    MetricsTask(Domain dom, double Tm, std::shared_ptr<RunAccum> acc,
                bool write_outputs, std::string snap_dir)
        : hh::AbstractTask<1, MetricsMsg, StepResult>("Metrics", 1),
          dom_(dom), Tm_(Tm), acc_(std::move(acc)),
          write_outputs_(write_outputs), snap_dir_(std::move(snap_dir)) {}

    void initialize() override {
        if (write_outputs_) {
            log_.open(snap_dir_ + "/metrics.csv");
            log_ << "step,t_us,x_laser_um,peakT_K,W_um\n";
        }
    }

    void execute(std::shared_ptr<MetricsMsg> m) override {
        const double dy = dom_.dy();
        const double W_um = (m->ymax < 0) ? 0.0
                          : (m->ymax - m->ymin + 1) * dy * 1e6;

        // Sub-grid melt width through the hottest column (measure_gpu:411-419).
        double W_sub_um = 0.0;
        if (m->tmax > Tm_ && !m->hot_col.empty()) {
            const std::vector<double>& col = m->hot_col;
            int a = -1, b = -1;
            for (int y = 0; y < dom_.ny; ++y)
                if (col[y] > Tm_) { if (a < 0) a = y; b = y; }
            if (a >= 0) {
                double y_lo = a * dy, y_hi = b * dy;
                if (a > 0)
                    y_lo = ((a - 1) + (Tm_ - col[a-1]) / (col[a] - col[a-1])) * dy;
                if (b < dom_.ny - 1)
                    y_hi = (b + (Tm_ - col[b]) / (col[b+1] - col[b])) * dy;
                W_sub_um = (y_hi - y_lo) * 1e6;
            }
        }

        // Running maxima (same gates as the standalone GPU / CPU solvers).
        if (m->tmax > acc_->peak_T) acc_->peak_T = m->tmax;
        if (W_um    > acc_->peak_W) acc_->peak_W = W_um;
        const bool cruise = !m->wrapped
                         && m->x_laser_csv >= 0.45 * dom_.Lx
                         && m->x_laser_csv <= 0.85 * dom_.Lx;
        if (cruise && W_sub_um > acc_->steady_W_sub) acc_->steady_W_sub = W_sub_um;
        acc_->steps_seen++;

        if (write_outputs_ && m->dump) {
            log_ << m->step << ',' << m->t * 1e6 << ',' << m->x_laser_csv * 1e6
                 << ',' << m->tmax << ',' << W_um << '\n';
        }
        this->addResult(std::make_shared<StepResult>(StepResult{m->step}));
    }

    void shutdown() override { if (log_.is_open()) log_.close(); }

private:
    Domain                    dom_;
    double                    Tm_;
    std::shared_ptr<RunAccum> acc_;
    bool                      write_outputs_;
    std::string               snap_dir_;
    std::ofstream             log_;
};

// ===========================================================================
// Snapshot writer — parallel file I/O off the GPU path.
// ===========================================================================
class SnapshotWriterTask : public hh::AbstractTask<1, SnapshotMsg, StepResult> {
public:
    SnapshotWriterTask(std::size_t n_threads, std::string snap_dir)
        : hh::AbstractTask<1, SnapshotMsg, StepResult>("SnapshotWriter", n_threads),
          snap_dir_(std::move(snap_dir)) {}

    void execute(std::shared_ptr<SnapshotMsg> s) override {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s/T_step_%06d.bin",
                      snap_dir_.c_str(), s->step);
        std::ofstream out(buf, std::ios::binary);
        out.write(reinterpret_cast<const char*>(s->field->data()),
                  (std::streamsize)(s->field->size() * sizeof(double)));
        this->addResult(std::make_shared<StepResult>(StepResult{s->step}));
    }

    std::shared_ptr<hh::AbstractTask<1, SnapshotMsg, StepResult>> copy() override {
        return std::make_shared<SnapshotWriterTask>(this->numberThreads(), snap_dir_);
    }

private:
    std::string snap_dir_;
};

}  // namespace lpbf
