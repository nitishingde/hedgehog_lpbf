#ifndef HEDGEHOG_LPBF_TASKS_H
#define HEDGEHOG_LPBF_TASKS_H

#include <hedgehog.h>

#include "kokkos_kernels.h"

namespace hh {
    template<KokkosExecutionSpace ExecutionSpace, size_t Separator, class ...AllTypes>
    class AbstractKokkosTask: public hh::AbstractTask<Separator, AllTypes...> {
    public:
        explicit AbstractKokkosTask(std::string const &name, size_t numberThreads, bool automaticStart = false):
            AbstractTask<Separator, AllTypes...>(name, numberThreads, automaticStart) {
            if constexpr(std::is_same_v<ExecutionSpace, Kokkos::Serial>) {
                this->coreTask()->printOptions().background({0x2f, 0xf7, 0xdb, 0xff});
                this->coreTask()->printOptions().font({0xff, 0xff, 0xff, 0xff});
            }

#ifdef KOKKOS_ENABLE_CUDA
            if constexpr(std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
                this->coreTask()->printOptions().background({0x76, 0xb9, 0x00, 0xff});
                this->coreTask()->printOptions().font({0xff, 0xff, 0xff, 0xff});
            }
#endif //KOKKOS_ENABLE_CUDA

#ifdef KOKKOS_ENABLE_HIP
            if constexpr(std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
                this->coreTask()->printOptions().background({0xed, 0x64, 0x27, 0xff});
                this->coreTask()->printOptions().font({0xff, 0xff, 0xff, 0xff});
            }
#endif //KOKKOS_ENABLE_HIP
        }

        explicit AbstractKokkosTask(std::shared_ptr<hh::core::CoreTask<Separator, AllTypes...>> coreTask, bool enablePeerAccess):
            AbstractTask<Separator, AllTypes...>(std::shared_ptr<hh::core::CoreTask<Separator, AllTypes...>>(coreTask)) {
            if constexpr(std::is_same_v<ExecutionSpace, Kokkos::Serial>) {
                this->coreTask()->printOptions().background({0x2f, 0xf7, 0xdb, 0xff});
                this->coreTask()->printOptions().font({0xff, 0xff, 0xff, 0xff});
            }

#ifdef KOKKOS_ENABLE_CUDA
            if constexpr(std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
                this->coreTask()->printOptions().background({0x76, 0xb9, 0x00, 0xff});
                this->coreTask()->printOptions().font({0xff, 0xff, 0xff, 0xff});
            }
#endif //KOKKOS_ENABLE_CUDA

#ifdef KOKKOS_ENABLE_HIP
            if constexpr(std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
                this->coreTask()->printOptions().background({0xed, 0x64, 0x27, 0xff});
                this->coreTask()->printOptions().font({0xff, 0xff, 0xff, 0xff});
            }
#endif //KOKKOS_ENABLE_HIP
        }

        /// @brief Default destructor
        ~AbstractKokkosTask() override = default;

        void initialize() override {
            executionSpace_ = createExecutionSpace<ExecutionSpace>(this->deviceId());
            initializeKokkos();
        }

        void shutdown() override {
            shutdownKokkos();
        }

        /// @brief Virtual initialization step, where user defined data structure can be initialized.
        virtual void initializeKokkos() {}

        /// @brief Virtual shutdown step, where user defined data structure can be destroyed.
        virtual void shutdownKokkos() {}

        /// @brief Getter for Kokkos task's execution space
        /// @return Kokkos::ExecutionSpace
        [[nodiscard]] ExecutionSpace executionSpace() { return executionSpace_; }

        /// @brief Fences executionSpace previous operations
        void fence() { executionSpace_.fence(); }

    protected:
        ExecutionSpace executionSpace_ = {}; ///< Execution Space
    };
}

namespace lpbf {
    template<std::floating_point Float, KokkosMemorySpace MemorySpace>
    struct InputData {
        Kokkos::View<Float*,  MemorySpace> expX;
        Kokkos::View<Float*,  MemorySpace> expY;
        Kokkos::View<Float**, MemorySpace> currentField;
        Kokkos::View<Float**, MemorySpace> nextField;
        Float                              dt;
        Domain<Float>                      domain;
        Process<Float>                     process;
        StepConst<Float>                   stepConst;
        Float                              laserX;
        Float                              laserY;
        Float                              laserStartX;
        Float                              laserStartY;
        Physics                            mode;
    };

    template<std::floating_point Float, KokkosExecutionSpace ExecutionSpace, KokkosMemorySpace MemorySpace = ExecutionSpace::memory_space, KokkosView View = Kokkos::View<Float**, MemorySpace>>
    class FdmTask final: public hh::AbstractKokkosTask<ExecutionSpace, 1, InputData<Float, MemorySpace>, View, InputData<Float, MemorySpace>> {
    public:
        using base = hh::AbstractKokkosTask<ExecutionSpace, 1, InputData<Float, MemorySpace>, View, InputData<Float, MemorySpace>>;

        explicit FdmTask(std::shared_ptr<AutoReleaseMemoryPool<View>> pool):
            base("FDM", 1, false), pool_(pool) {}

        void execute(std::shared_ptr<InputData<Float, MemorySpace>> data) override {
            const auto expX      = data->expX;
            const auto expY      = data->expY;
            const auto inpT      = data->currentField;
            const auto outT      = data->nextField;
            const auto stepConst = data->stepConst;
            const auto x_laser   = data->laserX;
            const auto NX        = inpT.extent(0);
            const auto NY        = inpT.extent(1);

            Kokkos::parallel_for("expXTable", Kokkos::RangePolicy(this->executionSpace(), 0, NX-2), KOKKOS_LAMBDA(const int i) {
                const auto xPos = static_cast<Float>(i) * stepConst.dx;
                expX(i+1) = Kokkos::exp(-2.0 * square(xPos - x_laser) * stepConst.inv_r0_sq);
            });

            if(data->mode == Physics::Baseline) {
                Kokkos::parallel_for("FDM.1", Kokkos::MDRangePolicy(this->executionSpace(), {1, 1}, {NX-1, NY-1}), KOKKOS_LAMBDA(const int32_t i, const int32_t j) {
                    const auto T   = inpT(i,   j);
                    const auto Txm = inpT(i-1, j);
                    const auto Txp = inpT(i+1, j);
                    const auto Tym = inpT(i,   j-1);
                    const auto Typ = inpT(i,   j+1);
                    const auto lap_x = (Txp - 2.0*T + Txm) * stepConst.inv_dx2;
                    const auto lap_y = (Typ - 2.0*T + Tym) * stepConst.inv_dy2;
                    const auto S     = stepConst.q_scale * stepConst.peak_S * expX(i) * expY(j) / (stepConst.rho_cp * stepConst.h);
                    auto       q_rad = 0.0;
                    if(stepConst.radiation) {
                        q_rad = stepConst.eps * sigma_sb() * (quad(T) - quad(stepConst.Tamb));
                    }
                    const auto q_top = stepConst.hConv * (T - stepConst.Tamb) + q_rad;
                    const auto q_bot = (stepConst.K / stepConst.dSub) * (T - stepConst.Tsub);
                    const auto loss  = (q_top + q_bot) * stepConst.inv_h / stepConst.rho_cp;
                    const auto dTdt  = stepConst.alpha * (lap_x + lap_y) + S - loss;

                    outT(i, j) = T + stepConst.dt * dTdt;
                });
            }
            else if(data->mode == Physics::Extended) {
                Kokkos::parallel_for("FDM.2", Kokkos::MDRangePolicy(this->executionSpace(), {1, 1}, {NX-1, NY-1}), KOKKOS_LAMBDA(const int32_t i, const int32_t j) {
                    const auto T   = inpT(i,   j);
                    const auto Txm = inpT(i-1, j);
                    const auto Txp = inpT(i+1, j);
                    const auto Tym = inpT(i,   j-1);
                    const auto Typ = inpT(i,   j+1);
                    const auto kCe = effectiveThermalConductivity(T, stepConst.kmult);
                    const auto kxm = 0.5 * (kCe + effectiveThermalConductivity(Txm, stepConst.kmult));
                    const auto kxp = 0.5 * (kCe + effectiveThermalConductivity(Txp, stepConst.kmult));
                    const auto kym = 0.5 * (kCe + effectiveThermalConductivity(Tym, stepConst.kmult));
                    const auto kyp = 0.5 * (kCe + effectiveThermalConductivity(Typ, stepConst.kmult));
                    const auto div_k_grad = 0.0
                        + (kxp*(Txp-T) - kxm*(T-Txm)) * stepConst.inv_dx2
                        + (kyp*(Typ-T) - kym*(T-Tym)) * stepConst.inv_dy2;
                    const auto q_in = stepConst.q_scale * stepConst.peak_S * expX(i) * expY(j);

                    auto q_rad = 0.0;
                    if(stepConst.radiation) {
                        q_rad = stepConst.eps * sigma_sb() * (quad(T) - quad(stepConst.Tamb));
                    }
                    const auto q_top = stepConst.hConv * (T - stepConst.Tamb) + q_rad;
                    const auto q_bot = (thermalConductivity(T) / stepConst.dSub) * (T - stepConst.Tsub);
                    const auto rhs   = div_k_grad + (q_in - q_top - q_bot) * stepConst.inv_h;

                    outT(i, j) = T + stepConst.dt * rhs / (stepConst.rho * effectiveSpecificHeatCapacity(T));
                });
            }
            else {
                using namespace std::string_literals;
                // throw std::runtime_error("Physics mode ["s + std::to_string(data->mode) + "]  not supported!");
                throw std::runtime_error("Physics mode not supported!");
            }

            Kokkos::parallel_for("BND_VERTICAL", Kokkos::RangePolicy(this->executionSpace(), 0, NY), KOKKOS_LAMBDA(const int32_t j) {
                outT(   0, j) = outT(   1, j);
                outT(NX-1, j) = outT(NX-2, j);
            });

            Kokkos::parallel_for("BND_HORIZONTAL", Kokkos::RangePolicy(this->executionSpace(), 0, NX), KOKKOS_LAMBDA(const int32_t i) {
                const int32_t isLeft  = (i == 0);
                const int32_t isRight = (i == (NX - 1));
                const auto    ri      = isLeft*1 + isRight*(NX-2) + (1-isLeft)*(1-isRight)*i;

                outT(i, 0)    = outT(ri, 1);
                outT(i, NY-1) = outT(ri, NY-2);
            });

            const auto field = pool_->allocate(MemoryManagerAllocateMode::Wait);
            Kokkos::deep_copy(this->executionSpace(), *field, outT);

            this->fence();
            this->addResult(field);
            this->addResult(data);
        }

        [[nodiscard]] std::string extraPrintingInformation() const override {
            return pool_->extraPrintingInformation();
        }

    private:
        std::shared_ptr<AutoReleaseMemoryPool<View>> pool_ = nullptr;
    };

    template<KokkosExecutionSpace ExecutionSpace, KokkosView View, std::floating_point Float = View::value_type>
    class ReducerTask final: public hh::AbstractKokkosTask<ExecutionSpace, 1, View, StepMetrics<Float>> {
    public:
        using base = hh::AbstractKokkosTask<ExecutionSpace, 1, View, StepMetrics<Float>>;

        explicit ReducerTask(const Float Tm, const int32_t threads): base("ReducerTask", threads, false), tm_(Tm) {}

        void execute(std::shared_ptr<View> field) override {
            const auto tempField = *field;
            auto       results   = ThermalReduction<Float>{};
            const auto NX        = tempField.extent(0);
            const auto NY        = tempField.extent(1);
            const auto Tm        = this->tm_;
            Kokkos::parallel_reduce("ThermalAnalysis", Kokkos::MDRangePolicy(this->executionSpace(), {0, 0}, {NX, NY}),
                KOKKOS_LAMBDA(const int32_t x, const int32_t y, ThermalReduction<Float> &local) {
                    const auto temperature = tempField(x, y);

                    if(local.Tmax < temperature) {
                        local.Tmax  = temperature;
                        local.x_hot = x;
                    }
                    if(Tm < temperature) {
                        local.y_min = Kokkos::min(local.y_min, y);
                        local.y_max = Kokkos::max(local.y_max, y);
                    }
                },
                ThermalReducer(results)
            );

            this->fence();
            this->addResult(std::make_shared<StepMetrics<Float>>(
                results.Tmax,
                (results.y_max < 0 or results.y_min == 1e9) ? Float{0} : Float((results.y_max - results.y_min + 1) * domain_.dy() * 1e6),
                Float{0},//TODO
                results.x_hot
            ));
        }

        [[nodiscard]] std::shared_ptr<hh::AbstractTask<1, View, StepMetrics<Float>>> copy() override {
            return std::make_shared<ReducerTask>(this->tm_, this->numberThreads());
        }

    private:
        Float         tm_     = {};
        Domain<Float> domain_ = {};
    };

    template<std::floating_point Float, class Inp0 = InputData<Float, Kokkos::DefaultExecutionSpace::memory_space>>
    class FdmStateTask: public hh::AbstractTask<1, Inp0, Inp0> {
    public:
        using base = hh::AbstractTask<1, Inp0, Inp0>;

        explicit FdmStateTask(const int32_t timeSteps):
            base("State", 1, false), timeSteps_(timeSteps) {}

        void execute(std::shared_ptr<Inp0> data) override {
            timeSteps_--;
            if(timeSteps_ == 0) {
                canTerminate_.store(true);
                return;
            }

            data->laserX += data->process.V*data->dt;
            if(0.90*data->domain.Lx < data->laserX) {
                data->laserX = data->laserStartX;
            }
            std::swap(data->currentField, data->nextField);
            this->addResult(data);
        }

        [[nodiscard]] bool canTerminate() const override {
            return canTerminate_.load();
        }

    private:
        std::atomic_bool canTerminate_ = false;
        int32_t          timeSteps_    = {};
    };
}

#endif //HEDGEHOG_LPBF_TASKS_H
