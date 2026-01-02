// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MODULE_BASE_H
#define SUROGATE_SRC_MODULES_MODULE_BASE_H

#include "module_concept.h"

namespace modules {

/**
 * @brief CRTP base class providing common module functionality
 *
 * Derived classes must implement:
 * - forward_impl(ctx, weights, input, acts) -> Tensor
 * - backward_impl(ctx, weights, acts, grad_output, grads) -> Tensor
 *
 * Optional methods:
 * - recompute_impl(ctx, weights, input, acts) -> void
 *
 * The CRTP pattern ensures zero-overhead dispatch to derived implementations.
 * No virtual function calls occur in the hot path.
 *
 * Usage:
 * @code
 * class MyModule : public ModuleBase<MyModule> {
 * public:
 *     struct Config { ... };
 *     struct Weights { ... };
 *     struct Activations { ... };
 *     struct Gradients { ... };
 *
 *     explicit MyModule(Config config) : mConfig(config) {}
 *
 *     Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& in, Activations& acts);
 *     Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
 *                          Tensor& grad_out, Gradients& grads);
 * private:
 *     Config mConfig;
 * };
 * @endcode
 */
template<typename Derived>
class ModuleBase {
public:
    // Note: Methods are templates to defer instantiation until call time.
    // This is necessary because Derived is incomplete at class definition time,
    // and we need to access Derived::Weights, Derived::Activations, etc.
    // Using a dummy template parameter D=Derived defers lookup until the method is called.

    /**
     * @brief Forward pass through the module
     *
     * @param ctx Module context with CUDA resources
     * @param weights Module weights (received from weight manager)
     * @param input Input tensor
     * @param acts Activation storage (filled during forward for backward use)
     * @return Output tensor
     */
    template<typename D = Derived>
    Tensor forward(ModuleContext& ctx, typename D::Weights& weights,
                   Tensor& input, typename D::Activations& acts) {
        return derived().forward_impl(ctx, weights, input, acts);
    }

    /**
     * @brief Backward pass through the module
     *
     * @param ctx Module context with CUDA resources
     * @param weights Module weights (same as used in forward)
     * @param acts Saved activations from forward pass
     * @param grad_output Gradient w.r.t. output
     * @param grads Weight gradient storage (filled during backward)
     * @return Gradient w.r.t. input
     */
    template<typename D = Derived>
    Tensor backward(ModuleContext& ctx, typename D::Weights& weights,
                    typename D::Activations& acts, Tensor& grad_output,
                    typename D::Gradients& grads) {
        return derived().backward_impl(ctx, weights, acts, grad_output, grads);
    }

    /**
     * @brief Backward pass with gradient accumulation flag
     *
     * @param ctx Module context with CUDA resources
     * @param weights Module weights
     * @param acts Saved activations from forward pass
     * @param grad_output Gradient w.r.t. output
     * @param grads Weight gradient storage
     * @param accumulate If true, add to existing gradients; if false, overwrite
     * @return Gradient w.r.t. input
     */
    template<typename D = Derived>
    Tensor backward(ModuleContext& ctx, typename D::Weights& weights,
                    typename D::Activations& acts, Tensor& grad_output,
                    typename D::Gradients& grads, bool accumulate) {
        return derived().backward_impl(ctx, weights, acts, grad_output, grads, accumulate);
    }

    /**
     * @brief Recompute activations from input (for gradient checkpointing)
     *
     * Called during backward when recomputation is enabled for this module.
     * Recomputes the forward pass to regenerate activations that were not saved.
     *
     * Default implementation calls forward_impl, which works for most modules.
     * Override recompute_impl in Derived for custom behavior.
     *
     * @param ctx Module context with CUDA resources
     * @param weights Module weights
     * @param input Original input tensor (must be saved or recomputed)
     * @param acts Activation storage to fill
     */
    template<typename D = Derived>
    void recompute(ModuleContext& ctx, typename D::Weights& weights,
                   Tensor& input, typename D::Activations& acts) {
        if constexpr (requires { derived().recompute_impl(ctx, weights, input, acts); }) {
            derived().recompute_impl(ctx, weights, input, acts);
        } else {
            // Default: just run forward again
            derived().forward_impl(ctx, weights, input, acts);
        }
    }

protected:
    ModuleBase() = default;
    ~ModuleBase() = default;

    // Non-copyable, non-movable (modules are lightweight config holders)
    ModuleBase(const ModuleBase&) = default;
    ModuleBase& operator=(const ModuleBase&) = default;
    ModuleBase(ModuleBase&&) = default;
    ModuleBase& operator=(ModuleBase&&) = default;

private:
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

/**
 * @brief Helper to check if a module supports recomputation
 */
template<typename Module>
concept SupportsRecompute = requires(Module m,
                                      ModuleContext& ctx,
                                      typename Module::Weights& w,
                                      Tensor& input,
                                      typename Module::Activations& acts) {
    { m.recompute(ctx, w, input, acts) } -> std::same_as<void>;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MODULE_BASE_H
