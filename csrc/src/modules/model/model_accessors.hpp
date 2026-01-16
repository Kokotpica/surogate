#pragma once

// Accessors + simple methods
// Note: Container classes (AdamW8BitMomentumContainer, etc.) are defined in modular_model_fwd.h

template<typename Block>
float ModularTransformerModel<Block>::get_loss() const {
    if (!mRunState) return 0.0f;

    // Get raw loss from IRunState base class (populated by reduce_loss)
    float raw_loss = mRunState->get_loss();

    // Normalize by valid token count (similar to LLamaModel::get_loss)
    // ValidTokenCount was reduced across ranks in backward_with_hook
    int valid_tokens;
    CUDA_CHECK(cudaMemcpy(&valid_tokens, mRunState->ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost));

    if (valid_tokens > 0) {
        // ValidTokenCount is reduced across ranks (sum). Loss is reduced with ncclAvg, so
        // divide by the average valid tokens per rank for correct mean CE.
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, mRunState->WorldSize));
        return raw_loss / avg_valid;
    } else {
        return 0.0f;
    }
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::weights() {
    return *mWeights;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_momentum() {
    // Return appropriate container based on which optimizer is active
    if (mNorMuonState && mNorMuonState->initialized) {
        // NorMuon optimizer: return combined AdamW + NorMuon momentum
        if (!mNorMuonMomentumContainer) {
            mNorMuonMomentumContainer.emplace(
                &mNorMuonState->adamw_state1, &mNorMuonState->adamw_absmax1,
                &mNorMuonState->momentum_state, &mNorMuonState->momentum_absmax);
        } else {
            mNorMuonMomentumContainer->update_pointers(
                &mNorMuonState->adamw_state1, &mNorMuonState->adamw_absmax1,
                &mNorMuonState->momentum_state, &mNorMuonState->momentum_absmax);
        }
        return *mNorMuonMomentumContainer;
    } else if (mAdamW8BitState && mAdamW8BitState->initialized) {
        // 8-bit AdamW optimizer
        if (!mAdamWMomentumContainer) {
            mAdamWMomentumContainer.emplace(&mAdamW8BitState->state1, &mAdamW8BitState->absmax1);
        } else {
            mAdamWMomentumContainer->update_pointers(&mAdamW8BitState->state1, &mAdamW8BitState->absmax1);
        }
        return *mAdamWMomentumContainer;
    }
    // No optimizer state initialized
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_momentum_scales() {
    // 8-bit quantized optimizers store scales in absmax tensors (included in momentum/variance)
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_variance() {
    // Return appropriate container based on which optimizer is active
    if (mNorMuonState && mNorMuonState->initialized) {
        // NorMuon optimizer: return combined AdamW variance + NorMuon variance buffers
        if (!mNorMuonVarianceContainer) {
            mNorMuonVarianceContainer.emplace(
                &mNorMuonState->adamw_state2, &mNorMuonState->adamw_absmax2,
                &mNorMuonState->variance_buffers);
        } else {
            mNorMuonVarianceContainer->update_pointers(
                &mNorMuonState->adamw_state2, &mNorMuonState->adamw_absmax2,
                &mNorMuonState->variance_buffers);
        }
        return *mNorMuonVarianceContainer;
    } else if (mAdamW8BitState && mAdamW8BitState->initialized) {
        // 8-bit AdamW optimizer
        if (!mAdamWVarianceContainer) {
            mAdamWVarianceContainer.emplace(&mAdamW8BitState->state2, &mAdamW8BitState->absmax2);
        } else {
            mAdamWVarianceContainer->update_pointers(&mAdamW8BitState->state2, &mAdamW8BitState->absmax2);
        }
        return *mAdamWVarianceContainer;
    }
    // No optimizer state initialized
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_variance_scales() {
    // 8-bit quantized optimizers store scales in absmax tensors (included in momentum/variance)
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
std::vector<std::byte> ModularTransformerModel<Block>::rng_state() const {
    std::stringstream tmp;
    static_cast<std::ostream&>(tmp) << mOptimizerRNG;
    auto view = tmp.rdbuf()->view();
    std::vector<std::byte> state;
    state.reserve(view.size());
    std::transform(view.begin(), view.end(), std::back_inserter(state),
                   [](char c) { return static_cast<std::byte>(c); });
    return state;
}

template<typename Block>
void ModularTransformerModel<Block>::set_rng_state(const std::vector<std::byte>& state) {
    std::stringstream tmp;
    tmp.write(reinterpret_cast<const char*>(state.data()), state.size());
    static_cast<std::istream&>(tmp) >> mOptimizerRNG;
}

template<typename Block>
std::string_view ModularTransformerModel<Block>::model_type() const {
    return mConfig.model_name();
}

template<typename Block>
IRunState& ModularTransformerModel<Block>::get_run_state() const {
    if (!mRunState) {
        throw std::logic_error("ModularTransformerModel::get_run_state() called before allocate_run_state()");
    }
    // ModularRunState inherits from IRunState, so this is safe
    return *mRunState;
}

