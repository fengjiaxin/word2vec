//
// Created by fengjiaxin on 2023/5/6.
//


#include "model.h"
#include "loss.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>

namespace word2vec {

Model::State::State(int32_t hiddenSize, int32_t outputSize)
        : lossValue_(0.0),
          nexamples_(0),
          hidden(hiddenSize),
          output(outputSize),
          grad(hiddenSize) {}

real Model::State::getLoss() const {
    return lossValue_ / nexamples_;
}

void Model::State::incrementNExamples(real loss) {
    lossValue_ += loss;
    ++nexamples_;
}

Model::Model(
        std::shared_ptr<Matrix> wi,
        std::shared_ptr<Matrix> wo,
        std::shared_ptr<Loss> loss)
        : wi_(wi), wo_(wo), loss_(loss) {}

void Model::computeHidden(const std::vector<int32_t> &input, State &state)
const {
    Vector &hidden = state.hidden;
    hidden.zero();
    for (auto it = input.cbegin(); it != input.cend(); ++it) {
        hidden.addRow(*wi_, *it);
    }
    hidden.mul(1.0 / input.size());
}

void Model::predict(
        const std::vector<int32_t> &input,
        int32_t k,
        real threshold,
        Predictions &heap,
        State &state) const {
    if (k == -1) {
        k = wo_->size(0); // output size
    } else if (k <= 0) {
        throw std::invalid_argument("k needs to be 1 or higher!");
    }
    heap.reserve(k + 1);
    computeHidden(input, state);

    loss_->predict(k, threshold, heap, state);
}

void Model::update(
        const std::vector<int32_t> &input,
        const std::vector<int32_t> &targets,
        int32_t targetIndex,
        real lr,
        State &state) {
    if (input.size() == 0) {
        return;
    }
    computeHidden(input, state);

    Vector &grad = state.grad;
    grad.zero();
    real lossValue = loss_->forward(targets, targetIndex, state, lr, true);
    state.incrementNExamples(lossValue);

    for (auto it = input.cbegin(); it != input.cend(); ++it) {
        wi_->addVectorToRow(grad, *it, 1.0);
    }
}

} // namespace word2vec
