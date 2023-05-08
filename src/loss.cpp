//
// Created by fengjiaxin on 2023/5/6.
//

#include "loss.h"
#include "utils.h"
#include "math_helper.h"

namespace word2vec {

bool comparePairs(
        const std::pair<real, int32_t>& l,
        const std::pair<real, int32_t>& r) {
    return l.first > r.first;
}


void Loss::predict(
        int32_t k,
        real threshold,
        Predictions& heap,
        Model::State& state) const {
    computeOutput(state);
    findKBest(k, threshold, heap, state.output);
    std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void Loss::findKBest(
        int32_t k,
        real threshold,
        Predictions& heap,
        const Vector& output) const {
    for (int32_t i = 0; i < output.size(); i++) {
        if (output[i] < threshold) {
            continue;
        }
        if (heap.size() == k && ml_log(output[i]) < heap.front().first) {
            continue;
        }
        heap.push_back(std::make_pair(ml_log(output[i]), i));
        std::push_heap(heap.begin(), heap.end(), comparePairs);
        if (heap.size() > k) {
            std::pop_heap(heap.begin(), heap.end(), comparePairs);
            heap.pop_back();
        }
    }
}



BinaryLogisticLoss::BinaryLogisticLoss(std::shared_ptr<Matrix>& wo)
    : Loss(wo) {}

real BinaryLogisticLoss::binaryLogistic(
        int32_t target,
        Model::State& state,
        bool isPositive,
        real lr,
        bool backprop) const {
    real score = ml_sigmoid(wo_->dotRow(state.hidden, target));
    if (backprop) {
        real alpha = lr * (real(isPositive) - score);
        state.grad.addRow(*wo_, target, alpha);
        wo_->addVectorToRow(state.hidden, target, alpha);
    }
    if (isPositive) {
        return -ml_log(score);
    } else {
        return -ml_log(1.0 - score);
    }
}

void BinaryLogisticLoss::computeOutput(Model::State& state) const {
    Vector& output = state.output;
    output.mul(*wo_, state.hidden);
    int32_t osz = output.size();
    for (int32_t i = 0; i < osz; i++) {
        output[i] = ml_sigmoid(output[i]);
    }
}



NegativeSamplingLoss::NegativeSamplingLoss(
        std::shared_ptr<Matrix>& wo,
        int neg,
        const std::vector<int32_t>& ids,
        const std::vector<int32_t>& wordCounts)
        : BinaryLogisticLoss(wo), neg_(neg), aliasSample(wordCounts, ids) {}

real NegativeSamplingLoss::forward(
        const std::vector<int32_t>& targets,
        int32_t targetIndex,
        Model::State& state,
        real lr,
        bool backprop) {
    assert(targetIndex >= 0);
    assert(targetIndex < targets.size());
    int32_t target = targets[targetIndex];
    real loss = binaryLogistic(target, state, true, lr, backprop);

    for (int32_t n = 0; n < neg_; n++) {
        int32_t negativeTarget = getNegative(target);
        loss += binaryLogistic(negativeTarget, state, false, lr, backprop);
    }
    return loss;
}

int32_t NegativeSamplingLoss::getNegative(
        int32_t target) {
    int32_t negative;
    do {
        negative = aliasSample.Next();
    } while (target == negative);
    return negative;
}




} // namespace word2vec
