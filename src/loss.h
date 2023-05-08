//
// Created by fengjiaxin on 2023/5/6.
//

#ifndef WORD2VEC_LOSS_H
#define WORD2VEC_LOSS_H


#include <memory>
#include <random>
#include <vector>

#include "matrix.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"
#include "alias_sample.h"

namespace word2vec {

class Loss {
protected:
    void findKBest(
            int32_t k,
            real threshold,
            Predictions& heap,
            const Vector& output) const;

protected:
    std::shared_ptr<Matrix>& wo_;

public:
    explicit Loss(std::shared_ptr<Matrix>& wo) : wo_(wo) {}
    virtual ~Loss() = default;

    virtual real forward(
            const std::vector<int32_t>& targets,
            int32_t targetIndex,
            Model::State& state,
            real lr,
            bool backprop) = 0;

    virtual void computeOutput(Model::State& state) const = 0;

    virtual void predict(
            int32_t /*k*/,
            real /*threshold*/,
            Predictions& /*heap*/,
            Model::State& /*state*/) const;
};

class BinaryLogisticLoss : public Loss {
protected:
    real binaryLogistic(
            int32_t target,
            Model::State& state,
            bool isPositive,
            real lr,
            bool backprop) const;

public:
    explicit BinaryLogisticLoss(std::shared_ptr<Matrix>& wo);
    virtual ~BinaryLogisticLoss() noexcept override = default;
    void computeOutput(Model::State& state) const override;
};



class NegativeSamplingLoss : public BinaryLogisticLoss {
protected:
    int neg_;
    AliasSample aliasSample;
    int32_t getNegative(int32_t target);

public:
    explicit NegativeSamplingLoss(
            std::shared_ptr<Matrix>& wo,
            int neg,
            const std::vector<int32_t>& ids,
            const std::vector<int32_t>& wordCounts);
    ~NegativeSamplingLoss() noexcept override = default;

    real forward(
            const std::vector<int32_t>& targets,
            int32_t targetIndex,
            Model::State& state,
            real lr,
            bool backprop) override;
};

} // namespace word2vec


#endif //WORD2VEC_LOSS_H
