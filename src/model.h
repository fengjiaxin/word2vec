//
// Created by fengjiaxin on 2023/5/6.
//

#ifndef WORD2VEC_MODEL_H
#define WORD2VEC_MODEL_H

#include <memory>
#include <utility>
#include <vector>

#include "matrix.h"
#include "real.h"
#include "utils.h"
#include "vector.h"


namespace word2vec {

class Loss;

class Model {
private:
    std::shared_ptr<Matrix> wi_;
    std::shared_ptr<Matrix> wo_;
    std::shared_ptr<Loss> loss_;

public:
    Model(
        std::shared_ptr<Matrix> wi,
        std::shared_ptr<Matrix> wo,
        std::shared_ptr<Loss> loss);
    Model(const Model& model) = delete;
    Model(Model&& model) = delete;
    Model& operator=(const Model& other) = delete;
    Model& operator=(Model&& other) = delete;

    class State {
    private:
        real lossValue_;
        int64_t nexamples_;

    public:
        Vector hidden;
        Vector output;
        Vector grad;

        State(int32_t hiddenSize, int32_t outputSize);
        real getLoss() const;
        void incrementNExamples(real loss);
    };

    void predict(
            const std::vector<int32_t>& input,
            int32_t k,
            real threshold,
            Predictions& heap,
            State& state) const;
    void update(
            const std::vector<int32_t>& input,
            const std::vector<int32_t>& targets,
            int32_t targetIndex,
            real lr,
            State& state);
    void computeHidden(const std::vector<int32_t>& input, State& state) const;

};




} // namespace word2vec

#endif //WORD2VEC_MODEL_H
