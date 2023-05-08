//
// Created by fengjiaxin on 2023/5/6.
//

#ifndef WORD2VEC_WORD2VEC_H
#define WORD2VEC_WORD2VEC_H


#include <time.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <tuple>

#include "args.h"
#include "matrix.h"
#include "dictionary.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace word2vec {

class Word2Vec {

private:
    std::shared_ptr<Args> args_;
    std::shared_ptr<Dictionary> dict_;
    std::shared_ptr<Matrix> input_;
    std::shared_ptr<Matrix> output_;
    std::shared_ptr<Model> model_;
    std::atomic<int64_t> tokenCount_{};
    std::atomic<real> loss_{};
    std::chrono::steady_clock::time_point start_;
    std::unique_ptr<Matrix> wordVectors_;
    std::exception_ptr trainException_;

    void signModel(std::ostream&);
    bool checkModel(std::istream&);
    void startThreads();
    void addInputVector(Vector&, int32_t) const;
    void trainThread(int32_t);
    std::vector<std::pair<real, std::string>> getNN(
            const Matrix& wordVectors,
            const Vector& queryVec,
            int32_t k,
            const std::set<std::string>& banSet);
    void lazyComputeWordVectors();
    void printInfo(real, real, std::ostream&);
    std::shared_ptr<Matrix> createRandomMatrix() const;
    std::shared_ptr<Matrix> createTrainOutputMatrix() const;
    std::vector<int32_t> getTargetCounts() const;
    std::vector<int32_t> getIds() const;
    std::shared_ptr<Loss> createLoss(std::shared_ptr<Matrix>& output);

    void cbow(Model::State& state, real lr, const std::vector<int32_t>& line);
    void skipgram(Model::State& state, real lr, const std::vector<int32_t>& line);

    void precomputeWordVectors(Matrix& wordVectors);
    bool keepTraining(int64_t ntokens) const;
    void buildModel();
    std::tuple<int64_t, double, double> progressInfo(real progress);

public:
    Word2Vec();

    int32_t getWordId(const std::string& word) const;

    void getWordVector(Vector& vec, const std::string& word) const;

    inline void getInputVector(Vector& vec, int32_t ind) {
        vec.zero();
        addInputVector(vec, ind);
    }

    Args getArgs() const;

    std::shared_ptr<const Dictionary> getDictionary() const;

    std::shared_ptr<const Matrix> getInputMatrix() const;

    std::shared_ptr<const Matrix> getOutputMatrix() const;

    void saveVectors(const std::string& filename);

    void saveModel(const std::string& filename);

    void saveOutput(const std::string& filename);

    void loadModel(std::istream& in);

    void loadModel(const std::string& filename);


    std::vector<std::pair<real, std::string>> getNN(
            const std::string& word,
            int32_t k);


    void train(const Args& args);


    int getDimension() const;


};

} // namespace word2vec


#endif //WORD2VEC_WORD2VEC_H
