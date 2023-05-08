//
// Created by fengjiaxin on 2023/5/6.
//

#include "word2vec.h"
#include "loss.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace word2vec {

const int32_t WORD2VEC_FILEFORMAT_MAGIC_INT32 = 793712314;

bool comparePairs(
        const std::pair<real, std::string>& l,
        const std::pair<real, std::string>& r) {
    return l.first > r.first;
}

std::shared_ptr<Loss> Word2Vec::createLoss(std::shared_ptr<Matrix>& output) {
    loss_name lossName = args_->loss;
    switch (lossName) {
        case loss_name::ns:
            return std::make_shared<NegativeSamplingLoss>(
                    output, args_->neg, getIds(), getTargetCounts());
        default:
            throw std::runtime_error("Unknown loss");
    }
}

Word2Vec::Word2Vec()
        : wordVectors_(nullptr), trainException_(nullptr) {}

void Word2Vec::addInputVector(Vector& vec, int32_t ind) const {
    vec.addRow(*input_, ind);
}

std::shared_ptr<const Dictionary> Word2Vec::getDictionary() const {
    return dict_;
}

Args Word2Vec::getArgs() const {
    return *args_.get();
}

std::shared_ptr<const Matrix> Word2Vec::getInputMatrix() const {
    assert(input_.get());
    return input_;
}

std::shared_ptr<const Matrix> Word2Vec::getOutputMatrix() const {
    assert(output_.get());
    return output_;
}

int32_t Word2Vec::getWordId(const std::string& word) const {
    return dict_->getId(word);
}


void Word2Vec::getWordVector(Vector& vec, const std::string& word) const {
    vec.zero();
    int32_t id = getWordId(word);
    addInputVector(vec,id);
}



void Word2Vec::saveVectors(const std::string& filename) {
    if (!input_ || !output_) {
        throw std::runtime_error("Model never trained");
    }
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::invalid_argument(
                filename + " cannot be opened for saving vectors!");
    }
    ofs << dict_->nwords() << " " << args_->dim << std::endl;
    Vector vec(args_->dim);
    for (int32_t i = 0; i < dict_->nwords(); i++) {
        std::string word = dict_->getWord(i);
        getWordVector(vec, word);
        ofs << word << " " << vec << std::endl;
    }
    ofs.close();
}

void Word2Vec::saveOutput(const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::invalid_argument(
                filename + " cannot be opened for saving vectors!");
    }
    int32_t n = dict_->nwords();
    ofs << n << " " << args_->dim << std::endl;
    Vector vec(args_->dim);
    for (int32_t i = 0; i < n; i++) {
        std::string word = dict_->getWord(i);
        vec.zero();
        vec.addRow(*output_, i);
        ofs << word << " " << vec << std::endl;
    }
    ofs.close();
}

bool Word2Vec::checkModel(std::istream& in) {
    int32_t magic;
    in.read((char*)&(magic), sizeof(int32_t));
    if (magic != WORD2VEC_FILEFORMAT_MAGIC_INT32) {
        return false;
    }
    return true;
}

void Word2Vec::signModel(std::ostream& out) {
    const int32_t magic = WORD2VEC_FILEFORMAT_MAGIC_INT32;
    out.write((char*)&(magic), sizeof(int32_t));
}

void Word2Vec::saveModel(const std::string& filename) {
    std::ofstream ofs(filename, std::ofstream::binary);
    if (!ofs.is_open()) {
        throw std::invalid_argument(filename + " cannot be opened for saving!");
    }
    if (!input_ || !output_) {
        throw std::runtime_error("Model never trained");
    }
    signModel(ofs);
    args_->save(ofs);
    dict_->save(ofs);
    input_->save(ofs);
    output_->save(ofs);
    ofs.close();
}

void Word2Vec::loadModel(const std::string& filename) {
    std::ifstream ifs(filename, std::ifstream::binary);
    if (!ifs.is_open()) {
        throw std::invalid_argument(filename + " cannot be opened for loading!");
    }
    if (!checkModel(ifs)) {
        throw std::invalid_argument(filename + " has wrong file format!");
    }
    loadModel(ifs);
    ifs.close();
}

std::vector<int32_t> Word2Vec::getTargetCounts() const {
    return dict_->getCounts();
}

std::vector<int32_t> Word2Vec::getIds() const {
    return dict_->getIds();
}


void Word2Vec::buildModel() {
    std::shared_ptr<Loss> loss = createLoss(output_);
    model_ = std::make_shared<Model>(input_, output_, loss);
}

void Word2Vec::loadModel(std::istream& in) {
    args_ = std::make_shared<Args>();
    input_ = std::make_shared<Matrix>();
    output_ = std::make_shared<Matrix>();
    args_->load(in);
    dict_ = std::make_shared<Dictionary>(args_, in);

    input_->load(in);
    output_->load(in);

    buildModel();
}

std::tuple<int64_t, double, double> Word2Vec::progressInfo(real progress) {
    double t = utils::getDuration(start_, std::chrono::steady_clock::now());
    double lr = args_->lr * (1.0 - progress);
    double wst = 0;

    int64_t eta = 2592000; // Default to one month in seconds (720 * 3600)

    if (progress > 0 && t >= 0) {
        eta = t * (1 - progress) / progress;
        wst = double(tokenCount_) / t / args_->thread;
    }

    return std::tuple<double, double, int64_t>(wst, lr, eta);
}

void Word2Vec::printInfo(real progress, real loss, std::ostream& log_stream) {
    double wst;
    double lr;
    int64_t eta;
    std::tie<double, double, int64_t>(wst, lr, eta) = progressInfo(progress);

    log_stream << std::fixed;
    log_stream << "Progress: ";
    log_stream << std::setprecision(1) << std::setw(5) << (progress * 100) << "%";
    log_stream << " words/sec/thread: " << std::setw(7) << int64_t(wst);
    log_stream << " lr: " << std::setw(9) << std::setprecision(6) << lr;
    log_stream << " avg.loss: " << std::setw(9) << std::setprecision(6) << loss;
    log_stream << " ETA: " << utils::ClockPrint(eta);
    log_stream << std::flush;
}



void Word2Vec::cbow(
        Model::State& state,
        real lr,
        const std::vector<int32_t>& line) {
    std::vector<int32_t> bow;
    int32_t boundary = args_->ws;
    for (int32_t w = 0; w < line.size(); w++) {
        bow.clear();
        for (int32_t c = -boundary; c <= boundary; c++) {
            if (c != 0 && w + c >= 0 && w + c < line.size()) {
                bow.push_back(line[w+c]);
            }
        }
        model_->update(bow, line, w, lr, state);
    }
}

void Word2Vec::skipgram(
        Model::State& state,
        real lr,
        const std::vector<int32_t>& line) {
    int32_t boundary = args_->ws;
    for (int32_t w = 0; w < line.size(); w++) {
        std::vector<int32_t> sg;
        sg.push_back(line[w]);
        for (int32_t c = -boundary; c <= boundary; c++) {
            if (c != 0 && w + c >= 0 && w + c < line.size()) {
                model_->update(sg, line, w + c, lr, state);
            }
        }
    }
}




void Word2Vec::precomputeWordVectors(Matrix& wordVectors) {
    Vector vec(args_->dim);
    wordVectors.zero();
    for (int32_t i = 0; i < dict_->nwords(); i++) {
        std::string word = dict_->getWord(i);
        getWordVector(vec, word);
        real norm = vec.norm();
        if (norm > 0) {
            wordVectors.addVectorToRow(vec, i, 1.0 / norm);
        }
    }
}

void Word2Vec::lazyComputeWordVectors() {
    if (!wordVectors_) {
        wordVectors_ = std::unique_ptr<Matrix>(
                new Matrix(dict_->nwords(), args_->dim));
        precomputeWordVectors(*wordVectors_);
    }
}

std::vector<std::pair<real, std::string>> Word2Vec::getNN(
        const std::string& word,
        int32_t k) {
    Vector query(args_->dim);

    getWordVector(query, word);

    lazyComputeWordVectors();
    assert(wordVectors_);
    return getNN(*wordVectors_, query, k, {word});
}

std::vector<std::pair<real, std::string>> Word2Vec::getNN(
        const Matrix& wordVectors,
        const Vector& query,
        int32_t k,
        const std::set<std::string>& banSet) {
    std::vector<std::pair<real, std::string>> heap;

    real queryNorm = query.norm();
    if (std::abs(queryNorm) < 1e-8) {
        queryNorm = 1;
    }

    for (int32_t i = 0; i < dict_->nwords(); i++) {
        std::string word = dict_->getWord(i);
        if (banSet.find(word) == banSet.end()) {
            real dp = wordVectors.dotRow(query, i);
            real similarity = dp / queryNorm;

            if (heap.size() == k && similarity < heap.front().first) {
                continue;
            }
            heap.push_back(std::make_pair(similarity, word));
            std::push_heap(heap.begin(), heap.end(), comparePairs);
            if (heap.size() > k) {
                std::pop_heap(heap.begin(), heap.end(), comparePairs);
                heap.pop_back();
            }
        }
    }
    std::sort_heap(heap.begin(), heap.end(), comparePairs);

    return heap;
}


bool Word2Vec::keepTraining(const int64_t ntokens) const {
    return tokenCount_ < args_->epoch * ntokens && !trainException_;
}

void Word2Vec::trainThread(int32_t threadId) {
    std::ifstream ifs(args_->input);
    utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

    Model::State state(args_->dim, output_->size(0));

    const int64_t ntokens = dict_->ntokens();
    int64_t localTokenCount = 0;
    std::vector<int32_t> line;
    try {
        while (keepTraining(ntokens)) {
            real progress = real(tokenCount_) / (args_->epoch * ntokens);
            real lr = args_->lr * (1.0 - progress);
            if (args_->model == model_name::cbow) {
                localTokenCount += dict_->getLine(ifs, line);
                cbow(state, lr, line);
            } else if (args_->model == model_name::sg) {
                localTokenCount += dict_->getLine(ifs, line);
                skipgram(state, lr, line);
            }
            if (localTokenCount > args_->lrUpdateRate) {
                tokenCount_ += localTokenCount;
                localTokenCount = 0;
                if (threadId == 0 && args_->verbose > 1) {
                    loss_ = state.getLoss();
                }
            }
        }
    } catch (Matrix::EncounteredNaNError&) {
        trainException_ = std::current_exception();
    }
    if (threadId == 0)
        loss_ = state.getLoss();
    ifs.close();
}

std::shared_ptr<Matrix> Word2Vec::createRandomMatrix() const {
    std::shared_ptr<Matrix> input = std::make_shared<Matrix>(
            dict_->nwords(), args_->dim);
    input->uniform(1.0 / args_->dim, args_->thread, args_->seed);

    return input;
}

std::shared_ptr<Matrix> Word2Vec::createTrainOutputMatrix() const {
    int64_t m = dict_->nwords();
    std::shared_ptr<Matrix> output =
            std::make_shared<Matrix>(m, args_->dim);
    output->zero();

    return output;
}

void Word2Vec::train(const Args& args) {
    args_ = std::make_shared<Args>(args);
    dict_ = std::make_shared<Dictionary>(args_);
    if (args_->input == "-") {
        // manage expectations
        throw std::invalid_argument("Cannot use stdin for training!");
    }
    std::ifstream ifs(args_->input);
    if (!ifs.is_open()) {
        throw std::invalid_argument(
                args_->input + " cannot be opened for training!");
    }
    dict_->readFromFile(ifs);
    ifs.close();

    input_ = createRandomMatrix();
    output_ = createTrainOutputMatrix();

    auto loss = createLoss(output_);
    model_ = std::make_shared<Model>(input_, output_, loss);
    startThreads();
}

// 真正意义上的开始训练
void Word2Vec::startThreads() {
    start_ = std::chrono::steady_clock::now();
    tokenCount_ = 0;
    loss_ = -1.0;
    trainException_ = nullptr;
    std::vector<std::thread> threads;
    if (args_->thread > 1) {
        for (int32_t i = 0; i < args_->thread; i++) {
            threads.emplace_back([=]() { trainThread(i); });
        }
    } else {
        // webassembly can't instantiate `std::thread`
        trainThread(0);
    }
    const int64_t ntokens = dict_->ntokens();
    // Same condition as trainThread
    while (keepTraining(ntokens)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));// 是为了打印整体训练信息
        if (loss_ >= 0 && args_->verbose > 1) {
            real progress = real(tokenCount_) / (args_->epoch * ntokens);
            std::cerr << "\r";
            printInfo(progress, loss_, std::cerr);
        }
    }
    for (int32_t i = 0; i < threads.size(); i++) {
        threads[i].join();
    }
    if (trainException_) {
        std::exception_ptr exception = trainException_;
        trainException_ = nullptr;
        std::rethrow_exception(exception);
    }
    if (args_->verbose > 0) {
        std::cerr << "\r";
        printInfo(1.0, loss_, std::cerr);
        std::cerr << std::endl;
    }
}

int Word2Vec::getDimension() const {
    return args_->dim;
}




} // namespace word2vec

