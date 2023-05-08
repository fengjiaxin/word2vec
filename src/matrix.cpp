//
// Created by fengjiaxin on 2023/5/5.
//

#include "matrix.h"
#include "vector.h"
#include <thread>
#include <random>
#include <cassert>

namespace word2vec {


Matrix::Matrix() : m_(0), n_(0) {}

Matrix::Matrix(int64_t m, int64_t n) : m_(m), n_(n), data_(m * n) {}

Matrix::Matrix(int64_t m, int64_t n, real* dataPtr)
        : m_(m), n_(n), data_(dataPtr, dataPtr + (m * n)) {}

Matrix::Matrix(Matrix&& other) noexcept
        : m_(other.m_), n_(other.n_), data_(std::move(other.data_)) {}



int64_t Matrix::size(int64_t dim) const {
    assert(dim == 0 || dim == 1);
    if (dim == 0) {
        return m_;
    }
    return n_;
}


void Matrix::zero() {
    std::fill(data_.begin(), data_.end(), 0.0);
}

void Matrix::uniformThread(real a, int block, int32_t seed) {
    std::minstd_rand rng(block + seed);
    std::uniform_real_distribution<> uniform(-a, a);
    int64_t blockSize = (m_ * n_) / 10;
    for (int64_t i = blockSize * block;
         i < (m_ * n_) && i < blockSize * (block + 1);
         i++) {
        data_[i] = uniform(rng);
    }
}

void Matrix::uniform(real a, unsigned int thread, int32_t seed) {
    if (thread > 1) {
        std::vector<std::thread> threads;
        for (int i = 0; i < thread; i++) {
            threads.emplace_back([=]() { uniformThread(a, i, seed); });
        }
        for (auto & item : threads) {
            item.join();
        }
    } else {
        // webassembly can't instantiate `std::thread`
        uniformThread(a, 0, seed);
    }
}


real Matrix::dotRow(const Vector& vec, int64_t i) const {
    assert(i >= 0);
    assert(i < m_);
    assert(vec.size() == n_);
    real d = 0.0;
    for (int64_t j = 0; j < n_; j++) {
        d += at(i, j) * vec[j];
    }
    if (std::isnan(d)) {
        throw EncounteredNaNError();
    }
    return d;
}

void Matrix::addVectorToRow(const Vector& vec, int64_t i, real a) {
    assert(i >= 0);
    assert(i < m_);
    assert(vec.size() == n_);
    for (int64_t j = 0; j < n_; j++) {
        data_[i * n_ + j] += a * vec[j];
    }
}

void Matrix::addRowToVector(Vector& x, int32_t i) const {
    assert(i >= 0);
    assert(i < this->size(0));
    assert(x.size() == this->size(1));
    for (int64_t j = 0; j < n_; j++) {
        x[j] += at(i, j);
    }
}

void Matrix::addRowToVector(Vector& x, int32_t i, real a) const {
    assert(i >= 0);
    assert(i < this->size(0));
    assert(x.size() == this->size(1));
    for (int64_t j = 0; j < n_; j++) {
        x[j] += a * at(i, j);
    }
}

void Matrix::save(std::ostream& out) const {
    out.write((char*)&m_, sizeof(int64_t));
    out.write((char*)&n_, sizeof(int64_t));
    out.write((char*)data_.data(), m_ * n_ * sizeof(real));
}

void Matrix::load(std::istream& in) {
    in.read((char*)&m_, sizeof(int64_t));
    in.read((char*)&n_, sizeof(int64_t));
    data_ = std::vector<real>(m_ * n_);
    in.read((char*)data_.data(), m_ * n_ * sizeof(real));
}

void Matrix::dump(std::ostream& out) const {
    out << m_ << " " << n_ << std::endl;
    for (int64_t i = 0; i < m_; i++) {
        for (int64_t j = 0; j < n_; j++) {
            if (j > 0) {
                out << " ";
            }
            out << at(i, j);
        }
        out << std::endl;
    }
};


} // namespace word2vec