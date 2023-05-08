//
// Created by fengjiaxin on 2023/5/5.
//

#ifndef WORD2VEC_MATRIX_H
#define WORD2VEC_MATRIX_H

#include <istream>
#include <ostream>
#include <vector>
#include <stdexcept>
#include "real.h"

namespace word2vec {

class Vector;

class Matrix {
private:
    int64_t m_;
    int64_t n_;
    std::vector<real> data_;

    void uniformThread(real, int, int32_t);

public:
    Matrix();

    explicit Matrix(int64_t, int64_t);

    explicit Matrix(int64_t m, int64_t n, real *dataPtr);

    Matrix(const Matrix &) = default;

    Matrix(Matrix &&) noexcept;

    Matrix &operator=(const Matrix &) = delete;

    Matrix &operator=(Matrix &&) = delete;

    ~Matrix() noexcept = default;

    int64_t size(int64_t dim) const;

    inline real *data() {
        return data_.data();
    }

    inline const real *data() const {
        return data_.data();
    }

    inline const real &at(int64_t i, int64_t j) const {
        assert(i * n_ + j < data_.size());
        return data_[i * n_ + j];
    };

    inline real &at(int64_t i, int64_t j) {
        return data_[i * n_ + j];
    };

    inline int64_t rows() const {
        return m_;
    }

    inline int64_t cols() const {
        return n_;
    }

    void zero();

    void uniform(real, unsigned int, int32_t);

    real dotRow(const Vector &, int64_t) const;

    void addVectorToRow(const Vector &, int64_t, real);

    void addRowToVector(Vector &x, int32_t i) const;

    void addRowToVector(Vector &x, int32_t i, real a) const;

    void save(std::ostream &) const;

    void load(std::istream &);

    void dump(std::ostream &) const;

    class EncounteredNaNError : public std::runtime_error {
    public:
        EncounteredNaNError() : std::runtime_error("Encountered NaN.") {}
    };
};

} // namespace word2vec


#endif //WORD2VEC_MATRIX_H
