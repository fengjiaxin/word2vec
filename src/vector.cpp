//
// Created by fengjiaxin on 2023/5/5.
//

#include "vector.h"
#include <cassert>
#include <iomanip>
#include <cmath>
#include "matrix.h"

namespace word2vec {

Vector::Vector(int64_t m) : data_(m) {}

void Vector::zero() {
    std::fill(data_.begin(), data_.end(), 0.0);
}

real Vector::norm() const {
    real sum = 0;
    for (int64_t i = 0; i < size(); i++) {
        sum += data_[i] * data_[i];
    }
    return std::sqrt(sum);
}

void Vector::mul(real a) {
    for (int64_t i = 0; i < size(); i++) {
        data_[i] *= a;
    }
}

void Vector::mul(const Matrix& A, const Vector& vec) {
    assert(A.size(0) == size());
    assert(A.size(1) == vec.size());
    for (int64_t i = 0; i < size(); i++) {
        data_[i] = A.dotRow(vec, i);
    }
}

void Vector::addVector(const Vector& source) {
    assert(size() == source.size());
    for (int64_t i = 0; i < size(); i++) {
        data_[i] += source.data_[i];
    }
}

void Vector::addVector(const Vector& source, real s) {
    assert(size() == source.size());
    for (int64_t i = 0; i < size(); i++) {
        data_[i] += s * source.data_[i];
    }
}

void Vector::addRow(const Matrix& A, int64_t i, real a) {
    assert(i >= 0);
    assert(i < A.size(0));
    assert(size() == A.size(1));
    A.addRowToVector(*this, i, a);
}

void Vector::addRow(const Matrix& A, int64_t i) {
    assert(i >= 0);
    assert(i < A.size(0));
    assert(size() == A.size(1));
    A.addRowToVector(*this, i);
}


std::ostream& operator<<(std::ostream& os, const Vector& v) {
    os << std::setprecision(5);
    for (int64_t j = 0; j < v.size() - 1; j++) {
        os << v[j] << ' ';
    }
    os << v[v.size() - 1];
    return os;
}

} // namespace word2vec