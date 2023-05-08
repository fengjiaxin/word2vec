//
// Created by fengjiaxin on 2023/5/5.
//

#ifndef WORD2VEC_VECTOR_H
#define WORD2VEC_VECTOR_H

#include <ostream>
#include <vector>
#include "real.h"

namespace word2vec {

class Matrix;

class Vector {
private:
    std::vector<real> data_;

public:
    explicit Vector(int64_t);

    Vector(const Vector &) = default;

    Vector(Vector &&) noexcept = default;

    Vector &operator=(const Vector &) = default;

    Vector &operator=(Vector &&) = default;

    inline real *data() {
        return data_.data();
    }

    inline const real *data() const {
        return data_.data();
    }

    inline real &operator[](int64_t i) {
        return data_[i];
    }

    inline const real &operator[](int64_t i) const {
        return data_[i];
    }

    inline int64_t size() const {
        return data_.size();
    }

    void zero();

    void mul(real);
    void mul(const Matrix&, const Vector&);

    real norm() const; // 计算归一化的值
    void addVector(const Vector &source);

    void addVector(const Vector &, real);

    void addRow(const Matrix &, int64_t); // 将向量加到 matrix的第i行
    void addRow(const Matrix &, int64_t, real); // 将向量* real 加到matrix的第i行
};

std::ostream &operator<<(std::ostream &, const Vector &);

} // namespace word2vec


#endif //WORD2VEC_VECTOR_H
