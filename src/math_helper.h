//
// Created by fengjiaxin on 2023/5/5.
//

#ifndef WORD2VEC_MATH_HELPER_H
#define WORD2VEC_MATH_HELPER_H




#include <cmath>
#include <vector>

namespace word2vec {

const int SIGMOID_TABLE_SIZE = 2048;
const int MAX_SIGMOID = 8;
const int LOG_TABLE_SIZE = 2048;


class FastMLMath {
    std::vector<float> t_sigmoid_;
    std::vector<float> t_log_;

public:
    FastMLMath() {
        t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
        for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
            double x = float(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
            t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
        }

        t_log_.reserve(LOG_TABLE_SIZE + 1);
        for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
            double x = (double(i) + 1e-5) / LOG_TABLE_SIZE;
            t_log_.push_back(std::log(x));
        }
    }

    float log(float x) const {
        if (x > 1.0) {
            return 0.0;
        }
        int64_t i = int64_t(x * LOG_TABLE_SIZE);
        return t_log_[i];
    }

    float sigmoid(float x) const {
        if (x < -MAX_SIGMOID) {
            return 0.0;
        } else if (x > MAX_SIGMOID) {
            return 1.0;
        } else {
            int64_t i = int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
            return t_sigmoid_[i];
        }
    }
};


static FastMLMath _fast_math{};

float ml_log(float x) { return _fast_math.log(x); }

float ml_sigmoid(float x) { return _fast_math.sigmoid(x); }

} // namespace word2vec

#endif //WORD2VEC_MATH_HELPER_H
