//
// Created by fengjiaxin on 2023/5/6.
//

#ifndef WORD2VEC_UTILS_H
#define WORD2VEC_UTILS_H

#include "real.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <ostream>
#include <vector>

namespace word2vec {

using Predictions = std::vector<std::pair<real, int32_t>>;

namespace utils {

int64_t size(std::ifstream&);

void seek(std::ifstream&, int64_t);

double getDuration(
        const std::chrono::steady_clock::time_point& start,
        const std::chrono::steady_clock::time_point& end);

class ClockPrint {
public:
    explicit ClockPrint(int32_t duration) : duration_(duration) {};
    friend std::ostream& operator<<(std::ostream& out, const ClockPrint& me);

private:
    int32_t duration_;
};

} // namespace utils

} // namespace word2vec

#endif //WORD2VEC_UTILS_H
