//
// Created by fengjiaxin on 2023/5/6.
//

#include "utils.h"
#include <ios>

namespace word2vec {

namespace utils {

int64_t size(std::ifstream& ifs) {
    ifs.seekg(std::streamoff(0), std::ios::end);
    return ifs.tellg();
}

void seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::streampos(pos));
}

double getDuration(
        const std::chrono::steady_clock::time_point& start,
        const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
}

std::ostream& operator<<(std::ostream& out, const ClockPrint& me) {
    int32_t etah = me.duration_ / 3600;
    int32_t etam = (me.duration_ % 3600) / 60;
    int32_t etas = (me.duration_ % 3600) % 60;

    out << std::setw(3) << etah << "h" << std::setw(2) << etam << "m";
    out << std::setw(2) << etas << "s";
    return out;
}

} // namespace utils

} // namespace word2vec