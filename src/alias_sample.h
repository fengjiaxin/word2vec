//
// Created by fengjiaxin on 2023/5/5.
// 别名采样

#ifndef WORD2VEC_ALIAS_SAMPLE_H
#define WORD2VEC_ALIAS_SAMPLE_H


#include <deque>
#include <vector>
#include <unordered_map>
#include <random>
#include <ctime>

// https://www.cnblogs.com/Lee-yl/p/12749070.html
namespace word2vec {

class AliasSample {
private:
    std::vector<float> probability;
    std::vector<int32_t> alias;
    std::vector<int32_t> ids;

    std::uniform_real_distribution<float> rnd_;
    std::uniform_int_distribution<int32_t> rnd_idx_;

    std::default_random_engine rnd {static_cast<uint_fast32_t>(time(nullptr))};
    int32_t len;

public:
    // id是已经对word进行编码
    explicit AliasSample(const std::vector<int32_t> &freqs, const std::vector<int32_t> &ids_) :
            probability(freqs.size(), 0), alias(freqs.size(), 0), ids(ids_),
            rnd_(0.0, 1.0), rnd_idx_(0, freqs.size() - 1) {
        assert(freqs.size() == ids_.size());

        len = freqs.size();
        std::vector<double> probs(len, 0);

        double denominator = 0.0;
        for (int32_t i = 0; i < len; ++i) {
            probs[i] = std::pow(freqs[i], 0.75);
            denominator += probs[i];
        }
        for (int32_t i = 0; i < len; i++) {
            probs[i] = probs[i] * len / denominator;
        }

        double average = 1.0;
        std::deque<int32_t> small;
        std::deque<int32_t> large;
        for (int32_t idx = 0; idx < probs.size(); ++idx) {
            if (probs[idx] >= average)
                large.push_back(idx);
            else
                small.push_back(idx);
        }

        while (!small.empty() && !large.empty()) {
            int32_t less = small.front();
            int32_t more = large.front();

            small.pop_front();
            large.pop_front();

            // 自身 + 被补充部分, probabilities.get(less) + probabilities.get(more) >= 1 一定成立
            probability[less] = probs[less];// 自身，需要被补充
            alias[less] = more;// more有盈余,用来补充less

            // more 补充 less后, 更新more
            probs[more] = (probs[more] + probs[less]) - average;

            if (probs[more] >= average)
                large.push_back(more); // more补充less之后还有盈余
            else
                small.push_back(more); // 补充之后需要被补充
        }

        while (!small.empty()) {
            probability[small.back()] = 1.0;
            small.pop_back();
        }
        while (!large.empty()) {
            probability[large.back()] = 1.0;
            large.pop_back();
        }
    }

    int32_t Next() {
        float d = rnd_(rnd);
        int32_t column = rnd_idx_(rnd);

        if (d < probability[column]) {
            return ids[column];
        } else {
            return ids[alias[column]];
        }
    }
};

} // namespace word2vec



#endif //WORD2VEC_ALIAS_SAMPLE_H
