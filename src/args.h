//
// Created by fengjiaxin on 2023/5/5.
//

#ifndef WORD2VEC_ARGS_H
#define WORD2VEC_ARGS_H


#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace word2vec {

enum class model_name : int { cbow = 1, sg };
enum class loss_name : int { ns = 1};

class Args {
protected:
    std::string boolToString(bool) const;
    std::string modelToString(model_name) const;

public:
    Args();
    std::string input;
    std::string output;
    double lr;
    int lrUpdateRate;
    int dim;
    int ws;
    int epoch;
    int minCount;
    int neg;
    loss_name loss;
    model_name model;
    int thread;
    int verbose;
    bool saveOutput;
    int seed;

    void parseArgs(const std::vector<std::string>& args);
    void printHelp();
    void printBasicHelp();
    void printDictionaryHelp();
    void printTrainingHelp();
    void save(std::ostream&);
    void load(std::istream&);
    void dump(std::ostream&) const;
    std::string lossToString(loss_name) const;
};

} // namespace word2vec



#endif //WORD2VEC_ARGS_H
