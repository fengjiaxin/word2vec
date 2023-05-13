//
// Created by fengjiaxin on 2023/5/5.
//

#ifndef WORD2VEC_DICTIONARY_H
#define WORD2VEC_DICTIONARY_H

#include <istream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "args.h"
#include "real.h"

namespace word2vec {

struct entry {
    std::string word;
    int32_t count;
};

class Dictionary {
protected:
    static const int32_t MAX_VOCAB_SIZE = 30000000;
    static const int32_t MAX_LINE_SIZE = 1024;

    int32_t find(const std::string&) const;
    int32_t find(const std::string&, uint32_t h) const;

    void reset(std::istream&) const;

    std::shared_ptr<Args> args_;
    std::vector<int32_t> word2int_; // 这个是对应的hash表
    std::vector<entry> words_;

    int32_t nwords_; //
    int64_t ntokens_;

public:
    explicit Dictionary(std::shared_ptr<Args>);
    explicit Dictionary(std::shared_ptr<Args>, std::istream&);
    int32_t nwords() const;
    int64_t ntokens() const;
    int32_t getId(const std::string&) const;
    std::string getWord(int32_t) const;
    uint32_t hash(const std::string& str) const;
    void add(const std::string&);

    bool readWord(std::istream&, std::string&) const;
    void readFromFile(std::istream&);
    void save(std::ostream&) const;
    void load(std::istream&);
    std::vector<int32_t> getCounts() const;
    std::vector<int32_t> getIds() const;
    int32_t getLine(std::istream&, std::vector<int32_t>&) const; // 训练模型的时候用到，调用前词典已经生成
    void threshold(int64_t);
    void dump(std::ostream&) const;
};

} // namespace word2vec


#endif //WORD2VEC_DICTIONARY_H
