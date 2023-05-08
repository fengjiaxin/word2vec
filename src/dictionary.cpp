//
// Created by fengjiaxin on 2023/5/5.
//


#include "dictionary.h"
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace word2vec {

const std::string Dictionary::EOS = "</s>";

Dictionary::Dictionary(std::shared_ptr<Args> args)
        : args_(args),
          word2int_(MAX_VOCAB_SIZE, -1),
          nwords_(0),
          ntokens_(0) {}

Dictionary::Dictionary(std::shared_ptr<Args> args, std::istream& in)
        : args_(args),
          nwords_(0),
          ntokens_(0) {
    load(in);
}

int32_t Dictionary::find(const std::string& w) const {
    return find(w, hash(w));
}

int32_t Dictionary::find(const std::string& w, uint32_t h) const {
    int32_t word2intsize = word2int_.size();
    int32_t id = h % word2intsize;
    while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
        id = (id + 1) % word2intsize;
    }
    return id;
}

void Dictionary::add(const std::string& w) {
    int32_t h = find(w);
    ntokens_++;
    if (word2int_[h] == -1) {
        entry e;
        e.word = w;
        e.count = 1;
        words_.push_back(e);
        word2int_[h] = nwords_++;
    } else {
        words_[word2int_[h]].count++;
    }
}

int32_t Dictionary::nwords() const {
    return nwords_;
}


int64_t Dictionary::ntokens() const {
    return ntokens_;
}

// 如果找不到 return -1
int32_t Dictionary::getId(const std::string& w) const {
    int32_t h = find(w);
    return word2int_[h];
}


std::string Dictionary::getWord(int32_t id) const {
    assert(id >= 0);
    assert(id < nwords_);
    return words_[id].word;
}

// The correct implementation of fnv should be:
// h = h ^ uint32_t(uint8_t(str[i]));
// Unfortunately, earlier version of fasttext used
// h = h ^ uint32_t(str[i]);
// which is undefined behavior (as char can be signed or unsigned).
// Since all fasttext models that were already released were trained
// using signed char, we fixed the hash function to make models
// compatible whatever compiler is used.
uint32_t Dictionary::hash(const std::string& str) const {
    uint32_t h = 2166136261;
    for (size_t i = 0; i < str.size(); i++) {
        h = h ^ uint32_t(int8_t(str[i]));
        h = h * 16777619;
    }
    return h;
}


bool Dictionary::readWord(std::istream& in, std::string& word) const {
    int c;
    std::streambuf& sb = *in.rdbuf();
    word.clear();
    while ((c = sb.sbumpc()) != EOF) { // sb.sbumpc() 读取一个字符，读取指针移动
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
            c == '\f' || c == '\0') {
            if (word.empty()) {
                if (c == '\n') {
                    word += EOS;
                    return true;
                }
                continue;
            } else {
                if (c == '\n')
                    sb.sungetc();
                return true;
            }
        }
        word.push_back(c);
    }
    // trigger eofbit
    in.get();
    return !word.empty();
}

void Dictionary::readFromFile(std::istream& in) {
    std::string word;
    int64_t minThreshold = 1;
    while (readWord(in, word)) {
        add(word);
        if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
            std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
        }
        if (nwords_ > 0.75 * MAX_VOCAB_SIZE) {
            minThreshold++;
            threshold(minThreshold);
        }
    }
    threshold(args_->minCount);
    if (args_->verbose > 0) {
        std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
        std::cerr << "Number of words:  " << nwords_ << std::endl;
    }
    if (nwords_ == 0) {
        throw std::invalid_argument(
                "Empty vocabulary. Try a smaller -minCount value.");
    }
}

void Dictionary::threshold(int64_t t) {

    sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
        return e1.count > e2.count;
    });
    words_.erase(
            remove_if(
                    words_.begin(),
                    words_.end(),
                    [&](const entry& e) {
                        return e.count < t;
                    }),
            words_.end());
    words_.shrink_to_fit();
    nwords_ = 0;
    std::fill(word2int_.begin(), word2int_.end(), -1);
    for (auto it = words_.begin(); it != words_.end(); ++it) {
        int32_t h = find(it->word);
        word2int_[h] = nwords_++;
    }
}



std::vector<int32_t> Dictionary::getCounts() const {
    std::vector<int32_t> counts;
    for (auto& w : words_) {
        counts.push_back(w.count);
    }
    return counts;
}

std::vector<int32_t> Dictionary::getIds() const {
    std::vector<int32_t> ids;
    for (auto& w : words_) {
        int32_t hashId = find(w.word);
        ids.push_back(word2int_[hashId]);
    }
    return ids;
}


void Dictionary::reset(std::istream& in) const {
    if (in.eof()) {
        in.clear();
        in.seekg(std::streampos(0));
    }
}


int32_t Dictionary::getLine(
        std::istream& in,
        std::vector<int32_t>& words) const {
    std::string token;
    int32_t ntokens = 0;

    reset(in);
    words.clear();
    while (readWord(in, token)) {
        int32_t h = find(token);
        int32_t wid = word2int_[h];
        if (wid < 0) { // 没找到word,因此扔掉
            continue;
        }

        ntokens++;
        words.push_back(wid);
        if (ntokens > MAX_LINE_SIZE || token == EOS) {
            break;
        }
    }
    return ntokens;
}


void Dictionary::save(std::ostream& out) const {
    out.write((char*)&nwords_, sizeof(int32_t));
    out.write((char*)&ntokens_, sizeof(int64_t));
    for (int32_t i = 0; i < nwords_; i++) {
        entry e = words_[i];
        out.write(e.word.data(), e.word.size() * sizeof(char));
        out.put(0);
        out.write((char*)&(e.count), sizeof(int64_t));
    }
}

void Dictionary::load(std::istream& in) {
    words_.clear();
    in.read((char*)&nwords_, sizeof(int32_t));
    in.read((char*)&ntokens_, sizeof(int64_t));
    for (int32_t i = 0; i < nwords_; i++) {
        char c;
        entry e;
        while ((c = in.get()) != 0) {
            e.word.push_back(c);
        }
        in.read((char*)&e.count, sizeof(int64_t));
        words_.push_back(e);
    }

    int32_t word2intsize = std::ceil(nwords_ / 0.7);
    word2int_.assign(word2intsize, -1);
    for (int32_t i = 0; i < nwords_; i++) {
        word2int_[find(words_[i].word)] = i;
    }
}


void Dictionary::dump(std::ostream& out) const {
    out << words_.size() << std::endl;
    for (auto it : words_) {
        out << it.word << " " << it.count  << std::endl;
    }
}

} // namespace word2vec
