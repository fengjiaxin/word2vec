//
// Created by fengjiaxin on 2023/5/11.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

const std::string EOS = "</s>";

bool readWord(std::istream& in, std::string& word) {
    int c;
    std::streambuf& sb = *in.rdbuf();
    word.clear();
    while ((c = sb.sbumpc()) != EOF) { // sbumpc, 读取当前字符，并前进指针
        if (c == ' ' || c == '\n') {
            if (word.empty()) {
                continue;
            } else {
                return true;
            }
        }
        word.push_back(c);
    }
    // trigger eofbit
    in.get(); // 而文件结束符是最后一个字符的下一个字符  https://zhuanlan.zhihu.com/p/574190452
    return !word.empty();
}

void readFromFile(std::istream& in) {
    std::string word;
    while (readWord(in, word)) {
        std::cout << word << std::endl;
    }
}


int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 2) {
        std::cout << "input file" << std::endl;
        exit(1);
    }
    std::string file(args[1]);
    std::ifstream ifs(file);
    if (!ifs.is_open()) {
        throw std::invalid_argument(
                file + " cannot be opened for training!");
    }
    readFromFile(ifs);
    ifs.close();

    return 0;
}