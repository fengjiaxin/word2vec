//
// Created by fengjiaxin on 2023/5/8.
//

#include "../src/alias_sample.h"
#include <vector>
#include <iostream>

int main() {
    std::vector<int32_t> a {1,4,3,2};
    std::vector<int32_t> freq {1,4,3,2};
    word2vec::AliasSample alias(freq, a);

    for (int i = 0; i < 100; ++i) {
        std::cout << alias.Next() << std::endl;
    }
}

