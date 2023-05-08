cmake_minimum_required(VERSION 3.19)
project(word2vec)

set(CMAKE_CXX_STANDARD 11)
include_directories(word2vec)


set(HEADER_FILES
        src/args.h
        src/dictionary.h
        src/word2vec.h
        src/loss.h
        src/matrix.h
        src/alias_sample.h
        src/math_helper.h
        src/model.h
        src/real.h
        src/utils.h
        src/vector.h)

set(SOURCE_FILES
        src/args.cpp
        src/dictionary.cpp
        src/word2vec.cpp
        src/loss.cpp
        src/main.cpp
        src/matrix.cpp
        src/model.cpp
        src/utils.cpp
        src/vector.cpp)



add_library(word2vec-static STATIC ${SOURCE_FILES} ${HEADER_FILES})

add_executable(word2vec src/main.cpp)
target_link_libraries(word2vec word2vec-static)
#add_executable(test_alias src/test_alias.cpp)