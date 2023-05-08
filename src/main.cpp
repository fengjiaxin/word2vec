//
// Created by fengjiaxin on 2023/5/6.
//


#include <iostream>
#include <queue>
#include <stdexcept>
#include "args.h"
#include "word2vec.h"

using namespace word2vec;

void printUsage() {
    std::cerr
            << "usage: word2vec <command> <args>\n\n"
            << "The commands supported by word2vec are:\n\n"
            << "  skipgram                train a skipgram model\n"
            << "  cbow                    train a cbow model\n"
            << "  print-word-vectors      print word vectors given a trained model\n"
            << "  nn                      query for nearest neighbors\n"
            << "  dump                    dump arguments,dictionary,input/output vectors\n"
            << std::endl;
}


void printPrintWordVectorsUsage() {
    std::cerr << "usage: word2vec print-word-vectors <model>\n\n"
              << "  <model>      model filename\n"
              << std::endl;
}



void printNNUsage() {
    std::cout << "usage: word2vec nn <model> <k>\n\n"
              << "  <model>      model filename\n"
              << "  <k>          (optional; 10 by default) predict top k words\n"
              << std::endl;
}



void printDumpUsage() {
    std::cout << "usage: word2vec dump <model> <option>\n\n"
              << "  <model>      model filename\n"
              << "  <option>     option from args,dict,input,output" << std::endl;
}

void printWordVectors(const std::vector<std::string> args) {
    if (args.size() != 3) {
        printPrintWordVectorsUsage();
        exit(EXIT_FAILURE);
    }
    Word2Vec word2Vec;
    word2Vec.loadModel(std::string(args[2]));
    Vector vec(word2Vec.getDimension());
    std::string prompt("Query word? ");
    std::cout << prompt;
    std::string word;
    while (std::cin >> word) {
        int32_t wordId = word2Vec.getWordId(word);
        if (wordId == -1) {
            std::cout << word << " not in dict." << std::endl;
        } else {
            word2Vec.getWordVector(vec, word);
            std::cout << word << " " << vec << std::endl;
        }
        std::cout << prompt;
    }
    exit(0);
}




void printPredictions(
        const std::vector<std::pair<real, std::string>>& predictions) {

    for (const auto& prediction : predictions) {
        std::cout << prediction.second;
        std::cout << " " << prediction.first;
        std::cout << std::endl;
    }
}


void nn(const std::vector<std::string> args) {
    int32_t k;
    if (args.size() == 3) {
        k = 10;
    } else if (args.size() == 4) {
        k = std::stoi(args[3]);
    } else {
        printNNUsage();
        exit(EXIT_FAILURE);
    }
    Word2Vec word2Vec;
    word2Vec.loadModel(std::string(args[2]));
    std::string prompt("Query word? ");
    std::cout << prompt;

    std::string queryWord;
    while (std::cin >> queryWord) {
        printPredictions(word2Vec.getNN(queryWord, k));
        std::cout << prompt;
    }
    exit(0);
}

void train(const std::vector<std::string> args) {
    Args a ;
    a.parseArgs(args);
    std::shared_ptr<Word2Vec> word2Vec = std::make_shared<Word2Vec>();
    std::string outputFileName;
    outputFileName = a.output + ".bin";

    std::ofstream ofs(outputFileName);
    if (!ofs.is_open()) {
        throw std::invalid_argument(
                outputFileName + " cannot be opened for saving.");
    }
    ofs.close();
    word2Vec->train(a);
    word2Vec->saveModel(outputFileName);
    word2Vec->saveVectors(a.output + ".vec");
    if (a.saveOutput) {
        word2Vec->saveOutput(a.output + ".output");
    }
}

void dump(const std::vector<std::string>& args) {
    if (args.size() < 4) {
        printDumpUsage();
        exit(EXIT_FAILURE);
    }

    std::string modelPath = args[2];
    std::string option = args[3];

    Word2Vec word2Vec;
    word2Vec.loadModel(modelPath);
    if (option == "args") {
        word2Vec.getArgs().dump(std::cout);
    } else if (option == "dict") {
        word2Vec.getDictionary()->dump(std::cout);
    } else if (option == "input") {
        word2Vec.getInputMatrix()->dump(std::cout);
    } else if (option == "output") {
        word2Vec.getOutputMatrix()->dump(std::cout);
    } else {
        printDumpUsage();
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 2) {
        printUsage();
        exit(EXIT_FAILURE);
    }
    std::string command(args[1]);
    if (command == "skipgram" || command == "cbow") {
        train(args);
    } else if (command == "print-word-vectors") {
        printWordVectors(args);
    } else if (command == "nn") {
        nn(args);
    } else if (command == "dump") {
        dump(args);
    } else {
        printUsage();
        exit(EXIT_FAILURE);
    }
    return 0;
}
