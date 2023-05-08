//
// Created by fengjiaxin on 2023/5/5.
//


#include "args.h"
#include <stdlib.h> // EXIT_FAILURE
#include <iostream>
#include <string>

namespace word2vec {

Args::Args() {
    lr = 0.05;
    dim = 32;
    ws = 5;
    epoch = 5;
    minCount = 5;
    neg = 5;
    loss = loss_name::ns;
    model = model_name::sg;
    thread = 12;
    lrUpdateRate = 100;
    verbose = 2;
    saveOutput = false;
    seed = 0;
}

std::string Args::lossToString(loss_name ln) const {
    switch (ln) {
        case loss_name::ns:
            return "ns";
    }
    return "Unknown loss!"; // should never happen
}

std::string Args::boolToString(bool b) const {
    if (b) {
        return "true";
    } else {
        return "false";
    }
}

std::string Args::modelToString(model_name mn) const {
    switch (mn) {
        case model_name::cbow:
            return "cbow";
        case model_name::sg:
            return "sg";
    }
    return "Unknown model name!"; // should never happen
}


void Args::parseArgs(const std::vector<std::string>& args) {
    std::string command(args[1]);
    if (command == "cbow") {
        model = model_name::cbow;
    } else if (command == "sg") {
        model = model_name::sg;
    }

    for (int ai = 2; ai < args.size(); ai += 2) {
        if (args[ai][0] != '-') {
            std::cerr << "Provided argument without a dash! Usage:" << std::endl;
            printHelp();
            exit(EXIT_FAILURE);
        }
        try {
            if (args[ai] == "-h") {
                std::cerr << "Here is the help! Usage:" << std::endl;
                printHelp();
                exit(EXIT_FAILURE);
            } else if (args[ai] == "-input") {
                input = std::string(args.at(ai + 1));
            } else if (args[ai] == "-output") {
                output = std::string(args.at(ai + 1));
            } else if (args[ai] == "-lr") {
                lr = std::stof(args.at(ai + 1));
            } else if (args[ai] == "-lrUpdateRate") {
                lrUpdateRate = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-dim") {
                dim = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-ws") {
                ws = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-epoch") {
                epoch = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-minCount") {
                minCount = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-neg") {
                neg = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-loss") {
                if (args.at(ai + 1) == "ns") {
                    loss = loss_name::ns;
                } else {
                    std::cerr << "Unknown loss: " << args.at(ai + 1) << std::endl;
                    printHelp();
                    exit(EXIT_FAILURE);
                }
            } else if (args[ai] == "-thread") {
                thread = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-verbose") {
                verbose = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-saveOutput") {
                saveOutput = true;
                ai--;
            } else if (args[ai] == "-seed") {
                seed = std::stoi(args.at(ai + 1));
            } else {
                std::cerr << "Unknown argument: " << args[ai] << std::endl;
                printHelp();
                exit(EXIT_FAILURE);
            }
        } catch (std::out_of_range) {
            std::cerr << args[ai] << " is missing an argument" << std::endl;
            printHelp();
            exit(EXIT_FAILURE);
        }
    }
    if (input.empty() || output.empty()) {
        std::cerr << "Empty input or output path." << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
    }
}

void Args::printHelp() {
    printBasicHelp();
    printDictionaryHelp();
    printTrainingHelp();
}

void Args::printBasicHelp() {
    std::cerr << "\nThe following arguments are mandatory:\n"
              << "  -input              training file path\n"
              << "  -output             output file path\n"
              << "\nThe following arguments are optional:\n"
              << "  -verbose            verbosity level [" << verbose << "]\n";
}

void Args::printDictionaryHelp() {
    std::cerr << "\nThe following arguments for the dictionary are optional:\n"
              << "  -minCount           minimal number of word occurences ["
              << minCount << "]\n";
}

void Args::printTrainingHelp() {
    std::cerr
            << "\nThe following arguments for training are optional:\n"
            << "  -lr                 learning rate [" << lr << "]\n"
            << "  -lrUpdateRate       change the rate of updates for the learning "
               "rate ["
            << lrUpdateRate << "]\n"
            << "  -dim                size of word vectors [" << dim << "]\n"
            << "  -ws                 size of the context window [" << ws << "]\n"
            << "  -epoch              number of epochs [" << epoch << "]\n"
            << "  -neg                number of negatives sampled [" << neg << "]\n"
            << "  -loss               loss function {ns, hs, softmax, one-vs-all} ["
            << lossToString(loss) << "]\n"
            << "  -thread             number of threads (set to 1 to ensure "
               "reproducible results) ["
            << thread << "]\n"
            << "  -saveOutput         whether output params should be saved ["
            << boolToString(saveOutput) << "]\n"
            << "  -seed               random generator seed  [" << seed << "]\n";
}


void Args::save(std::ostream& out) {
    out.write((char*)&(dim), sizeof(int));
    out.write((char*)&(ws), sizeof(int));
    out.write((char*)&(epoch), sizeof(int));
    out.write((char*)&(minCount), sizeof(int));
    out.write((char*)&(neg), sizeof(int));
    out.write((char*)&(loss), sizeof(loss_name));
    out.write((char*)&(model), sizeof(model_name));
    out.write((char*)&(lrUpdateRate), sizeof(int));
}

void Args::load(std::istream& in) {
    in.read((char*)&(dim), sizeof(int));
    in.read((char*)&(ws), sizeof(int));
    in.read((char*)&(epoch), sizeof(int));
    in.read((char*)&(minCount), sizeof(int));
    in.read((char*)&(neg), sizeof(int));
    in.read((char*)&(loss), sizeof(loss_name));
    in.read((char*)&(model), sizeof(model_name));
    in.read((char*)&(lrUpdateRate), sizeof(int));
}

void Args::dump(std::ostream& out) const {
    out << "dim"
        << " " << dim << std::endl;
    out << "ws"
        << " " << ws << std::endl;
    out << "epoch"
        << " " << epoch << std::endl;
    out << "minCount"
        << " " << minCount << std::endl;
    out << "neg"
        << " " << neg << std::endl;
    out << "loss"
        << " " << lossToString(loss) << std::endl;
    out << "model"
        << " " << modelToString(model) << std::endl;
    out << "lrUpdateRate"
        << " " << lrUpdateRate << std::endl;
}


} // namespace word2vec

