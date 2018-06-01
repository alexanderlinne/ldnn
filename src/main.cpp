#include <chrono>
#include <iostream>
#include <random>
#include <regex>
#include <string>

#include <cxxopts.hpp>
#include <INIReader.h>

#include "ldnn/data.hpp"

using namespace std::literals;

struct config_t {
    // The name of the csv that contains the input data
    std::string filename;

    // The dimension of the input vectors that contains the classification for
    // that vector.
    size_t classification_dimension;

    // The dimensions of the input vector the network should learn on (ignoring
    // the classification dimension)
    std::vector<size_t> dimensions;

    // Number of cross validation iterations.
    size_t iterations;

    // Number of gradient descent iterations.
    size_t gradient_iterations;
};

template<class T, class URBG>
auto random_partition(std::vector<T>& vec, double p, URBG&& gen)
    -> std::pair<std::vector<T>, std::vector<T>>
{
    util::shuffle(vec, gen);
    auto split_at = std::next(begin(vec),
        static_cast<size_t>(0.5 * vec.size()));
    return std::make_pair(
        std::vector<T>(begin(vec), split_at),
        std::vector<T>(split_at, end(vec)));
}

auto read_config(const std::string& filename)
    -> config_t
{
    auto ini_config = INIReader{filename};
    if (ini_config.ParseError() < 0) {
        throw std::invalid_argument{filename + " couldn't be opened or parsed!"};
    }
    auto config = config_t{};

    config.filename = ini_config.Get("data", "filename", "");
    config.classification_dimension = static_cast<size_t>(
        ini_config.GetInteger("data", "classification_dimension", 0));

    auto dim_str = ini_config.Get("data", "dimensions", "");
    if (!std::regex_match(dim_str, std::regex{"\\[([0-9]+,)+[0-9]+\\]"}))
        throw std::invalid_argument{"The value " + dim_str
            + "is not valid for parameter data.dimensions!"};
    auto r = std::regex{"[0-9]+"};
    std::for_each(
        std::sregex_iterator{dim_str.begin(), dim_str.end(), r},
        std::sregex_iterator{},
        [&](auto& match) {
            config.dimensions.push_back(std::stoul(match.str()));
        });

    config.iterations = static_cast<size_t>(
        ini_config.GetInteger("training", "iterations", 0));
    config.gradient_iterations = static_cast<size_t>(
        ini_config.GetInteger("training", "gradient_iterations", 0));

    return config;
}

int ldnn_main(int argc, char *argv[]) {
    auto cmdopt = cxxopts::Options{
        "ldnn", "C++ implementation of a Logistic Disjunctive Normal Network"};
    cmdopt.add_options()
        ("c,config", "ini config filename", cxxopts::value<std::string>());
    auto options = cmdopt.parse(argc, argv);
    auto config_filename = "ldnn.ini"s;
    if (options.count("config") > 0) {
        config_filename = options["config"].as<std::string>();
    }

    auto config = read_config(config_filename);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "initializing...\r" << std::flush;

    // Load and parse the input data.
    auto data = ldnn::read_csv_file<double>(config.filename, '\t');
    auto examples = ldnn::dimension_to_classification(
        data, config.classification_dimension);
    for (auto& cl : examples) {
        cl.vec = ldnn::select_dimensions(cl.vec, config.dimensions);
    }

    // Normalize the input data.
    for (auto dim : indices<size_t>(examples[0].vec.rank().value)) {
        auto minmax = util::minmax(examples,
            [&](auto& c) { return c.vec[dim]; });
        for (auto& c : examples) {
            c.vec[dim] -= minmax.first;
            c.vec[dim] /= minmax.second - minmax.first;
        }
    }

    for (auto iteration : indices(config.iterations)) {
        auto start_time = std::chrono::system_clock::now();

        auto output = std::to_string(iteration + 1) + "/"
            + std::to_string(config.iterations) + ": ";
        std::cout << output << "\r" << std::flush;

        auto partitioning = random_partition(examples, 0.5, gen);
        auto network = ldnn::network<double>(
            ldnn::network<double>::read_config(config_filename),
            partitioning.first, gen);
        for (auto step : indices(config.gradient_iterations)) {
            std::cout << output << step << "/"
                      << config.gradient_iterations << "\r" << std::flush;
            util::shuffle(partitioning.first, gen);
            network.gradient_descent(partitioning.first);
        }

        auto correct = size_t{0};
        for (auto& c : partitioning.second) {
            if ((network.classify(c.vec) > 0.5) == c.positive) {
                correct++;
            }
        }
        std::cout << 100.0 * correct / partitioning.second.size()
                  << "% correctly classified! ("
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now() - start_time).count()
                  << "ms)\n";
    }

    return 0;
}

int main(int argc, char *argv[]) {
    try {
        return ldnn_main(argc, argv);
    }
    catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
    }
    catch(...) {
        std::cout << "an unexpected error occurred" << "\n";
    }
}
