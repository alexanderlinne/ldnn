#pragma once

#include <iosfwd>
#include <fstream>
#include <string>
#include <sstream>

#include "network.hpp"

namespace ldnn {

    // Returns a range of ldnn::vector<T> created from the lines of the
    // given std::istream
    template<class T>
    auto read_csv_data(std::istream& i, char delimiter)
    {
        auto vecs = std::vector<vector<T>>{};
        for (auto line = std::string{}; std::getline(i, line); ) {
            auto lnstr = std::stringstream{line};
            auto dbls = std::vector<double>{};
            for (auto word = std::string{};
                std::getline(lnstr, word, delimiter); ) {
                try {
                    dbls.push_back(std::stod(word));
                } catch(const std::invalid_argument&) {
                    dbls.push_back(std::numeric_limits<double>::quiet_NaN());
                }
            }
            vecs.push_back(vector<T>{dbls});
        }

        if (vecs.size() == 0) {
            return vecs;
        }

        // Remove all vectors that contain only NaNs
        vecs.erase(util::remove_if(vecs, [&](auto& v) {
            for (auto& e : v) {
                if (!std::isnan(e)) {
                    return false;
                }
            }
            return true;
        }), end(vecs));

        // Ensure that all read vectors have the same rank
        auto rank = vecs[0].rank();
        for (auto& v : vecs) {
            if (v.rank() != rank) {
                throw std::invalid_argument{
                    "the data contains vectors of different lengths"};
            }
        }

        return vecs;
    }

    template<class T>
    auto read_csv_file(const std::string& filename, char delimiter) {
        auto file = std::ifstream(filename);
        if (!file.is_open()) {
            throw std::invalid_argument{"File couldn't be opened!"};
        }
        return read_csv_data<T>(file, delimiter);
    }

    template<class T>
    auto dimension_to_classification(
        const std::vector<vector<T>>& data, size_t dimension) {
        auto result = std::vector<typename network<T>::classification>();
        util::transform(data, std::back_inserter(result),
            [&](auto& vec) {
            return typename network<T>::classification{
                remove_dimension(vec, dimension),
                vec[dimension] == 1
            };
        });
        return result;
    }

} // namespace ldnn
