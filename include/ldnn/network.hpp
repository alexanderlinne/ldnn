#pragma once

#include <type_traits>

#include "ldnn/vector.hpp"

namespace ldnn {

template<class T = double>
class network {
    static_assert(std::is_floating_point<T>::value,
        "T has to be a floating-point type");

public:
    struct config_t {
        // Number of polytopes.
        size_t polytope_count;

        // Maximum number of halfspaces per polytope.
        size_t max_halfspaces;

        // Alpha parameter of the network.
        T alpha;

        // Number of iterations for the kmeans algorithm.
        size_t kmeans_iterations;
    };

    struct classification {
        vector<T> vec;
        bool positive;
    };

public:
    static config_t read_config(const std::string& filename) {
        auto ini_config = INIReader{filename};
        auto config = config_t{};

        config.polytope_count =
            ini_config.GetInteger("network", "polytope_count", 0);
        config.max_halfspaces =
            ini_config.GetInteger("network", "max_halfspaces", 0);
        config.alpha =
            ini_config.GetReal("network", "alpha", 0.0);
        config.kmeans_iterations =
            ini_config.GetInteger("network", "kmeans_iterations", 0);

        return config;
    }

public:
    template<class URBG>
    network(config_t config, const std::vector<classification>& examples, URBG&& gen)
        : config(config)
    {
        if (examples.size() == 0)
            throw std::invalid_argument("examples.size() == 0");

        // Check that all input data has the same rank.
        auto rank = examples[0].vec.rank();
        for (auto& c : examples) {
            if (c.vec.rank() != rank) {
                throw std::invalid_argument(
                    "all examples must have the same rank");
            }
        }

        // Allocate memory.
        weight.resize(config.polytope_count);
        bias.resize(config.polytope_count);
        for (auto i : indices(config.polytope_count)) {
            weight[i].resize(config.max_halfspaces, vector<T>(rank));
            bias[i].resize(config.max_halfspaces);
        }

        // Initialize the network.
        auto pos_examples = std::vector<vector<T>>{};
        auto neg_examples = std::vector<vector<T>>{};
        util::for_each(examples, [&](auto& c) {
            if (c.positive) {
                pos_examples.push_back(c.vec);
            } else {
                neg_examples.push_back(c.vec);
            }
        });
        auto pos_ctrds = kmeans(pos_examples,
            config.polytope_count, gen, config.kmeans_iterations);
        auto neg_ctrds = kmeans(neg_examples,
            config.max_halfspaces, gen, config.kmeans_iterations);
        for (auto i : indices(pos_ctrds.size())) {
            for (auto j : indices(neg_ctrds.size())) {
                weight[i][j] = normalize(pos_ctrds[i] - neg_ctrds[j]);
                bias[i][j] = weight[i][j] * (0.5 * (pos_ctrds[i] + neg_ctrds[j]));
            }
        }
    }

    auto classify(const vector<T>& v)
        -> T
    {
        auto result = T{1};
        for (auto i : indices(config.polytope_count)) {
            result *= T{1} - polytope(i, v);
        }

        return T{1} - result;
    }

    void gradient_descent(const classification& c) {
        for (auto i : indices(config.polytope_count)) {
            for (auto j : indices(config.max_halfspaces)) {
                auto diff = T{2} * error(c);

                for (auto r : indices(config.polytope_count)) {
                    if (i != r) {
                        diff *= (T{1} - polytope(r, c.vec));
                    }
                }

                diff *= polytope(i, c.vec) * (T{1} - halfspace(i, j, c.vec));
                diff *= config.alpha;

                weight[i][j] = weight[i][j] - (diff * c.vec);
                bias[i][j] -= diff;
            };
        }
    }

    template<class Range,
        class = typename std::enable_if<
            std::is_convertible<
                typename std::decay<Range>::type::value_type,
                classification
            >::value
        >::type
    >
    void gradient_descent(Range&& rng) {
        util::for_each(rng, [&](auto& c) { gradient_descent(c); });
    }

    T quadratic_error(const classification& c) {
        return util::square(error(c));
    }

    template<class Range,
        class = typename std::enable_if<
            std::is_convertible<
                typename std::decay<Range>::type::value_type,
                classification
            >::value
        >::type
    >
    T quadratic_error(Range&& data) {
        auto error = std::vector<T>{};
        util::transform(data, std::back_inserter(error),
            [&](auto& c) { return quadratic_error(c); });
        return util::accumulate(error, T{0});
    }

private:
    T error(const classification& c) {
        return classify(c.vec) - (c.positive ? T{1} : T{0});
    }

    template<class URBG>
    static auto kmeans(std::vector<vector<T>> data,
        size_t k, URBG&& gen, size_t iterations = 10)
        -> std::vector<vector<T>>
    {
        // Shuffle the input vector.
        util::shuffle(data, std::forward<URBG>(gen));

        // There have to be at least as many data elements as the number
        // of clusters to be calculated.
        if (k > data.size()) {
            throw std::invalid_argument("too many clusters for given data");
        }

        auto centroids = std::vector<vector<T>>(k);
        util::copy_n(data, k, begin(centroids));

        // Iterate
        auto clusters = std::vector<std::vector<vector<T>>>(k);
        auto add_to_nearest_cluster = [&](const auto& vec) {
            auto dist = std::vector<T>(centroids.size());
            util::transform(centroids, begin(dist),
                [&](auto& c) { return distance(vec, c); });
            clusters[util::index(dist, util::min_element(dist))].push_back(vec);
        };
        while (iterations-- > 0) {
            util::for_each(data, add_to_nearest_cluster);
            util::transform(clusters, centroids.begin(),
                [](auto& cluster) { return centroid(cluster); });
            util::for_each(clusters, [](auto& v) { v.clear(); });
        }

        return centroids;
    }

    auto halfspace(size_t i, size_t j, const vector<T>& v)
        -> T
    {
        auto denom = T{1} + std::exp(-(weight[i][j] * v) - bias[i][j]);

        // std::exp may return inf: print an error and return 1 / inf ~ 0
        if (std::isinf(denom)) {
            std::cerr << "invalid denom: " << denom
                << ", i: " << i << ", j: " << j
                << ", weight * v: " << (weight[i][j] * v)
                << ", bias: " << bias[i][j]
                << "\n";
            return T{0};
        }

        return T{1} / denom;
    }

    auto polytope(size_t i, const vector<T>& v)
        -> T
    {
        auto result = T{1};
        for (auto j : indices(config.max_halfspaces)) {
            result *= halfspace(i, j, v);
        }
        return result;
    }

private:

    config_t config;
    std::vector<std::vector<vector<T>>> weight;
    std::vector<std::vector<T>> bias;
};

} // namespace ldnn
