#pragma once

#include <type_traits>

#include <range/v3/action/shuffle.hpp>
#include <range/v3/algorithm/min_element.hpp>
#include <range/v3/view/remove_if.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>

#include "ldnn/vector.hpp"

namespace ldnn {

template<class T = double>
class network {
    static_assert(std::is_floating_point_v<T>);

public:
    struct classification {
        vector<T> vec;
        bool positive;
    };

public:
    template<class URBG>
    network(size_t polytope_count, size_t max_halfspaces, T alpha,
        const std::vector<classification>& examples, URBG&& gen)
        : alpha(alpha),
        polytope_count(polytope_count),
        max_halfspaces(max_halfspaces)
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
        weight.resize(polytope_count);
        bias.resize(polytope_count);
        for (auto i : indices(polytope_count)) {
            weight[i].resize(max_halfspaces, vector<T>(rank));
            bias[i].resize(max_halfspaces);
        }

        // Initialize the network.
        auto positives = [](bool positives) {
            return ranges::view::remove_if([=](auto& c) { return positives ^ c.positive; })
                | ranges::view::transform([](auto& c) { return c.vec; });
        };
        auto pos_ctrds = kmeans(examples | positives(true), polytope_count, gen);
        auto neg_ctrds = kmeans(examples | positives(false), max_halfspaces, gen);
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
        for (auto i : indices(polytope_count)) {
            result *= T{1} - polytope(i, v);
        }

        return T{1} - result;
    }

    void gradient_descent(const classification& c) {
        for (auto i : indices(polytope_count)) {
            for (auto j : indices(max_halfspaces)) {
                auto diff = T{2} * error(c);

                for (auto r : indices(polytope_count)) {
                    if (i != r) {
                        diff *= (T{1} - polytope(r, c.vec));
                    }
                }

                diff *= polytope(i, c.vec) * (T{1} - halfspace(i, j, c.vec));
                diff *= alpha;

                weight[i][j] = weight[i][j] - (diff * c.vec);
                bias[i][j] -= diff;
            };
        }
    }

    template<class Range,
        class = std::enable_if<
            std::is_convertible_v<
                typename std::decay_t<Range>::value_type,
                classification
            >>>
    void gradient_descent(Range&& rng) {
        ranges::for_each(rng, [&](auto& c) { gradient_descent(c); });
    }

    T quadratic_error(const classification& c) {
        return util::square(error(c));
    }

    template<class Range,
        class = std::enable_if<
            std::is_convertible_v<
                typename std::decay_t<Range>::value_type,
                classification
            >>>
    T quadratic_error(Range&& rng) {
        auto error = rng | ranges::view::transform(
                             [&](auto& c) { return quadratic_error(c); });
        return ranges::accumulate(error, T{0});
    }

private:
    T error(const classification& c) {
        return classify(c.vec) - (c.positive ? T{1} : T{0});
    }

    template<class Range, class URBG>
    static auto kmeans(Range&& rng,
        size_t k, URBG&& gen, size_t iterations = 100)
        -> std::vector<vector<T>>
    {
        // Create a shuffled vector from the input range.
        const auto data = rng | ranges::to_<std::vector<vector<T>>>()
                              | ranges::action::shuffle(gen);

        // There have to be at least as many data elements as the number
        // of clusters to be calculated.
        if (k > data.size()) {
            throw std::invalid_argument("too many clusters for given data");
        }

        auto centroids = data | ranges::view::take(k)
                              | ranges::to_<std::vector<vector<T>>>();

        // Iterate
        auto clusters = std::vector<std::vector<vector<T>>>(k);
        auto add_to_nearest_cluster = [&](const auto& vec) {
            auto dist = centroids
                | ranges::view::transform([&](auto& c) { return distance(vec, c); })
                | ranges::to_<std::vector<double>>();
            clusters[util::index(dist, ranges::min_element(dist))].push_back(vec);
        };
        while (iterations-- > 0) {
            ranges::for_each(data, add_to_nearest_cluster);
            ranges::transform(clusters, centroids.begin(),
                [](auto& cluster) { return centroid(cluster); });
            ranges::for_each(clusters, [](auto& v) { v.clear(); });
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
        for (auto j : indices(max_halfspaces)) {
            result *= halfspace(i, j, v);
        }
        return result;
    }

private:

    T alpha;
    size_t polytope_count;
    size_t max_halfspaces;
    std::vector<std::vector<vector<T>>> weight;
    std::vector<std::vector<T>> bias;
};

} // namespace ldnn
