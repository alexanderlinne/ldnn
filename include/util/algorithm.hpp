#pragma once

#include <algorithm>
#include <iterator>

namespace util {

    template<class Range, class It>
    auto index(Range&& rng, It it)
    {
        return std::distance(std::begin(rng), it);
    }

    // helper functions

    template<class... T>
    auto sum(T&&... v)
    {
        return (std::forward<T>(v) + ...);
    }

    template<class T>
    auto square(const T& v)
    {
        return v * v;
    }

    template<class... Ts>
    auto multiply(const Ts&... v)
    {
        return (v * ...);
    }

    template<class T>
    auto multiply_by(T&& a)
    {
        return [a = std::forward<T>(a)](auto&& b) {
            return a * b;
        };
    }

} // namespace util
