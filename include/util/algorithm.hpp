#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>

using std::begin;
using std::end;

namespace util {

    template<class Range>
    auto size(Range&& rng)
    {
        return std::distance(begin(rng), end(rng));
    }

    template<class Range, class It>
    auto index(Range&& rng, It it)
    {
        return std::distance(begin(rng), it);
    }

    template<class Range, class T>
    auto accumulate(Range&& rng, T init)
        -> T
    {
        return std::accumulate(begin(rng), end(rng), init);
    }

    template<class Range, class OutputIt>
    auto copy(Range&& rng, OutputIt d_first)
        -> OutputIt
    {
        return std::copy(begin(rng), end(rng), d_first);
    }

    template<class InRange, class Size, class OutputIt>
    auto copy_n(InRange&& in_rng, Size count, OutputIt result)
        -> OutputIt
    {
        return std::copy_n(begin(in_rng), count, result);
    }

    template<class Range, class T>
    void fill(Range&& rng, const T& value)
    {
        std::fill(begin(rng), end(rng), value);
    }

    template<class Range, class UnaryFunction>
    auto for_each(Range&& rng, UnaryFunction f)
        -> UnaryFunction
    {
        return std::for_each(begin(rng), end(rng), f);
    }

    template<class Range, class Generator>
    void generate(Range&& rng, Generator g)
    {
        std::generate(begin(rng), end(rng), g);
    }

    template<class Range>
    auto min_element(Range&& rng)
    {
        return std::min_element(begin(rng), end(rng));
    }

    template<class Range>
    auto minmax_element(Range&& rng)
    {
        return std::minmax_element(begin(rng), end(rng));
    }

    template<class Range>
    auto minmax(Range&& rng) {
#ifdef __cpp_structured_bindings
        auto [min, max] = minmax_element(std::forward<Range>(rng));
#else
        decltype(begin(std::declval<Range>())) min, max;
        std::tie(min, max) = minmax_element(std::forward<Range>(rng));
#endif
        return std::make_pair(*min, *max);
    }

    template<class Range, class URBG>
    void shuffle(Range&& rng, URBG&& g)
    {
        std::shuffle(begin(rng), end(rng), std::forward<URBG>(g));
    }

    template<class Range, class OutputIt, class UnaryOperation>
    auto transform(Range&& rng, OutputIt d_first, UnaryOperation unary_op)
        -> OutputIt
    {
        return std::transform(begin(rng), end(rng), d_first, unary_op);
    }

    // helper functions

#ifdef __cpp_fold_expressions

    template<class... T>
    auto sum(T&&... v)
    {
        return (std::forward<T>(v) + ...);
    }

    template<class... T>
    auto multiply(T&&... v)
    {
        return (std::forward<T>(v) * ...);
    }

#else

    template<class T, class U>
    auto sum(T&& a, U&& b)
    {
        return std::forward<T>(a) + std::forward<U>(b);
    }

    template<class T, class U>
    auto multiply(T&& a, U&& b)
    {
        return std::forward<T>(a) * std::forward<U>(b);
    }

#endif

    template<class T>
    auto square(const T& v)
    {
        return v * v;
    }

    template<class T>
    auto multiply_by(T&& a)
    {
        return [a = std::forward<T>(a)](auto&& b) {
            return a * b;
        };
    }

} // namespace util
