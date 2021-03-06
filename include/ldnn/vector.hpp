#pragma once

#include <iosfwd>
#include <vector>

#include "util/algorithm.hpp"
#include "util/indices.hpp"
#include "util/iterator/ostream_joiner.hpp"

namespace ldnn {

    struct rank_t {
        size_t value;
    };

    auto operator==(rank_t l, rank_t r)
        -> bool
    {
        return l.value == r.value;
    }
    auto operator!=(rank_t l, rank_t r)
        -> bool
    {
        return !(l == r);
    }
    auto operator>(rank_t l, rank_t r)
        -> bool
    {
        return l.value > r.value;
    }
    auto operator<(rank_t l, rank_t r)
        -> bool
    {
        return !(l > r) && !(l == r);
    }
    auto operator>=(rank_t l, rank_t r)
        -> bool
    {
        return l > r || l == r;
    }
    auto operator<=(rank_t l, rank_t r)
        -> bool
    {
        return l < r || l == r;
    }

    auto operator<(size_t l, rank_t r)
        -> bool
    {
        return l < r.value;
    }

} // namespace ldnn

template<>
auto indices<ldnn::rank_t>(ldnn::rank_t last)
{
    return indices(last.value);
}

namespace ldnn {

    template<class T = double>
    struct vector {
        static_assert(std::is_floating_point<T>::value,
            "T has to be a floating-point type");

        using value_type = T;
        using reference = T&;
        using const_reference = T const&;

        vector() = default;

        vector(rank_t rank)
            : vector(rank, 0)
        {}

        vector(rank_t rank, value_type initial_value) {
            data.resize(rank.value);
            util::fill(data, initial_value);
        }

        vector(const std::vector<T>& init) : data(init) {}
        vector(std::vector<T>&& init) : data(std::move(init)) {}

        vector(std::initializer_list<value_type> init)
            : data(init)
        {}

        vector(vector const&) = default;
        vector(vector&&) = default;

        vector& operator=(vector const&) = default;
        vector& operator=(vector&&) = default;

        auto begin() noexcept
            -> decltype(auto)
        {
            return data.begin();
        }

        auto end() noexcept
            -> decltype(auto)
        {
            return data.end();
        }

        auto begin() const noexcept
            -> decltype(auto)
        {
            return data.begin();
        }

        auto end() const noexcept
            -> decltype(auto)
        {
            return data.end();
        }

        auto cbegin() const noexcept
            -> decltype(auto)
        {
            return data.cbegin();
        }

        auto cend() const noexcept
            -> decltype(auto)
        {
            return data.cend();
        }

        auto rank() const noexcept
            -> rank_t
        {
            return {data.size()};
        }

        auto operator[](size_t index)
            -> reference
        {
            return data[index];
        }

        auto operator[](size_t index) const
            -> const_reference
        {
            return data[index];
        }

    private:
        std::vector<T> data;
    };

    template<class T>
    auto operator<<(std::ostream& o, const vector<T>& v)
        -> std::ostream&
    {
        o << "(";
        util::copy(v, util::iterator::ostream_joiner<T>(std::cout, " "));
        return o << ")";
    }

    template<class T, class U,
        class = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    auto scale(const vector<T>& vec, U scale)
        -> vector<T>
    {
        auto result = vector<T>{vec.rank()};
        util::transform(vec, result.begin(), util::multiply_by(scale));
        return result;
    }

    template<class T>
    auto length(const vector<T>& vec)
        -> typename vector<T>::value_type
    {
        auto sum = T{0};
        util::for_each(vec, [&](auto& e) { sum += util::square(e); });
        return std::sqrt(sum);
    }

    template<class T>
    auto normalize(const vector<T>& vec)
        -> vector<T>
    {
        return scale(vec, T{1} / length(vec));
    }

    namespace detail {

        template<class T, class U, class Fn>
        auto vector_merge(const vector<T>& l, const vector<U>& r, Fn&& fn)
        {
            if (l.rank() != r.rank())
                throw std::invalid_argument{"rank differs"};

            auto result = vector<decltype(fn(l[0], r[0]))>{l.rank()};
            for (auto i : indices(result.rank())) {
                result[i] = fn(l[i], r[i]);
            }

            return result;
        }

    } // namespace detail

    template<class T, class U>
    auto operator==(const vector<T>& l, const vector<U>& r)
        -> bool
    {
        if (l.rank() != r.rank())
            return false;
        auto equal = true;
        for (auto i : indices(l.rank()))
            equal &= l[i] == r[i];
        return equal;
    }

    template<class T, class U>
    auto operator!=(const vector<T>& l, const vector<U>& r)
        -> bool
    {
        return !(l == r);
    }

    template<class T, class U>
    auto operator*(const vector<T>& l, const vector<U>& r)
        -> decltype(l[0] * r[0])
    {
        auto _tmp = detail::vector_merge(l, r,
            [](auto a, auto b) { return a * b; });
        return util::accumulate(_tmp, T{0});
    }

    template<class T, class U,
        class = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    auto operator*(const vector<T>& l, U r)
        -> vector<T>
    {
        return scale(l, r);
    }

    template<class T, class U,
        class = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    auto operator*(U r, const vector<T>& l)
        -> vector<T>
    {
        return l * r;
    }

    template<class T, class U,
        class = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    auto operator/(const vector<T>& l, U r)
        -> vector<T>
    {
        return scale(l, U{1} / r);
    }

    template<class T, class U>
    auto operator+(const vector<T>& l, const vector<U>& r)
        -> vector<decltype(l[0] + r[0])>
    {
        return detail::vector_merge(l, r, [](auto a, auto b) { return a + b; });
    }

    template<class T, class U>
    auto operator-(const vector<T>& l, const vector<U>& r)
        -> vector<decltype(l[0] - r[0])>
    {
        return detail::vector_merge(l, r, [](auto a, auto b) { return a - b; });
    }

    template<class T, class U>
    auto distance(const vector<T>& l, const vector<U>& r)
        -> decltype(length(l - r))
    {
        return length(l - r);
    }

    template<class InputIterator>
    auto centroid(InputIterator first, InputIterator last) {
        if (first == last) {
            throw std::invalid_argument("empty range");
        }

        using value_type =
            typename std::iterator_traits<InputIterator>::value_type;
        auto zero = value_type{(*first).rank()};
        return std::accumulate(first, last, zero)
            / static_cast<double>(std::distance(first, last));
    }

    template<class InputRange>
    auto centroid(InputRange&& r) {
        return centroid(std::begin(r), std::end(r));
    }

    template<class T, class IdxRange,
        class = typename std::enable_if<
            std::is_integral<typename std::decay<IdxRange>::type::value_type>::value
        >::type
    >
    auto select_dimensions(const vector<T>& vec, IdxRange&& dims)
        -> vector<T>
    {
        auto elems = std::vector<T>{};
        util::transform(dims, std::back_inserter(elems),
            [&](auto& dim) { return vec[dim]; });
        return vector<T>{elems};
    }

    template<class T, class I>
    auto select_dimensions(const vector<T>& vec, std::initializer_list<I> dims)
        -> vector<T>
    {
        return select_dimensions(vec, std::vector<I>(dims));
    }

    template<class T, class IdxRange,
        class = typename std::enable_if<
            std::is_integral<typename std::decay<IdxRange>::type::value_type>::value
        >::type
    >
    auto select_dimensions(const std::vector<vector<T>>& vecs, IdxRange&& dims)
    {
        auto result = std::vector<vector<T>>{};
        util::transform(vecs, std::back_inserter(result),
            [&](auto& vec) { return select_dimensions(vec, dims); });
        return result;
    }

    template<class T, class I>
    auto select_dimensions(const std::vector<vector<T>>& vecs,
        std::initializer_list<I> dims)
    {
        return select_dimensions(vecs, std::vector<I>(dims));
    }

    template<class T, class Idx>
    auto remove_dimension(const vector<T>& vec, Idx dim)
        -> vector<T>
    {
        if (dim < 0 || dim > vec.rank().value) {
            throw std::invalid_argument{"dimension out of range"};
        }

        auto result = vector<T>{rank_t{vec.rank().value - 1}};
        for (auto i = Idx{0}, j = Idx{0}; i < vec.rank(); ++i) {
            if (i == dim) {
                continue;
            }
            result[j] = vec[i];
            j++;
        }
        return result;
    };

} // namespace ldnn
