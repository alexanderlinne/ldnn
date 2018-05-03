#pragma once

namespace util::iterator {

    template<class T>
    struct generator
    {
        T first;
        T last;

        struct iterator {
            T value;

            using iterator_category = std::input_iterator_tag;
            using difference_type = ptrdiff_t;
            using value_type = T;
            using reference = const T&;
            using pointer = const T*;

            auto operator*() const
                -> reference
            {
                return value;
            }

            auto operator->() const
                -> reference
            {
                return std::addressof(value);
            }

            auto operator++()
                -> iterator&
            {
                ++value;
                return *this;
            }

            auto operator++(int) = delete;

            auto operator==(const iterator& other) const
                -> bool
            {
                return value == other.value;
            }

            auto operator!=(const iterator& other) const
                -> bool
            {
                return !(*this == other);
            }
        };

        auto begin()
            -> iterator
        {
            return {first};
        }

        auto end()
            -> iterator
        {
            return {last};
        }
    };

} // namespace util::iterator
