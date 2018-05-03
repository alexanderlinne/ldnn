#pragma once

#include <iosfwd>
#include <string_view>

namespace util::iterator {

    template<class T, class CharT = char, class Traits = std::char_traits<CharT>>
    class ostream_joiner
    {
    public:
        using iterator_category = std::output_iterator_tag;
        using difference_type = ptrdiff_t;
        using value_type = void;
        using reference = void;
        using pointer = void;

        using char_type = CharT;
        using traits_type = Traits;
        using ostream_type = std::basic_ostream<CharT, Traits>;

        using string_view_type = std::basic_string_view<CharT>;

    public:
        ostream_joiner(ostream_type& stream, string_view_type delimiter)
            : ostream_joiner(stream, delimiter, "", "")
        {}

        ostream_joiner(
            ostream_type& stream,
            string_view_type delimiter,
            string_view_type start,
            string_view_type end
        )
            : stream(stream), delimiter(delimiter), start(start), end(end)
        {}

        ~ostream_joiner()
        {
            if (!first) {
                stream << end;
            }
        }

        auto operator++()
            -> ostream_joiner&
        {
            return *this;
        }
        auto operator++(int)
            -> ostream_joiner&
        {
            return *this;
        }
        auto operator*()
            -> ostream_joiner&
        {
            return *this;
        }

        auto operator=(const T& value)
            -> ostream_joiner&
        {
            if (!first) {
                stream << delimiter;
            } else {
                stream << start;
            }
            first = false;

            stream << value;

            return *this;
        }

    private:
        ostream_type& stream;
        bool first = true;
        string_view_type delimiter;
        string_view_type start;
        string_view_type end;
    };

} // namespace util::iterator
