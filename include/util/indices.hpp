#pragma once

#include <range/v3/view/indices.hpp>

template<class T>
auto indices(T first, T last)
{
    return ranges::view::indices(first, last);
}

template<class T>
auto indices(T last)
{
    return ranges::view::indices(T{0}, last);
}