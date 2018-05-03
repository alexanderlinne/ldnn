#pragma once

#include "util/iterator/generator.hpp"

template<class T>
auto range(T first, T last)
{
    return util::iterator::generator<T>{first, last};
}

template<class T>
auto range(T last)
{
    return util::iterator::generator<T>{T{0}, last};
}