#pragma once

#include "iterator/generator.hpp"

template<class T>
auto indices(T first, T last)
{
    return util::iterator::generator<T>{first, last};
}

template<class T>
auto indices(T last)
{
    return util::iterator::generator<T>{T{0}, last};
}