/**
 * Copyright (c) 2018, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 * Boost Software License - Version 1.0
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
*
*/
#ifndef HADOKEN_STRING_VIEW_IMPL_HPP
#define HADOKEN_STRING_VIEW_IMPL_HPP

#include "../string_view.hpp"

#include <cstring>
#include <limits>

namespace hadoken {

inline string_view::string_view() noexcept :
    _pstr(nullptr),
    _len(0){

}

inline string_view::string_view(const char *c_str, std::size_t length) :
    _pstr(c_str),
    _len(length){

}

inline string_view::string_view(const char *c_str) :
    _pstr(c_str),
    _len(strlen(c_str)){

}


inline string_view::size_type string_view::size() const noexcept{
    return _len;
}

inline string_view::size_type string_view::length() const noexcept{
    return size();
}

inline bool string_view::empty() const noexcept{
    return (_len == 0);
}

inline string_view::size_type string_view::max_size() const noexcept{
    return std::numeric_limits<decltype(_len)>::max();
}


inline void string_view::swap(string_view &other) noexcept{
    using std::swap;
    swap(_pstr, other._pstr);
    swap(_len, other._len);
}

} // hadoken

#endif // STRING_VIEW_IMPL_HPP
