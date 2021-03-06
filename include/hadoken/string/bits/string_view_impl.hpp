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
#include <cassert>
#include <limits>

namespace hadoken {

inline string_view::string_view() noexcept :
    _pstr(nullptr),
    _len(0){

}

inline string_view::string_view(const std::string &str) noexcept :
    _pstr(str.c_str()),
    _len(str.size()){

}

inline string_view::string_view(const char *c_str, std::size_t length) :
    _pstr(c_str),
    _len(length){

}

inline string_view::string_view(const char *c_str) :
    _pstr(c_str),
    _len(strlen(c_str)){

}



inline string_view::const_iterator string_view::begin() const{
    return _pstr;
}

inline string_view::const_iterator string_view::end() const{
    return _pstr + _len;
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


inline int string_view::compare(const string_view &other) const noexcept{
    const size_type min_size = std::min(_len, other._len);

    for(size_type i =0; i < min_size; ++i){
        if(_pstr[i] < other._pstr[i])
            return -1;
        if(_pstr[i] > other._pstr[i])
            return 1;
    }
    if (_len < other._len) return -1;
    if (_len > other._len) return 1;
    return 0;
}

inline string_view::const_pointer string_view::data() const noexcept{
    return _pstr;
}

inline char string_view::operator [](std::size_t pos) const{
    assert(pos < _len);
    return _pstr[pos];
}


inline void string_view::swap(string_view &other) noexcept{
    using std::swap;
    swap(_pstr, other._pstr);
    swap(_len, other._len);
}


inline bool operator==(const string_view & first, const string_view & second){
    return first.compare(second) == 0;
}

inline std::ostream & operator <<(std::ostream & o, const string_view & sv){
    o.write(sv._pstr, sv._len);
    return o;
}

inline std::string to_string(const string_view &sv){
    return std::string(sv.data(), sv.size());
}

} // hadoken

#endif // STRING_VIEW_IMPL_HPP
