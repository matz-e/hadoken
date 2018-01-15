/**
 * Copyright (c) 2016, Adrien Devresse <adrien.devresse@epfl.ch>
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

#define BOOST_TEST_MODULE stringTests
#define BOOST_TEST_MAIN

#include <iostream>
#include <map>
#include <stdexcept>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>


#include <hadoken/string/string_view.hpp>

#include <hadoken/utility/range.hpp>


#include "test_helpers.hpp"



BOOST_AUTO_TEST_CASE( small_vector_test_unique_ptr)
{
    using namespace hadoken;

    const char* msg = "hello bob #42~€é";

    string_view empty, truncated(msg, 5), full(msg);

    BOOST_CHECK_EQUAL(empty.empty(), true);
    BOOST_CHECK_EQUAL(truncated.empty(), false);
    BOOST_CHECK_EQUAL(full.empty(), false);


    BOOST_CHECK_EQUAL(empty.size(), 0);
    BOOST_CHECK_EQUAL(empty.size(), empty.length());
    BOOST_CHECK_EQUAL(truncated.size(), 5);
    BOOST_CHECK_EQUAL(truncated.size(), truncated.length());
    BOOST_CHECK_EQUAL(full.size(), strlen(msg));
}

