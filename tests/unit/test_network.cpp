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

#define BOOST_TEST_MODULE networkTests
#define BOOST_TEST_MAIN

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <future>
#include <algorithm>
#include <random>
#include <numeric>

#include <chrono>

#include <boost/test/unit_test.hpp>

#include <hadoken/network/uri.hpp>



using namespace hadoken;
using cl = std::chrono::system_clock;

BOOST_AUTO_TEST_CASE( uri_parsing)
{
    // test full url

    const std::string valid_uri_simple("http://johndoe:password@example.org:1443/mypath");
    uri my_uri(valid_uri_simple);

    BOOST_CHECK(my_uri.is_valid());
    BOOST_CHECK_EQUAL(my_uri.get_scheme(), "http");
    BOOST_CHECK_EQUAL(my_uri.get_userinfo(), "johndoe:password");
    BOOST_CHECK_EQUAL(my_uri.get_host(), "example.org");
    BOOST_CHECK_EQUAL(my_uri.get_port(), 1443);
    BOOST_CHECK_EQUAL(my_uri.get_path(), "/mypath");


    // test url without userinfo and port
    const std::string no_userinfo_uri_simple("torrent://w3c.org/hello/world/here");
    uri no_userinfo_uri(no_userinfo_uri_simple);

    BOOST_CHECK(no_userinfo_uri.is_valid());
    BOOST_CHECK_EQUAL(no_userinfo_uri.get_scheme(), "torrent");
    BOOST_CHECK_EQUAL(no_userinfo_uri.get_userinfo(), "");
    BOOST_CHECK_EQUAL(no_userinfo_uri.get_host(), "w3c.org");
    BOOST_CHECK_EQUAL(no_userinfo_uri.get_port(), 0);
    BOOST_CHECK_EQUAL(no_userinfo_uri.get_path(), "/hello/world/here");


    // test url without path
    const std::string no_path_uri_str("s3s://bucket.org");
    uri no_path_rui(no_path_uri_str);

    BOOST_CHECK(no_path_rui.is_valid());
    BOOST_CHECK_EQUAL(no_path_rui.get_scheme(), "s3s");
    BOOST_CHECK_EQUAL(no_path_rui.get_userinfo(), "");
    BOOST_CHECK_EQUAL(no_path_rui.get_host(), "bucket.org");
    BOOST_CHECK_EQUAL(no_path_rui.get_port(), 0);
    BOOST_CHECK_EQUAL(no_path_rui.get_path(), "");


    // test magnet urn
    const std::string magnet_urn("magnet:?xt.1=urn:sha1:YNCKHTQCWBTRNJIV4WNAE52SJUQCZO5C&xt.2=urn:sha1:TXGCZQTH26NL6OUQAJJPFALHG2LTGBC7");
    uri magnet_uri(magnet_urn);
    BOOST_CHECK(my_uri.is_valid());
    BOOST_CHECK_EQUAL(magnet_uri.get_scheme(), "magnet");
    BOOST_CHECK_EQUAL(magnet_uri.get_userinfo(), "");
    BOOST_CHECK_EQUAL(magnet_uri.get_host(), "");
    BOOST_CHECK_EQUAL(magnet_uri.get_port(), 0);
    BOOST_CHECK_EQUAL(magnet_uri.get_path(), "");


    // test file url
    const std::string file_uri_str("file:///etc/services");
    uri file_uri(file_uri_str);
    BOOST_CHECK(file_uri.is_valid());
    BOOST_CHECK_EQUAL(file_uri.get_scheme(), "file");
    BOOST_CHECK_EQUAL(file_uri.get_userinfo(), "");
    BOOST_CHECK_EQUAL(file_uri.get_host(), "");
    BOOST_CHECK_EQUAL(file_uri.get_port(), 0);
    BOOST_CHECK_EQUAL(file_uri.get_path(), "/etc/services");


    // random invalid string
    const std::string false_str("joe la mouk");
    uri false_uri(false_str);
    BOOST_CHECK(false_uri.is_valid() == false);

    BOOST_CHECK_THROW({
                          false_uri.get_scheme();
                      }, std::invalid_argument);

    BOOST_CHECK_THROW({
                          false_uri.get_userinfo();
                      }, std::invalid_argument);

    BOOST_CHECK_THROW({
                          false_uri.get_host();
                      }, std::invalid_argument);

    BOOST_CHECK_THROW({
                          false_uri.get_path();
                      }, std::invalid_argument);

    BOOST_CHECK_THROW({
                          false_uri.get_query();
                      }, std::invalid_argument);


    BOOST_CHECK_THROW({
                          false_uri.get_fragment();
                      }, std::invalid_argument);

    // invalid scheme
    const std::string broken_scheme_str("fil€€:///tmp");
    uri broken_scheme(broken_scheme_str);
    BOOST_CHECK(broken_scheme.is_valid() == false);

}



BOOST_AUTO_TEST_CASE( test_encode)
{
    std::string res;

    res = percent_encode("hello_world");

    BOOST_CHECK_EQUAL(res, "hello_world");

    res = percent_decode(res);

    BOOST_CHECK_EQUAL(res, "hello_world");


    res= percent_encode("il pleut, il pleut bergère, rentre_t€s bläncs mOutön$");

    BOOST_CHECK_EQUAL(res, "il%20pleut%2C%20il%20pleut%20berg%C3%A8re%2C%20rentre_t%E2%82%ACs%20bl%C3%A4ncs%20mOut%C3%B6n%24");

    res = percent_decode(res);


  //  BOOST_CHECK_EQUAL(res, "il pleut, il pleut bergère, rentre_t€s bläncs mOutön$");


    res = percent_encode("私はガラスを食べられます。それは私を傷つけません");

   BOOST_CHECK_EQUAL(res, "%E7%A7%81%E3%81%AF%E3%82%AC%E3%83%A9%E3%82%B9%E3%82%92%E9%A3%9F%E3%81%B9%E3%82%89%E3%82%8C%E3%81%BE%E3%81%99%E3%80%82%E3%81%9D%E3%82%8C%E3%81%AF%E7%A7%81%E3%82%92%E5%82%B7%E3%81%A4%E3%81%91%E3%81%BE%E3%81%9B%E3%82%93");
}
