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


#include <mutex>
#include <thread>
#include <future>

#include <boost/test/floating_point_comparison.hpp>


#include <boost/random.hpp>
#include <boost/chrono.hpp>

#include <hadoken/thread/spinlock.hpp>
#include <hadoken/format/format.hpp>


using namespace boost::chrono;

typedef  system_clock::time_point tp;
typedef  system_clock cl;


template<typename LockType>
std::size_t lock_test(std::size_t n_thread, const std::string & lock_name){

	const std::size_t iter = 200000;

    tp t1, t2;

    t1 = cl::now();

    const std::string msg = "hello world, ";
    std::vector<std::future<void> > res;
    double a = 0.0, inc = 1.0;
    LockType lock;

    using namespace hadoken::thread;


    for(std::size_t i =0; i < n_thread; ++i){
        res.emplace_back(
            std::async(std::launch::async, [&] {
            for(std::size_t j =0; j < iter; ++j){
                std::lock_guard<LockType> guard(lock);
                a += inc;
				inc += 1.0;
            }
        }));
    }

    for(auto & f : res){
        f.wait();
    }

    t2 = cl::now();

    std::cout << lock_name << ": " << boost::chrono::duration_cast<milliseconds>(t2 -t1) << std::endl;

    return std::size_t(a);
}



int main(){

	const std::size_t ncore = std::thread::hardware_concurrency();
    std::size_t junk=0;

    hadoken::format::scat(std::cout, "test lock with ", ncore, " cores");

    junk += lock_test<std::mutex>(1, "std::mutex_single");

    junk += lock_test<hadoken::thread::spin_lock>(1, "hadoken::thread::spinlock_single");

    junk += lock_test<std::mutex>(ncore/2, "std::mutex_thread=core/2");

    junk += lock_test<hadoken::thread::spin_lock>(ncore/2, "hadoken::thread::spinlock_thread=core/2");


    junk += lock_test<std::mutex>(ncore, "std::mutex_thread=core");

    junk += lock_test<hadoken::thread::spin_lock>(ncore, "hadoken::thread::spinlock_thread=core");

    junk += lock_test<std::mutex>(2*ncore, "std::mutex_thread=2*core");

    junk += lock_test<hadoken::thread::spin_lock>(2*ncore, "hadoken::thread::spinlock_thread=2*core");

   std::cout << "end junk " << junk << std::endl;

}
