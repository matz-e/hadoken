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
#ifndef _HADOKEN_SPINLOCK_HPP_
#define _HADOKEN_SPINLOCK_HPP_

#include <atomic>
#include <thread>

namespace hadoken {

namespace thread{

///
/// \brief The spin_lock class
///
/// spinlock implementation
///
/// follow the STL requirement for BasicLockable and
/// can consequently be used by STL/boost lock_guard and unique_lock
///
class spin_lock{
public:
    inline spin_lock() : _lock(ATOMIC_FLAG_INIT) {}

    inline void lock() noexcept {
            std::uint64_t counter = 1;
            while(1){
                if(! _lock.test_and_set()){
                    return;
                }
                
#ifndef HADOKEN_SPIN_NO_YIELD
                if(counter % 128 == 0){
                    std::this_thread::yield();                    
                }
#endif                 
                counter ++;
             
           }
    }

    inline void unlock() noexcept{
        _lock.clear();
    }

private:
    spin_lock(const spin_lock &) = delete;
    spin_lock & operator=(const spin_lock&) = delete;

    std::atomic_flag _lock;
};


} // thread


} //hadoken

#endif // SPINLOCK_HPP
