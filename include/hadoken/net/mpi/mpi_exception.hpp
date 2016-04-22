/**
 * Copyright (C) 2016 Adrien Devresse
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
*/
#ifndef MPI_EXCEPTION_HPP
#define MPI_EXCEPTION_HPP

#include <stdexcept>
#include <exception>

class mpi_exception: public std::runtime_error {
public:
    inline mpi_exception(int code, const std::string & msg): std::runtime_error(msg), _code(code){}
    inline virtual ~mpi_exception() throw() {}

    int value() const { return _code; }

private:
    int _code;
};

#endif // MPI_EXCEPTION_HPP
