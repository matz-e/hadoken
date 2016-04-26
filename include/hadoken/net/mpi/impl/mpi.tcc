/**
 * Copyright (C) 2016 Adrien Devresse
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
*/
#ifndef HADOKEN_MPI_COMM_IMPL_HPP
#define HADOKEN_MPI_COMM_IMPL_HPP

#include <errno.h>

#include <mpi.h>

#include <boost/atomic.hpp>
#include <boost/array.hpp>

#include "../mpi.hpp"
#include "../mpi_exception.hpp"

namespace hadoken{


namespace mpi{

namespace impl{


inline std::string thread_opt_string(int opt_string){
    const boost::array<std::string,4> val_str = { { "MPI_THREAD_SINGLE", "MPI_THREAD_FUNNELED", "MPI_THREAD_SERIALIZED",
                              "MPI_THREAD_MULTIPLE" } };
    const boost::array<int, 4>  val = { { MPI_THREAD_SINGLE , MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED , MPI_THREAD_MULTIPLE } };

    for(std::size_t i=0; i < val.size(); ++i){
        if(val[i] == opt_string){
            return val_str[i];
        }
    }
    return "UNKNOWN";
}

/*
template<typename T>
inline MPI_Datatype _mpi_datatype_mapper(T value){
    (void) value;
    throw mpi_exception(EINVAL, "Invalid MPI Datatype");
}*/


inline MPI_Datatype _mpi_datatype_mapper(char value){
    (void) value;
    return MPI_CHAR;
}

inline MPI_Datatype _mpi_datatype_mapper(unsigned char value){
    (void) value;
    return MPI_BYTE;
}


inline MPI_Datatype _mpi_datatype_mapper(int value){
    (void) value;
    return MPI_INT;
}

inline MPI_Datatype _mpi_datatype_mapper(unsigned int value){
    (void) value;
    return MPI_UNSIGNED;
}


inline MPI_Datatype _mpi_datatype_mapper(long value){
    (void) value;
    return MPI_LONG;
}

inline MPI_Datatype _mpi_datatype_mapper(unsigned long value){
    (void) value;
    return MPI_UNSIGNED_LONG;
}

inline MPI_Datatype _mpi_datatype_mapper(long long value){
    (void) value;
    return MPI_LONG_LONG;
}

inline MPI_Datatype _mpi_datatype_mapper(unsigned long long value){
    (void) value;
    return MPI_UNSIGNED_LONG_LONG;
}

inline MPI_Datatype _mpi_datatype_mapper(float value){
    (void) value;
    return MPI_FLOAT;
}

inline MPI_Datatype _mpi_datatype_mapper(double value){
    (void) value;
    return MPI_DOUBLE;
}

inline MPI_Datatype _mpi_datatype_mapper(long double value){
    (void) value;
    return MPI_LONG_DOUBLE;
}



template<typename T>
inline T _mpi_reduce_mapper(const T & pvalue, MPI_Datatype data_type, MPI_Op operation, const MPI_Comm comm){
    T res;
    const T errcode = MPI_Allreduce(static_cast<void *>(const_cast<T*>(&pvalue)),
                                    static_cast<void*>(&res),
                                    1, data_type, operation, comm);
    assert(errcode == MPI_SUCCESS);
    return res;
}


} //impl



mpi_scope_env::mpi_scope_env(int *argc, char ***argv) : initialized(false){
    MPI_Initialized(&initialized);

    if(!initialized){
        const int thread_support= MPI_THREAD_MULTIPLE;
        int provided;
        if( MPI_Init_thread(argc, argv, thread_support, &provided) != MPI_SUCCESS ){
            throw mpi_exception(EINVAL, std::string("Unable to init MPI with ") + impl::thread_opt_string(thread_support));
        }
        if(provided != thread_support){
            std::cerr << "mpi_scope_env(MPI_Init_thread): MPI Thread level provided (" << impl::thread_opt_string(provided)
                         << ") different of required (" << impl::thread_opt_string(thread_support) << ")\n";
        }
    }
}


mpi_scope_env::~mpi_scope_env(){
    // Finalize only if initialized in this scope
    if(!initialized){
        MPI_Finalize();
    }
}

mpi_comm::mpi_comm() :
    _rank(0),
    _size(0),
    _comm(MPI_COMM_WORLD)
{

    MPI_Comm_rank(_comm, &_rank);
    MPI_Comm_size(_comm, &_size);

}



inline mpi_comm::~mpi_comm(){
    //
}





// MPI All Reduce wrappers

template <typename T>
inline T mpi_comm::all_max(const T & pvalue){
    return impl::_mpi_reduce_mapper(pvalue, impl::_mpi_datatype_mapper(pvalue), MPI_MAX, _comm);
}


template <typename T>
inline T mpi_comm::all_sum(const T & pvalue){
    return impl::_mpi_reduce_mapper(pvalue, impl::_mpi_datatype_mapper(pvalue), MPI_SUM, _comm);
}


// all_gather / all_gatherv
template <typename T>
inline std::vector<T> mpi_comm::all_gather(const T & local_value){
    std::vector<T> res(size());

    if( MPI_Allgather(static_cast<void*>(const_cast<T*>(&local_value)), 1, impl::_mpi_datatype_mapper(local_value),
                  &(res[0]), 1, impl::_mpi_datatype_mapper(local_value), _comm) != MPI_SUCCESS){
        throw mpi_exception(ECOMM, "Error during MPI_Allgather() ");
    }
    return res;
}



template <typename T>
inline void mpi_comm::send(const T & local_value, int dest_node, int tag){

    if( MPI_Send(static_cast<void*>(const_cast<T*>(&local_value)), 1, impl::_mpi_datatype_mapper(local_value),
                  dest_node, tag, _comm) != MPI_SUCCESS){
        throw mpi_exception(ECOMM, "Error during MPI_send() ");
    }
}

template <typename T>
inline void mpi_comm::recv(int src_node, int tag, T & value){
    MPI_Status status;

    if( MPI_Recv(&value, 1, impl::_mpi_datatype_mapper(value), src_node, tag, _comm, &status)  != MPI_SUCCESS){
        throw mpi_exception(ECOMM, "Error during MPI_recv() ");
    }
}


inline void mpi_comm::barrier(){
    if(MPI_Barrier(_comm) != MPI_SUCCESS){
        throw mpi_exception(ECOMM, "Error during MPI_barrier() ");
    }
}


} // mpi


} //hadoken

#endif



