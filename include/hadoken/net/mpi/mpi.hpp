/**
 * Copyright (C) 2016 Adrien Devresse
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
*/
#ifndef HADOKEN_MPI_COMM_HPP
#define HADOKEN_MPI_COMM_HPP

#include <mpi.h>

#include <boost/noncopyable.hpp>
#include "mpi_exception.hpp"


namespace hadoken{

namespace mpi {

/**
 * @brief mpi environment scoper
 *
 *  initialize MPI when constructed, finalize MPI when destroyed
 */
class mpi_scope_env : private boost::noncopyable{
public:
    inline mpi_scope_env(int* argc, char*** argv);

    inline virtual ~mpi_scope_env();

private:
    int initialized;
};

/**
 * @brief mpi communicator
 *
 * bridge to all the MPI functions
 */
class mpi_comm : private boost::noncopyable
{
public:

    /**
     * @brief mpi_comm
     *
     * construct an MPI communication channel
     *
     * MPI operation
     */

    inline mpi_comm();
    inline virtual ~mpi_comm();

    /**
     * @brief rank
     * @return mpi rank of the process
     */
    inline int rank() const{
        return _rank;
    }

    /**
     * @brief size
     * @return size of the mpi communication domain
     */
    inline int size() const{
        return _size;
    }


    /**
     * @brief isMaster
     * @return true if current node is root ( 0 )
     */
    inline bool is_master() const{
        return _rank ==0;
    }

    /// @brief synchronization barrier
    ///
    /// blocking until all the nodes reach it
    inline void barrier();


    /// send local_value to node dest
    /// @param local_value value to send
    /// @param dest node id of the reciver
    /// @param tag identity tag
    template <typename T>
    inline void send(const T & local_value, int dest_node, int tag);



    /// received data from an other node
    /// @param src node id of the sender
    /// @param tag identity tag
    /// @return value received
    template <typename T>
    inline void recv(int src_node, int tag, T & value);


    /// return gathered information from all nodes
    /// to all nodes
    ///
    /// collective operation
    template <typename T>
    inline std::vector<T> all_gather(const T & local_value);


    /// @brief return the highest element of the nodes
    /// @param local_value value to compare
    /// @return highest element between all nodes
    ///
    /// collective operation
    template <typename T>
    inline T all_max(const T & local_value);


    /// @brief return the sum of all the elements of the nodes
    /// @param local_value value to sum
    /// @return sum of all the elements
    ///
    /// collective operation
    template <typename T>
    inline T all_sum(const T & local_value);

private:
    int _rank;
    int _size;
    MPI_Comm _comm;

};



/// various const definitions

const int any_tag = MPI_ANY_TAG;
const int any_source = MPI_ANY_SOURCE;

}

}

#include "impl/mpi.tcc"

#endif
