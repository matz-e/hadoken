#define BOOST_TEST_MODULE mpiTests
#define BOOST_TEST_MAIN

#include <boost/test/included/unit_test.hpp>

#include <hadoken/net/mpi/mpi.hpp>

int argc = boost::unit_test::framework::master_test_suite().argc;
char ** argv = boost::unit_test::framework::master_test_suite().argv;

using namespace hadoken::mpi;


struct MpiFixture{
    MpiFixture():  _env(&argc, &argv){

    }

    ~MpiFixture(){

    }

    mpi_scope_env _env;
};



BOOST_GLOBAL_FIXTURE( MpiFixture);


template<typename T>
T conv_or_max_integral(const int val){
    if( std::numeric_limits<T>::digits >= std::numeric_limits<int>::digits ){
        return T(val);
    }
    if( val > int(std::numeric_limits<T>::max()) ){
        return std::numeric_limits<T>::max();
    }
    return T(val);
}


BOOST_AUTO_TEST_CASE( mpiTests )
{

    mpi_comm runtime;
    const int rank = runtime.rank();
    const int size = runtime.size();

    BOOST_CHECK(rank >=0);
    BOOST_CHECK(size > 0);
    BOOST_CHECK(rank < size);

    std::cout << " rank:" << rank << " size:" << size;

    runtime.barrier();

}


BOOST_AUTO_TEST_CASE( mpiMax)
{

    mpi_comm runtime;
    const int rank = runtime.rank();
    const int size = runtime.size();

    int proc_number = (rank+1)*10;
    int max_proc_number = 0;

    max_proc_number = runtime.all_max(proc_number);

    BOOST_CHECK(max_proc_number == (size)*10);




}


BOOST_AUTO_TEST_CASE( mpiSum)
{

    mpi_comm runtime;

    const int rank = runtime.rank();
    const int size = runtime.size();

    int proc_number = (rank+1)*10;
    int sum_proc_number=0;
    for(int i =1; i < runtime.size()+1; ++i)
        sum_proc_number+=i*10;

    int sum_all = runtime.all_sum(proc_number);


    size_t sum_size = runtime.all_sum(static_cast<size_t>(size));


    BOOST_CHECK(sum_proc_number == sum_all);
    BOOST_CHECK(sum_size == static_cast<size_t>(size)* static_cast<size_t>(size));


}


BOOST_AUTO_TEST_CASE( mpi_all_gather_int)
{

    mpi_comm runtime;

    int rank = runtime.rank();

    std::vector<int> vals = runtime.all_gather(rank);

    BOOST_CHECK(vals.size() >=1 );
    BOOST_CHECK(int(vals.size()) == runtime.size());

    for(int i = 0 ; i < int(vals.size()); ++i){
        BOOST_CHECK_EQUAL(vals[i], i);
    }

}


typedef boost::mpl::list<char, unsigned char,
                        int, unsigned int,
                        long, unsigned long,
                        long long, unsigned long long,
                        float, double,
                        long double> test_types;

BOOST_AUTO_TEST_CASE_TEMPLATE( mpi_all_gather_number, T, test_types )
{
    mpi_comm runtime;

    int rank = runtime.rank();

    std::vector<T> vals = runtime.all_gather<T>( conv_or_max_integral<T>(rank) );

    BOOST_CHECK(vals.size() >=1 );
    BOOST_CHECK(int(vals.size()) == runtime.size());

    for(int i = 0 ; i < int(vals.size()); ++i){
        BOOST_CHECK_EQUAL(vals[i], conv_or_max_integral<T>(i) );
    }
}


BOOST_AUTO_TEST_CASE( mpi_send_recv_ring_int )
{
    mpi_comm runtime;

    if(runtime.size() ==1){
        std::cout << "Only one single node mpi_send_recv_ring can not be executed\n";
        return;
    }

    int rank = runtime.rank();
    int next_rank = ((rank +1 == runtime.size())?0:rank+1);

    if(runtime.is_master())
        runtime.send(0, next_rank, 256);

    int v;
    runtime.recv(any_source, any_tag, v);

    v = v+1;

    std::cout << "recv_val:" << v << "\n";

    if(runtime.is_master() == false){
        runtime.send(v, next_rank, 256);
    }


    if(runtime.is_master()){
        BOOST_CHECK_EQUAL(v, runtime.size() );
    } else{
        BOOST_CHECK_EQUAL(v, rank );
    }

}


BOOST_AUTO_TEST_CASE_TEMPLATE( mpi_send_recv_ring, T, test_types )
{
    mpi_comm runtime;

    if(runtime.size() ==1){
        std::cout << "Only one single node mpi_send_recv_ring can not be executed\n";
        return;
    }

    int rank = runtime.rank();
    int next_rank = ((rank +1 == runtime.size())?0:rank+1);

    if(runtime.is_master())
        runtime.send(T(0), next_rank, 256);

    T v;
    runtime.recv(any_source, any_tag, v);

    v = (( std::numeric_limits<T>::max() == (v) )?v:v+1);

    std::cout << "recv_val:" << v << "\n";

    if(runtime.is_master() == false){
        runtime.send(v, next_rank, 256);
    }



    if(runtime.is_master()){
        BOOST_CHECK_EQUAL(v, conv_or_max_integral<T>(runtime.size()) );
    } else{
        BOOST_CHECK_EQUAL(v, conv_or_max_integral<T>(rank) );
    }

}

