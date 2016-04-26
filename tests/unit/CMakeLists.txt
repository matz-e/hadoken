
if(UNIT_TESTS)

find_package(Boost 1.41.0 REQUIRED system unit_test_framework)



include_directories(${CMAKE_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS} )

link_directories( ${Boost_LIBRARY_DIRS} )


## generic Test
LIST(APPEND test_generic_src "test_generic.cpp")

add_executable(test_generic ${test_generic_src} ${HADOKEN_HEADERS_1} ${HADOKEN_HEADERS_2})
target_link_libraries(test_generic ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES})

add_test(NAME test_generic_unit COMMAND ${TESTS_PREFIX} ${TESTS_PREFIX_ARGS} ${CMAKE_CURRENT_BINARY_DIR}/test_generic)



if(HADOKEN_MPI)
find_package(MPI)


## mpi Test
LIST(APPEND MpiRuntime_src "mpi_test.cpp")

include_directories(${MPI_C_INCLUDE_PATH})

add_executable(mpi_test ${MpiRuntime_src})
target_link_libraries(mpi_test ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} ${Boost_SYSTEM_LIBRARIES} ${MPI_LIBRARIES})

add_test(NAME MpiTest_unit COMMAND ${TESTS_PREFIX} ${TESTS_PREFIX_ARGS} ${CMAKE_CURRENT_BINARY_DIR}/mpi_test)


endif()

endif()
