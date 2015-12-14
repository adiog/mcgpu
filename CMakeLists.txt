cmake_minimum_required(VERSION 3.3)

project(mcgpu)

enable_testing()

include(FindCUDA)

set(CMAKE_CXX_COMPILER g++-5.2)
set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_SOURCE_DIR})

# Generate device.hpp header file
add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/device.hpp
        COMMAND
            ${CMAKE_CURRENT_SOURCE_DIR}/script/gen_device_hpp.sh ${CMAKE_CURRENT_SOURCE_DIR}/device ${CMAKE_CURRENT_BINARY_DIR}/device.hpp
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/device/gen_device_hpp.cu
)

# Collect source files
file(GLOB SOURCE_HELPERS ${CMAKE_CURRENT_SOURCE_DIR}/src/mcgpu/helpers/*.cu)
file(GLOB SOURCE_HOST ${CMAKE_CURRENT_SOURCE_DIR}/src/mcgpu/host/*.cpp)
file(GLOB SOURCE_SIMULATION ${CMAKE_CURRENT_SOURCE_DIR}/src/mcgpu/simulation/*.cu)
file(GLOB SOURCE_MARKET ${CMAKE_CURRENT_SOURCE_DIR}/src/mcgpu/model/*.cu)
file(GLOB SOURCE_PAYOFF_ASIAN ${CMAKE_CURRENT_SOURCE_DIR}/src/mcgpu/payoff/asian/*.cu)
file(GLOB SOURCE_PAYOFF_EUROPEAN ${CMAKE_CURRENT_SOURCE_DIR}/src/mcgpu/payoff/european/*.cu)

# Define all source files
set(MCGPU_SOURCE ${SOURCE_HOST} ${SOURCE_HELPERS} ${SOURCE_SIMULATION} ${SOURCE_MARKET} ${SOURCE_PAYOFF_ASIAN} ${SOURCE_PAYOFF_EUROPEAN})

# Concatenate all source files (ugly workaround needed to use pointer to function on device)
add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/mcgpu.cu
        COMMAND
            ${CMAKE_CURRENT_SOURCE_DIR}/script/collect.sh ${CMAKE_CURRENT_BINARY_DIR}/mcgpu.cu ${MCGPU_SOURCE}
        DEPENDS
            ${MCGPU_SOURCE}
)

# Main library
cuda_add_library(mcgpu SHARED ${CMAKE_CURRENT_BINARY_DIR}/mcgpu.cu ${CMAKE_CURRENT_BINARY_DIR}/device.hpp)

# Temp executables
add_executable(bs_asian_call test/blackscholes/asian_call.cpp)
target_link_libraries(bs_asian_call mcgpu gtest)
add_test(bs_asian_call bs_asian_call)

add_executable(bs_asian_barrier test/blackscholes/asian_barrier.cpp)
target_link_libraries(bs_asian_barrier mcgpu gtest)
add_test(bs_asian_barrier bs_asian_barrier)

add_executable(bs_asian_paris test/blackscholes/asian_paris.cpp)
target_link_libraries(bs_asian_paris mcgpu gtest)
add_test(bs_asian_paris bs_asian_paris)

add_executable(bs_european_call test/blackscholes/european_call.cpp)
target_link_libraries(bs_european_call mcgpu gtest)
add_test(bs_european_call bs_european_call)

add_executable(bs_european_put test/blackscholes/european_put.cpp)
target_link_libraries(bs_european_put mcgpu gtest)
add_test(bs_european_put bs_european_put)

add_executable(bs_european_bullspread test/blackscholes/european_bullspread.cpp)
target_link_libraries(bs_european_bullspread mcgpu gtest)
add_test(bs_european_bullspread bs_european_bullspread)

add_executable(bs_european_binary test/blackscholes/european_binary.cpp)
target_link_libraries(bs_european_binary mcgpu gtest)
add_test(bs_european_binary bs_european_binary)

# Benchmarks
add_executable(bs_basic_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/test/benchmark/blackscholes_basic_benchmark.cpp)
target_link_libraries(bs_basic_benchmark mcgpu benchmark)
add_test(bs_basic_benchmark bs_basic_benchmark)
