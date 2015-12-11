cmake_minimum_required(VERSION 3.3)

project(mcgpu)

include(FindCUDA)

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
set(MCGPU_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/maintest/maintest.cu ${SOURCE_HOST} ${SOURCE_HELPERS} ${SOURCE_SIMULATION} ${SOURCE_MARKET} ${SOURCE_PAYOFF_ASIAN} ${SOURCE_PAYOFF_EUROPEAN})

# Concatenate all source files (ugly workaround needed to use pointer to function on device)
add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/mcgpu.cu
        COMMAND
            ${CMAKE_CURRENT_SOURCE_DIR}/script/collect.sh ${CMAKE_CURRENT_BINARY_DIR}/mcgpu.cu ${MCGPU_SOURCE}
        DEPENDS
            ${MCGPU_SOURCE}
)

# Main executable
cuda_add_executable(mcgpu ${CMAKE_CURRENT_BINARY_DIR}/mcgpu.cu ${CMAKE_CURRENT_BINARY_DIR}/device.hpp)

