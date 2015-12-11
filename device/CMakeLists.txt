cmake_minimum_required(VERSION 3.3)

project(gen_device_hpp)

include(FindCUDA)

cuda_add_executable(gen_device_hpp gen_device_hpp.cu)
add_test(gen_device_hpp ./gen_device_hpp > device.hpp)
