cmake_minimum_required(VERSION 3.6)

project(mera VERSION 1.3.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MERA_TVM_ROOT "../mera-tvm")

if(NOT DEFINED ENV{TVM_LIBRARY_PATH})
message(FATAL_ERROR "TVM_LIBRARY_PATH environment has not been set.")
endif()

file(GLOB SRC src/*.cc)

add_library(meratvm SHARED IMPORTED)
set_property(TARGET meratvm PROPERTY IMPORTED_LOCATION $ENV{TVM_LIBRARY_PATH}/libmeratvm_runtime.so)

add_library(mera SHARED ${SRC})
target_include_directories(mera PUBLIC include)
target_include_directories(mera PRIVATE ${MERA_TVM_ROOT}/3rdparty/dlpack/include)
target_include_directories(mera PRIVATE ${MERA_TVM_ROOT}/3rdparty/dmlc-core/include)
target_include_directories(mera PRIVATE ${MERA_TVM_ROOT}/include)
target_link_libraries(mera meratvm stdc++fs)

# Apps

add_executable(inference apps/inference.cpp)
target_include_directories(inference PUBLIC include)
target_link_libraries(inference mera)
