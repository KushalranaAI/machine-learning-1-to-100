cmake_minimum_required(VERSION 3.14)
project(mlcpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(mlcpp INTERFACE)
target_include_directories(mlcpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
  target_compile_definitions(mlcpp INTERFACE _USE_MATH_DEFINES)
  target_compile_options(mlcpp INTERFACE /W4)
else()
  target_compile_options(mlcpp INTERFACE -Wall -Wextra)
endif()

# The examples read data/iris.csv from the repository root.
add_compile_definitions(MLCPP_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data")

foreach(example linear_regression logistic_regression knn naive_bayes decision_tree kmeans pca neural_network)
  add_executable(${example} examples/${example}.cpp)
  target_link_libraries(${example} PRIVATE mlcpp)
endforeach()

enable_testing()
add_executable(test_mlcpp tests/test_mlcpp.cpp)
target_link_libraries(test_mlcpp PRIVATE mlcpp)
add_test(NAME mlcpp_tests COMMAND test_mlcpp)
