cmake_minimum_required(VERSION 3.25)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(csrc)

