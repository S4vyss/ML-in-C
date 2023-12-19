cmake_minimum_required(VERSION 3.27)
project(ml C)

set(CMAKE_C_STANDARD 23)

add_executable(amabatukam playground.c GoingDeep.c)
target_link_libraries(amabatukam m)