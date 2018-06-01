cmake_minimum_required(VERSION 3.5)
project(ldnn)
set(TARGET ldnn)

set(CMAKE_GENERATOR "Unix Makefiles")

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_CXX_STANDARD 14)
list(APPEND CMAKE_CXX_FLAGS "-std=c++14 -O3 -march=native -Wall")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third-party/cxxopts/include)
include_directories(${PROJECT_SOURCE_DIR}/third-party/inih/cpp)

# add pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
list(APPEND LIBRARIES Threads::Threads)

set(SOURCE_FILES
    third-party/inih/ini.c
    third-party/inih/cpp/INIReader.cpp
    src/main.cpp
)

add_executable(
    ${TARGET}
    ${SOURCE_FILES}
)

target_link_libraries(
    ${TARGET}
    ${LIBRARIES}
)