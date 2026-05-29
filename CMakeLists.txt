cmake_minimum_required(VERSION 3.24)

option(HYDRA_TARGET_SM120 "Build for compute capability 12.0 / sm_120" ON)
option(HYDRA_TARGET_SM89 "Build for compute capability 8.9 / sm_89" ON)
option(HYDRA_TARGET_SM86 "Build for compute capability 8.6 / sm_86" ON)

set(HYDRA_CUDA_ARCHITECTURES "")
if(HYDRA_TARGET_SM120)
    list(APPEND HYDRA_CUDA_ARCHITECTURES 120)
endif()
if(HYDRA_TARGET_SM89)
    list(APPEND HYDRA_CUDA_ARCHITECTURES 89)
endif()
if(HYDRA_TARGET_SM86)
    list(APPEND HYDRA_CUDA_ARCHITECTURES 86)
endif()

if(NOT HYDRA_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "Enable at least one HYDRA_TARGET_SM* option.")
endif()

set(CMAKE_CUDA_ARCHITECTURES "${HYDRA_CUDA_ARCHITECTURES}")

project(Hydra LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenSSL REQUIRED)

add_executable(Hydra
    src/Hydra.cu
    src/Platform.cpp
    src/HttpClient.cpp
    src/RuleParser.cpp
)

target_include_directories(Hydra PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(Hydra PROPERTIES
    CUDA_ARCHITECTURES "${HYDRA_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

target_compile_options(Hydra PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release,RelWithDebInfo>>:-O2>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release,RelWithDebInfo>>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(Hydra PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    )
endif()

if(WIN32)
    target_link_libraries(Hydra PRIVATE OpenSSL::SSL OpenSSL::Crypto ws2_32)
else()
    target_link_libraries(Hydra PRIVATE ${OPENSSL_SSL_LIBRARY} ${OPENSSL_CRYPTO_LIBRARY})
endif()

message(STATUS "Hydra CUDA architectures: ${HYDRA_CUDA_ARCHITECTURES} (release default: sm_86 + sm_89 + sm_120)")
message(STATUS "Hydra Windows build: enabled with OpenSSL and Winsock")
