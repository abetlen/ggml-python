cmake_minimum_required(VERSION 3.21)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
)

message(SKBUILD_STATE="${SKBUILD_STATE}")

if(SKBUILD_STATE STREQUAL "editable")
    # Temporary fix for https://github.com/scikit-build/scikit-build-core/issues/374
    set(GGML_PYTHON_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ggml/lib)
else()
    set(GGML_PYTHON_INSTALL_DIR ${SKBUILD_PLATLIB_DIR}/ggml/lib)
endif()

set(BUILD_SHARED_LIBS "On")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
if(EMSCRIPTEN)
    if(DEFINED EMSCRIPTEN_SYSTEM_PROCESSOR)
        set(CMAKE_SYSTEM_PROCESSOR ${EMSCRIPTEN_SYSTEM_PROCESSOR} CACHE STRING "Target processor" FORCE)
    else()
        set(CMAKE_SYSTEM_PROCESSOR wasm32 CACHE STRING "Target processor" FORCE)
    endif()

    set(GGML_NATIVE "Off" CACHE BOOL "ggml: enable -march=native" FORCE)
    set(GGML_OPENMP "Off" CACHE BOOL "ggml: use OpenMP" FORCE)
    set(GGML_METAL "Off" CACHE BOOL "ggml: use Metal" FORCE)
    set(GGML_BLAS "Off" CACHE BOOL "ggml: use BLAS" FORCE)
    set(GGML_CUDA "Off" CACHE BOOL "ggml: use CUDA" FORCE)
    set(GGML_HIP "Off" CACHE BOOL "ggml: use HIP" FORCE)
    set(GGML_VULKAN "Off" CACHE BOOL "ggml: use Vulkan" FORCE)
    set(GGML_OPENCL "Off" CACHE BOOL "ggml: use OpenCL" FORCE)
    set(GGML_RPC "Off" CACHE BOOL "ggml: use RPC" FORCE)

    set(CMAKE_INSTALL_BINDIR ggml/lib CACHE PATH "Install binaries" FORCE)
    set(CMAKE_INSTALL_INCLUDEDIR ggml/include CACHE PATH "Install headers" FORCE)
    set(CMAKE_INSTALL_LIBDIR ggml/lib CACHE PATH "Install libraries" FORCE)
endif()
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
if (APPLE)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(GGML_AVX "Off" CACHE BOOL "ggml: enable AVX" FORCE)
        set(GGML_AVX2 "Off" CACHE BOOL "ggml: enable AVX2" FORCE)
        set(GGML_FMA "Off" CACHE BOOL "ggml: enable FMA" FORCE)
        set(GGML_F16C "Off" CACHE BOOL "ggml: enable F16C" FORCE)
    endif()

    if (NOT DEFINED GGML_METAL)
        set(GGML_METAL "Off" CACHE BOOL "ggml: enable Metal" FORCE)
    endif()

    if (GGML_METAL)
        set(GGML_METAL_EMBED_LIBRARY "On" CACHE BOOL "ggml: embed metal library" FORCE)
    endif()
endif()
add_subdirectory(vendor/ggml)

set(GGML_PYTHON_TARGETS
    ggml
    ggml-base
    ggml-cpu
    ggml-blas
    ggml-cann
    ggml-cuda
    ggml-hexagon
    ggml-hip
    ggml-metal
    ggml-musa
    ggml-opencl
    ggml-openvino
    ggml-rpc
    ggml-sycl
    ggml-vulkan
    ggml-virtgpu
    ggml-webgpu
    ggml-zdnn
    ggml-zendnn
)

foreach(GGML_PYTHON_TARGET IN LISTS GGML_PYTHON_TARGETS)
    if(TARGET ${GGML_PYTHON_TARGET})
        if(EMSCRIPTEN)
            set_target_properties(${GGML_PYTHON_TARGET} PROPERTIES
                OUTPUT_NAME "${GGML_PYTHON_TARGET}.cpython-00-wasm32-emscripten"
            )
        endif()

        if(UNIX AND NOT APPLE)
            set_target_properties(${GGML_PYTHON_TARGET} PROPERTIES
                NO_SONAME TRUE
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            )
        endif()

        install(
            TARGETS ${GGML_PYTHON_TARGET}
            ARCHIVE DESTINATION ${GGML_PYTHON_INSTALL_DIR}
            LIBRARY DESTINATION ${GGML_PYTHON_INSTALL_DIR}
            RUNTIME DESTINATION ${GGML_PYTHON_INSTALL_DIR}
            FRAMEWORK DESTINATION ${GGML_PYTHON_INSTALL_DIR}
            RESOURCE DESTINATION ${GGML_PYTHON_INSTALL_DIR}
        )
    endif()
endforeach()

install(
    FILES $<TARGET_RUNTIME_DLLS:ggml>
    DESTINATION ${GGML_PYTHON_INSTALL_DIR}
)
