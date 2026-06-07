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
