cmake_minimum_required(VERSION 3.21)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
)

message(SKBUILD_STATE="${SKBUILD_STATE}")

if(SKBUILD_STATE STREQUAL "editable")
    # Temporary fix for https://github.com/scikit-build/scikit-build-core/issues/374
    set(GGML_PYTHON_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ggml)
else()
    set(GGML_PYTHON_INSTALL_DIR ${SKBUILD_PLATLIB_DIR}/ggml)
endif()

set(BUILD_SHARED_LIBS "On")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
if (APPLE)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(GGML_AVX "Off" CACHE BOOL "ggml: enable AVX" FORCE)
        set(GGML_AVX2 "Off" CACHE BOOL "ggml: enable AVX2" FORCE)
        set(GGML_FMA "Off" CACHE BOOL "ggml: enable FMA" FORCE)
        set(GGML_F16C "Off" CACHE BOOL "ggml: enable F16C" FORCE)
    endif()

    set(GGML_METAL_EMBED_LIBRARY "On" CACHE BOOL "ggml: embed metal library" FORCE)
endif()
add_subdirectory(vendor/ggml)
install(
    TARGETS ggml 
    ARCHIVE DESTINATION ${GGML_PYTHON_INSTALL_DIR}
    LIBRARY DESTINATION ${GGML_PYTHON_INSTALL_DIR}
    RUNTIME DESTINATION ${GGML_PYTHON_INSTALL_DIR}
    FRAMEWORK DESTINATION ${GGML_PYTHON_INSTALL_DIR}
    RESOURCE DESTINATION ${GGML_PYTHON_INSTALL_DIR}
)
install(
    FILES $<TARGET_RUNTIME_DLLS:ggml>
    DESTINATION ${GGML_PYTHON_INSTALL_DIR}
)