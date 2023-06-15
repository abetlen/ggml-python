cmake_minimum_required(VERSION 3.15...3.26)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
)

find_package(
    Python 
    COMPONENTS Interpreter Development.Module
    REQUIRED
)

find_package(
    SWIG
    COMPONENTS python
    REQUIRED
)

include(UseSWIG)

set(BUILD_SHARED_LIBS "On")

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_subdirectory(vendor/ggml)

swig_add_library(
    _ggml
    LANGUAGE python OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}"
    SOURCES _ggml.i
)

target_include_directories(_ggml PUBLIC ${CMAKE_CURRENT_SCOURCE_DIR}/vendor/ggml/include/ggml)
set_property(TARGET _ggml PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)

target_link_libraries(_ggml PRIVATE Python::Module)
target_link_libraries(_ggml PRIVATE ggml)

set_target_properties(_ggml PROPERTIES OUTPUT_NAME _ggml)
set_target_properties(_ggml PROPERTIES SWIG_MODULE_NAME _ggml)
set_target_properties(_ggml PROPERTIES OUTPUT_DIR .)

if(WIN32)
    set_property(TARGET _ggml PROPERTY SUFFIX ".${Python_SOABI}.pyd")
else()
    set_property(TARGET _ggml
                 PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()

install(TARGETS _ggml DESTINATION . LIBRARY DESTINATION .)
