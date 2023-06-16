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
    libggml
    TYPE USE_BUILD_SHARED_LIBS
    LANGUAGE python
    OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}/ggml"
    SOURCES _ggml.i
)

target_include_directories(libggml PUBLIC ${CMAKE_CURRENT_SCOURCE_DIR}/vendor/ggml/include/ggml)
set_property(TARGET libggml PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
set_property(TARGET libggml PROPERTY SWIG_DEPENDS ggml)

target_link_libraries(libggml PRIVATE Python::Module)
target_link_libraries(libggml PRIVATE ggml)

if(WIN32)
    set_property(TARGET libggml PROPERTY SUFFIX ".${Python_SOABI}.pyd")
else()
    set_property(
        TARGET libggml
        PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}"
    )
endif()

install(
    TARGETS libggml 
    DESTINATION . 
)
