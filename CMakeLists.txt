cmake_minimum_required(VERSION 3.4...3.22)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
)

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  ggml_ext
  STABLE_ABI
  NB_STATIC
  # REFERENCE: https://nanobind.readthedocs.io/en/latest/faq.html#how-can-i-avoid-conflicts-with-other-projects-using-nanobind
  NB_DOMAIN ggml
  ggml/ggml_ext.cpp
)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_subdirectory(vendor/ggml)
target_include_directories(ggml_ext PUBLIC ${CMAKE_CURRENT_SCOURCE_DIR}/vendor/ggml/include/ggml)
target_link_libraries(ggml_ext PRIVATE ggml)
if(WIN32)
  set_property(TARGET ggml_ext PROPERTY SUFFIX ".${Python_SOABI}.pyd")
else()
  set_property(TARGET ggml_ext
               PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()
set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS ggml_ext 
    LIBRARY DESTINATION ggml
)