cmake_minimum_required(VERSION 3.4...3.22)

project(ggml_python)

set(BUILD_SHARED_LIBS "On")
add_subdirectory(vendor/ggml)
install(
    TARGETS ggml 
    LIBRARY DESTINATION ggml
    RUNTIME DESTINATION ggml
)