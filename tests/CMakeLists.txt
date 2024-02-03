# ==============================================================================
# Tests
# ==============================================================================

add_executable(scalable_ccd_tests
  test_cuda_stq.cpp
)

# ==============================================================================
# Required Libraries
# ==============================================================================
target_link_libraries(scalable_ccd_tests PUBLIC scalable_ccd::scalable_ccd)

include(catch2)
target_link_libraries(scalable_ccd_tests PUBLIC Catch2::Catch2WithMain)

include(libigl)
target_link_libraries(scalable_ccd_tests PUBLIC igl::core)

# Extra warnings (link last for highest priority)
include(scalable_ccd_warnings)
target_link_libraries(scalable_ccd_tests PRIVATE scalable_ccd::warnings)

# ==============================================================================
# Compiler options
# ==============================================================================
target_compile_definitions(scalable_ccd_tests PUBLIC CATCH_CONFIG_ENABLE_BENCHMARKING)

# ==============================================================================
# Register tests
# ==============================================================================
FetchContent_GetProperties(catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
catch_discover_tests(scalable_ccd_tests)