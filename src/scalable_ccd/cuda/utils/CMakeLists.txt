set(SOURCES
  assert.cuh
  device_buffer.cuh
  device_matrix.cuh
  device_variable.cuh
  limits.cuh
  profiler.cpp
  profiler.hpp
  timer.cuh
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

# ==============================================================================
# Subfolders
# ==============================================================================