set(SOURCES
  ipc_ccd_strategy.cu
  ipc_ccd_strategy.hpp
  memory_handler.cpp
  memory_handler.hpp
  scalar.cu
  scalar.cuh
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

# ==============================================================================
# Subfolders
# ==============================================================================
add_subdirectory(broad_phase)
add_subdirectory(narrow_phase)
add_subdirectory(utils)