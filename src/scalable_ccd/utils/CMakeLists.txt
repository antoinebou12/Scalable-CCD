set(SOURCES
  logger.cpp
  logger.hpp
  merge_local_overlaps.cpp
  merge_local_overlaps.hpp
  pca.cpp
  pca.hpp
  timer.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

# ==============================================================================
# Subfolders
# ==============================================================================