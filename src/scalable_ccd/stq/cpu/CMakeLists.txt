set(SOURCES
    aabb.cpp
    aabb.hpp
    io.cpp
    io.hpp
    sweep.cpp
    sweep.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################