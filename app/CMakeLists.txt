set(SOURCES
    main.cpp

    io.cpp
    io.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd_bin PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################