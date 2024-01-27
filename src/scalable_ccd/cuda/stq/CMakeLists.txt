set(SOURCES
    aabb.cu
    aabb.cuh
    broadphase.cu
    broadphase.cuh
    collision.cu
    collision.cuh
    io.cu
    io.cuh
    pca.cu
    pca.cuh
    queue.cu
    queue.cuh
    sweep.cu
    sweep.cuh
    timer.cuh
    types.cuh
    util.cu
    util.cuh
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################