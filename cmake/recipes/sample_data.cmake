# Sample Data (https://github.com/Continuous-Collision-Detection/Sample-Scalable-CCD-Data)
# License: Apache-2.0

if(TARGET scalable-ccd::data)
    return()
endif()

include(ExternalProject)

set(SCALABLE_CCD_DATA_DIR "${PROJECT_SOURCE_DIR}/tests/data/" CACHE PATH "Where should scalable-ccd download and look for test data?")
option(SCALABLE_CCD_USE_EXISTING_DATA_DIR "Use and existing data directory instead of downloading it" OFF)

if(SCALABLE_CCD_USE_EXISTING_DATA_DIR)
    ExternalProject_Add(
        scalable_ccd_data_download
        PREFIX ${FETCHCONTENT_BASE_DIR}/scalable-ccd-test-data
        SOURCE_DIR ${SCALABLE_CCD_DATA_DIR}
        # NOTE: No download step
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
    )
else()
    ExternalProject_Add(
        scalable_ccd_data_download
        PREFIX ${FETCHCONTENT_BASE_DIR}/scalable-ccd-test-data
        SOURCE_DIR ${SCALABLE_CCD_DATA_DIR}
        GIT_REPOSITORY https://github.com/Continuous-Collision-Detection/Sample-Scalable-CCD-Data
        GIT_TAG ac7b75447daeeac76c1a1f20ac4dacfdeaf09f6d
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
    )
endif()

# Create a dummy target for convenience
add_library(scalable_ccd_data INTERFACE)
add_library(scalable_ccd::data ALIAS scalable_ccd_data)

add_dependencies(scalable_ccd_data scalable_ccd_data_download)

target_compile_definitions(scalable_ccd_data INTERFACE  SCALABLE_CCD_DATA_DIR=\"${SCALABLE_CCD_DATA_DIR}\")