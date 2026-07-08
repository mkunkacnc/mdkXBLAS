cmake_minimum_required(VERSION 4.3.3)
project(mdkXBLAS)
#
# Suppose repository is in mdkXBLAS directory
# cd path/to/mdkXBLAS
#
# cmake -B build --preset mdkXBLAS-gcc
#  or
# cmake -B build --preset mdkXBLAS-gcc --fresh
#
# cmake --build build
#  or
# cmake --build build --clean-first
#
# ctest --test-dir build
#
# ------------------------------------------------------------
# Output directories
# ------------------------------------------------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" CACHE STRING "")

# ------------------------------------------------------------
# Fortran symbol mangling autodetection
# ------------------------------------------------------------
enable_language(Fortran)

file(WRITE ${CMAKE_BINARY_DIR}/fc_mangle_test.f90
"subroutine mangle_test(a,b)\n"
"  real :: a, b\n"
"  b = a\n"
"end subroutine mangle_test\n"
)

execute_process(
  COMMAND ${CMAKE_Fortran_COMPILER} -c fc_mangle_test.f90
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  OUTPUT_QUIET ERROR_QUIET
)

execute_process(
  COMMAND ${CMAKE_NM} fc_mangle_test.o
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  OUTPUT_VARIABLE NM_OUT
)

string(REGEX MATCH "mangle_test[^[:space:]]*" MANGLED ${NM_OUT})
message(STATUS "Detected Fortran symbol: ${MANGLED}")

if(MANGLED STREQUAL "mangle_test_")
  add_compile_definitions(CONFIG_FC_UNDERSCORE)
elseif(MANGLED STREQUAL "mangle_test__")
  add_compile_definitions(CONFIG_FC_DOUBLE_UNDERSCORE)
elseif(MANGLED STREQUAL "mangle_test")
  add_compile_definitions(CONFIG_FC_NO_UNDERSCORE)
elseif(MANGLED STREQUAL "MANGLE_TEST")
  add_compile_definitions(CONFIG_FC_UPPERCASE)
else()
  message(WARNING "Unknown Fortran mangling: ${MANGLED}")
endif()

# ------------------------------------------------------------

add_library(xblas)
target_sources(xblas
  PRIVATE
    src/axpby/BLAS_caxpby_s-f2c.c
    src/axpby/BLAS_caxpby_s.cpp
    src/axpby/BLAS_caxpby_s_x-f2c.c
    src/axpby/BLAS_caxpby_s_x.cpp
    src/axpby/BLAS_caxpby_x-f2c.c
    src/axpby/BLAS_caxpby_x.cpp
    src/axpby/BLAS_daxpby_s-f2c.c
    src/axpby/BLAS_daxpby_s.cpp
    src/axpby/BLAS_daxpby_s_x-f2c.c
    src/axpby/BLAS_daxpby_s_x.cpp
    src/axpby/BLAS_daxpby_x-f2c.c
    src/axpby/BLAS_daxpby_x.cpp
    src/axpby/BLAS_saxpby_x-f2c.c
    src/axpby/BLAS_saxpby_x.cpp
    src/axpby/BLAS_zaxpby_c-f2c.c
    src/axpby/BLAS_zaxpby_c.cpp
    src/axpby/BLAS_zaxpby_c_x-f2c.c
    src/axpby/BLAS_zaxpby_c_x.cpp
    src/axpby/BLAS_zaxpby_d-f2c.c
    src/axpby/BLAS_zaxpby_d.cpp
    src/axpby/BLAS_zaxpby_d_x-f2c.c
    src/axpby/BLAS_zaxpby_d_x.cpp
    src/axpby/BLAS_zaxpby_x-f2c.c
    src/axpby/BLAS_zaxpby_x.cpp
    src/common/BLAS_error.cpp
    src/common/BLAS_fpinfo_x-f2c.c
    src/common/BLAS_fpinfo_x.cpp
    src/common/blas_malloc.cpp
    src/dot/BLAS_cdot_c_s-f2c.c
    src/dot/BLAS_cdot_c_s.cpp
    src/dot/BLAS_cdot_c_s_x-f2c.c
    src/dot/BLAS_cdot_c_s_x.cpp
    src/dot/BLAS_cdot_s_c-f2c.c
    src/dot/BLAS_cdot_s_c.cpp
    src/dot/BLAS_cdot_s_c_x-f2c.c
    src/dot/BLAS_cdot_s_c_x.cpp
    src/dot/BLAS_cdot_s_s-f2c.c
    src/dot/BLAS_cdot_s_s.cpp
    src/dot/BLAS_cdot_s_s_x-f2c.c
    src/dot/BLAS_cdot_s_s_x.cpp
    src/dot/BLAS_cdot_x-f2c.c
    src/dot/BLAS_cdot_x.cpp
    src/dot/BLAS_ddot_d_s-f2c.c
    src/dot/BLAS_ddot_d_s.cpp
    src/dot/BLAS_ddot_d_s_x-f2c.c
    src/dot/BLAS_ddot_d_s_x.cpp
    src/dot/BLAS_ddot_s_d-f2c.c
    src/dot/BLAS_ddot_s_d.cpp
    src/dot/BLAS_ddot_s_d_x-f2c.c
    src/dot/BLAS_ddot_s_d_x.cpp
    src/dot/BLAS_ddot_s_s-f2c.c
    src/dot/BLAS_ddot_s_s.cpp
    src/dot/BLAS_ddot_s_s_x-f2c.c
    src/dot/BLAS_ddot_s_s_x.cpp
    src/dot/BLAS_ddot_x-f2c.c
    src/dot/BLAS_ddot_x.cpp
    src/dot/BLAS_sdot_x-f2c.c
    src/dot/BLAS_sdot_x.cpp
    src/dot/BLAS_zdot_c_c-f2c.c
    src/dot/BLAS_zdot_c_c.cpp
    src/dot/BLAS_zdot_c_c_x-f2c.c
    src/dot/BLAS_zdot_c_c_x.cpp
    src/dot/BLAS_zdot_c_z-f2c.c
    src/dot/BLAS_zdot_c_z.cpp
    src/dot/BLAS_zdot_c_z_x-f2c.c
    src/dot/BLAS_zdot_c_z_x.cpp
    src/dot/BLAS_zdot_d_d-f2c.c
    src/dot/BLAS_zdot_d_d.cpp
    src/dot/BLAS_zdot_d_d_x-f2c.c
    src/dot/BLAS_zdot_d_d_x.cpp
    src/dot/BLAS_zdot_d_z-f2c.c
    src/dot/BLAS_zdot_d_z.cpp
    src/dot/BLAS_zdot_d_z_x-f2c.c
    src/dot/BLAS_zdot_d_z_x.cpp
    src/dot/BLAS_zdot_x-f2c.c
    src/dot/BLAS_zdot_x.cpp
    src/dot/BLAS_zdot_z_c-f2c.c
    src/dot/BLAS_zdot_z_c.cpp
    src/dot/BLAS_zdot_z_c_x-f2c.c
    src/dot/BLAS_zdot_z_c_x.cpp
    src/dot/BLAS_zdot_z_d-f2c.c
    src/dot/BLAS_zdot_z_d.cpp
    src/dot/BLAS_zdot_z_d_x-f2c.c
    src/dot/BLAS_zdot_z_d_x.cpp
    src/gbmv/BLAS_cgbmv_c_s-f2c.c
    src/gbmv/BLAS_cgbmv_c_s.cpp
    src/gbmv/BLAS_cgbmv_c_s_x-f2c.c
    src/gbmv/BLAS_cgbmv_c_s_x.cpp
    src/gbmv/BLAS_cgbmv_s_c-f2c.c
    src/gbmv/BLAS_cgbmv_s_c.cpp
    src/gbmv/BLAS_cgbmv_s_c_x-f2c.c
    src/gbmv/BLAS_cgbmv_s_c_x.cpp
    src/gbmv/BLAS_cgbmv_s_s-f2c.c
    src/gbmv/BLAS_cgbmv_s_s.cpp
    src/gbmv/BLAS_cgbmv_s_s_x-f2c.c
    src/gbmv/BLAS_cgbmv_s_s_x.cpp
    src/gbmv/BLAS_cgbmv_x-f2c.c
    src/gbmv/BLAS_cgbmv_x.cpp
    src/gbmv/BLAS_dgbmv_d_s-f2c.c
    src/gbmv/BLAS_dgbmv_d_s.cpp
    src/gbmv/BLAS_dgbmv_d_s_x-f2c.c
    src/gbmv/BLAS_dgbmv_d_s_x.cpp
    src/gbmv/BLAS_dgbmv_s_d-f2c.c
    src/gbmv/BLAS_dgbmv_s_d.cpp
    src/gbmv/BLAS_dgbmv_s_d_x-f2c.c
    src/gbmv/BLAS_dgbmv_s_d_x.cpp
    src/gbmv/BLAS_dgbmv_s_s-f2c.c
    src/gbmv/BLAS_dgbmv_s_s.cpp
    src/gbmv/BLAS_dgbmv_s_s_x-f2c.c
    src/gbmv/BLAS_dgbmv_s_s_x.cpp
    src/gbmv/BLAS_dgbmv_x-f2c.c
    src/gbmv/BLAS_dgbmv_x.cpp
    src/gbmv/BLAS_sgbmv_x-f2c.c
    src/gbmv/BLAS_sgbmv_x.cpp
    src/gbmv/BLAS_zgbmv_c_c-f2c.c
    src/gbmv/BLAS_zgbmv_c_c.cpp
    src/gbmv/BLAS_zgbmv_c_c_x-f2c.c
    src/gbmv/BLAS_zgbmv_c_c_x.cpp
    src/gbmv/BLAS_zgbmv_c_z-f2c.c
    src/gbmv/BLAS_zgbmv_c_z.cpp
    src/gbmv/BLAS_zgbmv_c_z_x-f2c.c
    src/gbmv/BLAS_zgbmv_c_z_x.cpp
    src/gbmv/BLAS_zgbmv_d_d-f2c.c
    src/gbmv/BLAS_zgbmv_d_d.cpp
    src/gbmv/BLAS_zgbmv_d_d_x-f2c.c
    src/gbmv/BLAS_zgbmv_d_d_x.cpp
    src/gbmv/BLAS_zgbmv_d_z-f2c.c
    src/gbmv/BLAS_zgbmv_d_z.cpp
    src/gbmv/BLAS_zgbmv_d_z_x-f2c.c
    src/gbmv/BLAS_zgbmv_d_z_x.cpp
    src/gbmv/BLAS_zgbmv_x-f2c.c
    src/gbmv/BLAS_zgbmv_x.cpp
    src/gbmv/BLAS_zgbmv_z_c-f2c.c
    src/gbmv/BLAS_zgbmv_z_c.cpp
    src/gbmv/BLAS_zgbmv_z_c_x-f2c.c
    src/gbmv/BLAS_zgbmv_z_c_x.cpp
    src/gbmv/BLAS_zgbmv_z_d-f2c.c
    src/gbmv/BLAS_zgbmv_z_d.cpp
    src/gbmv/BLAS_zgbmv_z_d_x-f2c.c
    src/gbmv/BLAS_zgbmv_z_d_x.cpp
    src/gbmv2/BLAS_cgbmv2_c_s-f2c.c
    src/gbmv2/BLAS_cgbmv2_c_s.cpp
    src/gbmv2/BLAS_cgbmv2_c_s_x-f2c.c
    src/gbmv2/BLAS_cgbmv2_c_s_x.cpp
    src/gbmv2/BLAS_cgbmv2_s_c-f2c.c
    src/gbmv2/BLAS_cgbmv2_s_c.cpp
    src/gbmv2/BLAS_cgbmv2_s_c_x-f2c.c
    src/gbmv2/BLAS_cgbmv2_s_c_x.cpp
    src/gbmv2/BLAS_cgbmv2_s_s-f2c.c
    src/gbmv2/BLAS_cgbmv2_s_s.cpp
    src/gbmv2/BLAS_cgbmv2_s_s_x-f2c.c
    src/gbmv2/BLAS_cgbmv2_s_s_x.cpp
    src/gbmv2/BLAS_cgbmv2_x-f2c.c
    src/gbmv2/BLAS_cgbmv2_x.cpp
    src/gbmv2/BLAS_dgbmv2_d_s-f2c.c
    src/gbmv2/BLAS_dgbmv2_d_s.cpp
    src/gbmv2/BLAS_dgbmv2_d_s_x-f2c.c
    src/gbmv2/BLAS_dgbmv2_d_s_x.cpp
    src/gbmv2/BLAS_dgbmv2_s_d-f2c.c
    src/gbmv2/BLAS_dgbmv2_s_d.cpp
    src/gbmv2/BLAS_dgbmv2_s_d_x-f2c.c
    src/gbmv2/BLAS_dgbmv2_s_d_x.cpp
    src/gbmv2/BLAS_dgbmv2_s_s-f2c.c
    src/gbmv2/BLAS_dgbmv2_s_s.cpp
    src/gbmv2/BLAS_dgbmv2_s_s_x-f2c.c
    src/gbmv2/BLAS_dgbmv2_s_s_x.cpp
    src/gbmv2/BLAS_dgbmv2_x-f2c.c
    src/gbmv2/BLAS_dgbmv2_x.cpp
    src/gbmv2/BLAS_sgbmv2_x-f2c.c
    src/gbmv2/BLAS_sgbmv2_x.cpp
    src/gbmv2/BLAS_zgbmv2_c_c-f2c.c
    src/gbmv2/BLAS_zgbmv2_c_c.cpp
    src/gbmv2/BLAS_zgbmv2_c_c_x-f2c.c
    src/gbmv2/BLAS_zgbmv2_c_c_x.cpp
    src/gbmv2/BLAS_zgbmv2_c_z-f2c.c
    src/gbmv2/BLAS_zgbmv2_c_z.cpp
    src/gbmv2/BLAS_zgbmv2_c_z_x-f2c.c
    src/gbmv2/BLAS_zgbmv2_c_z_x.cpp
    src/gbmv2/BLAS_zgbmv2_d_d-f2c.c
    src/gbmv2/BLAS_zgbmv2_d_d.cpp
    src/gbmv2/BLAS_zgbmv2_d_d_x-f2c.c
    src/gbmv2/BLAS_zgbmv2_d_d_x.cpp
    src/gbmv2/BLAS_zgbmv2_d_z-f2c.c
    src/gbmv2/BLAS_zgbmv2_d_z.cpp
    src/gbmv2/BLAS_zgbmv2_d_z_x-f2c.c
    src/gbmv2/BLAS_zgbmv2_d_z_x.cpp
    src/gbmv2/BLAS_zgbmv2_x-f2c.c
    src/gbmv2/BLAS_zgbmv2_x.cpp
    src/gbmv2/BLAS_zgbmv2_z_c-f2c.c
    src/gbmv2/BLAS_zgbmv2_z_c.cpp
    src/gbmv2/BLAS_zgbmv2_z_c_x-f2c.c
    src/gbmv2/BLAS_zgbmv2_z_c_x.cpp
    src/gbmv2/BLAS_zgbmv2_z_d-f2c.c
    src/gbmv2/BLAS_zgbmv2_z_d.cpp
    src/gbmv2/BLAS_zgbmv2_z_d_x-f2c.c
    src/gbmv2/BLAS_zgbmv2_z_d_x.cpp
    src/ge_sum_mv/BLAS_cge_sum_mv_c_s-f2c.c
    src/ge_sum_mv/BLAS_cge_sum_mv_c_s.cpp
    src/ge_sum_mv/BLAS_cge_sum_mv_c_s_x-f2c.c
    src/ge_sum_mv/BLAS_cge_sum_mv_c_s_x.cpp
    src/ge_sum_mv/BLAS_cge_sum_mv_s_c-f2c.c
    src/ge_sum_mv/BLAS_cge_sum_mv_s_c.cpp
    src/ge_sum_mv/BLAS_cge_sum_mv_s_c_x-f2c.c
    src/ge_sum_mv/BLAS_cge_sum_mv_s_c_x.cpp
    src/ge_sum_mv/BLAS_cge_sum_mv_s_s-f2c.c
    src/ge_sum_mv/BLAS_cge_sum_mv_s_s.cpp
    src/ge_sum_mv/BLAS_cge_sum_mv_s_s_x-f2c.c
    src/ge_sum_mv/BLAS_cge_sum_mv_s_s_x.cpp
    src/ge_sum_mv/BLAS_cge_sum_mv_x-f2c.c
    src/ge_sum_mv/BLAS_cge_sum_mv_x.cpp
    src/ge_sum_mv/BLAS_dge_sum_mv_d_s-f2c.c
    src/ge_sum_mv/BLAS_dge_sum_mv_d_s.cpp
    src/ge_sum_mv/BLAS_dge_sum_mv_d_s_x-f2c.c
    src/ge_sum_mv/BLAS_dge_sum_mv_d_s_x.cpp
    src/ge_sum_mv/BLAS_dge_sum_mv_s_d-f2c.c
    src/ge_sum_mv/BLAS_dge_sum_mv_s_d.cpp
    src/ge_sum_mv/BLAS_dge_sum_mv_s_d_x-f2c.c
    src/ge_sum_mv/BLAS_dge_sum_mv_s_d_x.cpp
    src/ge_sum_mv/BLAS_dge_sum_mv_s_s-f2c.c
    src/ge_sum_mv/BLAS_dge_sum_mv_s_s.cpp
    src/ge_sum_mv/BLAS_dge_sum_mv_s_s_x-f2c.c
    src/ge_sum_mv/BLAS_dge_sum_mv_s_s_x.cpp
    src/ge_sum_mv/BLAS_dge_sum_mv_x-f2c.c
    src/ge_sum_mv/BLAS_dge_sum_mv_x.cpp
    src/ge_sum_mv/BLAS_sge_sum_mv_x-f2c.c
    src/ge_sum_mv/BLAS_sge_sum_mv_x.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_c_c-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_c_c.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_c_c_x-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_c_c_x.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_c_z-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_c_z.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_c_z_x-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_c_z_x.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_d_d-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_d_d.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_d_d_x-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_d_d_x.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_d_z-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_d_z.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_d_z_x-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_d_z_x.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_x-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_x.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_z_c-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_z_c.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_z_c_x-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_z_c_x.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_z_d-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_z_d.cpp
    src/ge_sum_mv/BLAS_zge_sum_mv_z_d_x-f2c.c
    src/ge_sum_mv/BLAS_zge_sum_mv_z_d_x.cpp
    src/gemm/BLAS_cgemm_c_s-f2c.c
    src/gemm/BLAS_cgemm_c_s.cpp
    src/gemm/BLAS_cgemm_c_s_x-f2c.c
    src/gemm/BLAS_cgemm_c_s_x.cpp
    src/gemm/BLAS_cgemm_s_c-f2c.c
    src/gemm/BLAS_cgemm_s_c.cpp
    src/gemm/BLAS_cgemm_s_c_x-f2c.c
    src/gemm/BLAS_cgemm_s_c_x.cpp
    src/gemm/BLAS_cgemm_s_s-f2c.c
    src/gemm/BLAS_cgemm_s_s.cpp
    src/gemm/BLAS_cgemm_s_s_x-f2c.c
    src/gemm/BLAS_cgemm_s_s_x.cpp
    src/gemm/BLAS_cgemm_x-f2c.c
    src/gemm/BLAS_cgemm_x.cpp
    src/gemm/BLAS_dgemm_d_s-f2c.c
    src/gemm/BLAS_dgemm_d_s.cpp
    src/gemm/BLAS_dgemm_d_s_x-f2c.c
    src/gemm/BLAS_dgemm_d_s_x.cpp
    src/gemm/BLAS_dgemm_s_d-f2c.c
    src/gemm/BLAS_dgemm_s_d.cpp
    src/gemm/BLAS_dgemm_s_d_x-f2c.c
    src/gemm/BLAS_dgemm_s_d_x.cpp
    src/gemm/BLAS_dgemm_s_s-f2c.c
    src/gemm/BLAS_dgemm_s_s.cpp
    src/gemm/BLAS_dgemm_s_s_x-f2c.c
    src/gemm/BLAS_dgemm_s_s_x.cpp
    src/gemm/BLAS_dgemm_x-f2c.c
    src/gemm/BLAS_dgemm_x.cpp
    src/gemm/BLAS_sgemm_x-f2c.c
    src/gemm/BLAS_sgemm_x.cpp
    src/gemm/BLAS_zgemm_c_c-f2c.c
    src/gemm/BLAS_zgemm_c_c.cpp
    src/gemm/BLAS_zgemm_c_c_x-f2c.c
    src/gemm/BLAS_zgemm_c_c_x.cpp
    src/gemm/BLAS_zgemm_c_z-f2c.c
    src/gemm/BLAS_zgemm_c_z.cpp
    src/gemm/BLAS_zgemm_c_z_x-f2c.c
    src/gemm/BLAS_zgemm_c_z_x.cpp
    src/gemm/BLAS_zgemm_d_d-f2c.c
    src/gemm/BLAS_zgemm_d_d.cpp
    src/gemm/BLAS_zgemm_d_d_x-f2c.c
    src/gemm/BLAS_zgemm_d_d_x.cpp
    src/gemm/BLAS_zgemm_d_z-f2c.c
    src/gemm/BLAS_zgemm_d_z.cpp
    src/gemm/BLAS_zgemm_d_z_x-f2c.c
    src/gemm/BLAS_zgemm_d_z_x.cpp
    src/gemm/BLAS_zgemm_x-f2c.c
    src/gemm/BLAS_zgemm_x.cpp
    src/gemm/BLAS_zgemm_z_c-f2c.c
    src/gemm/BLAS_zgemm_z_c.cpp
    src/gemm/BLAS_zgemm_z_c_x-f2c.c
    src/gemm/BLAS_zgemm_z_c_x.cpp
    src/gemm/BLAS_zgemm_z_d-f2c.c
    src/gemm/BLAS_zgemm_z_d.cpp
    src/gemm/BLAS_zgemm_z_d_x-f2c.c
    src/gemm/BLAS_zgemm_z_d_x.cpp
    src/gemv/BLAS_cgemv_c_s-f2c.c
    src/gemv/BLAS_cgemv_c_s.cpp
    src/gemv/BLAS_cgemv_c_s_x-f2c.c
    src/gemv/BLAS_cgemv_c_s_x.cpp
    src/gemv/BLAS_cgemv_s_c-f2c.c
    src/gemv/BLAS_cgemv_s_c.cpp
    src/gemv/BLAS_cgemv_s_c_x-f2c.c
    src/gemv/BLAS_cgemv_s_c_x.cpp
    src/gemv/BLAS_cgemv_s_s-f2c.c
    src/gemv/BLAS_cgemv_s_s.cpp
    src/gemv/BLAS_cgemv_s_s_x-f2c.c
    src/gemv/BLAS_cgemv_s_s_x.cpp
    src/gemv/BLAS_cgemv_x-f2c.c
    src/gemv/BLAS_cgemv_x.cpp
    src/gemv/BLAS_dgemv_d_s-f2c.c
    src/gemv/BLAS_dgemv_d_s.cpp
    src/gemv/BLAS_dgemv_d_s_x-f2c.c
    src/gemv/BLAS_dgemv_d_s_x.cpp
    src/gemv/BLAS_dgemv_s_d-f2c.c
    src/gemv/BLAS_dgemv_s_d.cpp
    src/gemv/BLAS_dgemv_s_d_x-f2c.c
    src/gemv/BLAS_dgemv_s_d_x.cpp
    src/gemv/BLAS_dgemv_s_s-f2c.c
    src/gemv/BLAS_dgemv_s_s.cpp
    src/gemv/BLAS_dgemv_s_s_x-f2c.c
    src/gemv/BLAS_dgemv_s_s_x.cpp
    src/gemv/BLAS_dgemv_x-f2c.c
    src/gemv/BLAS_dgemv_x.cpp
    src/gemv/BLAS_sgemv_x-f2c.c
    src/gemv/BLAS_sgemv_x.cpp
    src/gemv/BLAS_zgemv_c_c-f2c.c
    src/gemv/BLAS_zgemv_c_c.cpp
    src/gemv/BLAS_zgemv_c_c_x-f2c.c
    src/gemv/BLAS_zgemv_c_c_x.cpp
    src/gemv/BLAS_zgemv_c_z-f2c.c
    src/gemv/BLAS_zgemv_c_z.cpp
    src/gemv/BLAS_zgemv_c_z_x-f2c.c
    src/gemv/BLAS_zgemv_c_z_x.cpp
    src/gemv/BLAS_zgemv_d_d-f2c.c
    src/gemv/BLAS_zgemv_d_d.cpp
    src/gemv/BLAS_zgemv_d_d_x-f2c.c
    src/gemv/BLAS_zgemv_d_d_x.cpp
    src/gemv/BLAS_zgemv_d_z-f2c.c
    src/gemv/BLAS_zgemv_d_z.cpp
    src/gemv/BLAS_zgemv_d_z_x-f2c.c
    src/gemv/BLAS_zgemv_d_z_x.cpp
    src/gemv/BLAS_zgemv_x-f2c.c
    src/gemv/BLAS_zgemv_x.cpp
    src/gemv/BLAS_zgemv_z_c-f2c.c
    src/gemv/BLAS_zgemv_z_c.cpp
    src/gemv/BLAS_zgemv_z_c_x-f2c.c
    src/gemv/BLAS_zgemv_z_c_x.cpp
    src/gemv/BLAS_zgemv_z_d-f2c.c
    src/gemv/BLAS_zgemv_z_d.cpp
    src/gemv/BLAS_zgemv_z_d_x-f2c.c
    src/gemv/BLAS_zgemv_z_d_x.cpp
    src/gemv2/BLAS_cgemv2_c_s-f2c.c
    src/gemv2/BLAS_cgemv2_c_s.cpp
    src/gemv2/BLAS_cgemv2_c_s_x-f2c.c
    src/gemv2/BLAS_cgemv2_c_s_x.cpp
    src/gemv2/BLAS_cgemv2_s_c-f2c.c
    src/gemv2/BLAS_cgemv2_s_c.cpp
    src/gemv2/BLAS_cgemv2_s_c_x-f2c.c
    src/gemv2/BLAS_cgemv2_s_c_x.cpp
    src/gemv2/BLAS_cgemv2_s_s-f2c.c
    src/gemv2/BLAS_cgemv2_s_s.cpp
    src/gemv2/BLAS_cgemv2_s_s_x-f2c.c
    src/gemv2/BLAS_cgemv2_s_s_x.cpp
    src/gemv2/BLAS_cgemv2_x-f2c.c
    src/gemv2/BLAS_cgemv2_x.cpp
    src/gemv2/BLAS_dgemv2_d_s-f2c.c
    src/gemv2/BLAS_dgemv2_d_s.cpp
    src/gemv2/BLAS_dgemv2_d_s_x-f2c.c
    src/gemv2/BLAS_dgemv2_d_s_x.cpp
    src/gemv2/BLAS_dgemv2_s_d-f2c.c
    src/gemv2/BLAS_dgemv2_s_d.cpp
    src/gemv2/BLAS_dgemv2_s_d_x-f2c.c
    src/gemv2/BLAS_dgemv2_s_d_x.cpp
    src/gemv2/BLAS_dgemv2_s_s-f2c.c
    src/gemv2/BLAS_dgemv2_s_s.cpp
    src/gemv2/BLAS_dgemv2_s_s_x-f2c.c
    src/gemv2/BLAS_dgemv2_s_s_x.cpp
    src/gemv2/BLAS_dgemv2_x-f2c.c
    src/gemv2/BLAS_dgemv2_x.cpp
    src/gemv2/BLAS_sgemv2_x-f2c.c
    src/gemv2/BLAS_sgemv2_x.cpp
    src/gemv2/BLAS_zgemv2_c_c-f2c.c
    src/gemv2/BLAS_zgemv2_c_c.cpp
    src/gemv2/BLAS_zgemv2_c_c_x-f2c.c
    src/gemv2/BLAS_zgemv2_c_c_x.cpp
    src/gemv2/BLAS_zgemv2_c_z-f2c.c
    src/gemv2/BLAS_zgemv2_c_z.cpp
    src/gemv2/BLAS_zgemv2_c_z_x-f2c.c
    src/gemv2/BLAS_zgemv2_c_z_x.cpp
    src/gemv2/BLAS_zgemv2_d_d-f2c.c
    src/gemv2/BLAS_zgemv2_d_d.cpp
    src/gemv2/BLAS_zgemv2_d_d_x-f2c.c
    src/gemv2/BLAS_zgemv2_d_d_x.cpp
    src/gemv2/BLAS_zgemv2_d_z-f2c.c
    src/gemv2/BLAS_zgemv2_d_z.cpp
    src/gemv2/BLAS_zgemv2_d_z_x-f2c.c
    src/gemv2/BLAS_zgemv2_d_z_x.cpp
    src/gemv2/BLAS_zgemv2_x-f2c.c
    src/gemv2/BLAS_zgemv2_x.cpp
    src/gemv2/BLAS_zgemv2_z_c-f2c.c
    src/gemv2/BLAS_zgemv2_z_c.cpp
    src/gemv2/BLAS_zgemv2_z_c_x-f2c.c
    src/gemv2/BLAS_zgemv2_z_c_x.cpp
    src/gemv2/BLAS_zgemv2_z_d-f2c.c
    src/gemv2/BLAS_zgemv2_z_d.cpp
    src/gemv2/BLAS_zgemv2_z_d_x-f2c.c
    src/gemv2/BLAS_zgemv2_z_d_x.cpp
    src/hbmv/BLAS_chbmv_c_s-f2c.c
    src/hbmv/BLAS_chbmv_c_s.cpp
    src/hbmv/BLAS_chbmv_c_s_x-f2c.c
    src/hbmv/BLAS_chbmv_c_s_x.cpp
    src/hbmv/BLAS_chbmv_x-f2c.c
    src/hbmv/BLAS_chbmv_x.cpp
    src/hbmv/BLAS_zhbmv_c_c-f2c.c
    src/hbmv/BLAS_zhbmv_c_c.cpp
    src/hbmv/BLAS_zhbmv_c_c_x-f2c.c
    src/hbmv/BLAS_zhbmv_c_c_x.cpp
    src/hbmv/BLAS_zhbmv_c_z-f2c.c
    src/hbmv/BLAS_zhbmv_c_z.cpp
    src/hbmv/BLAS_zhbmv_c_z_x-f2c.c
    src/hbmv/BLAS_zhbmv_c_z_x.cpp
    src/hbmv/BLAS_zhbmv_x-f2c.c
    src/hbmv/BLAS_zhbmv_x.cpp
    src/hbmv/BLAS_zhbmv_z_c-f2c.c
    src/hbmv/BLAS_zhbmv_z_c.cpp
    src/hbmv/BLAS_zhbmv_z_c_x-f2c.c
    src/hbmv/BLAS_zhbmv_z_c_x.cpp
    src/hbmv/BLAS_zhbmv_z_d-f2c.c
    src/hbmv/BLAS_zhbmv_z_d.cpp
    src/hbmv/BLAS_zhbmv_z_d_x-f2c.c
    src/hbmv/BLAS_zhbmv_z_d_x.cpp
    src/hemm/BLAS_chemm_c_s-f2c.c
    src/hemm/BLAS_chemm_c_s.cpp
    src/hemm/BLAS_chemm_c_s_x-f2c.c
    src/hemm/BLAS_chemm_c_s_x.cpp
    src/hemm/BLAS_chemm_x-f2c.c
    src/hemm/BLAS_chemm_x.cpp
    src/hemm/BLAS_zhemm_c_c-f2c.c
    src/hemm/BLAS_zhemm_c_c.cpp
    src/hemm/BLAS_zhemm_c_c_x-f2c.c
    src/hemm/BLAS_zhemm_c_c_x.cpp
    src/hemm/BLAS_zhemm_c_z-f2c.c
    src/hemm/BLAS_zhemm_c_z.cpp
    src/hemm/BLAS_zhemm_c_z_x-f2c.c
    src/hemm/BLAS_zhemm_c_z_x.cpp
    src/hemm/BLAS_zhemm_x-f2c.c
    src/hemm/BLAS_zhemm_x.cpp
    src/hemm/BLAS_zhemm_z_c-f2c.c
    src/hemm/BLAS_zhemm_z_c.cpp
    src/hemm/BLAS_zhemm_z_c_x-f2c.c
    src/hemm/BLAS_zhemm_z_c_x.cpp
    src/hemm/BLAS_zhemm_z_d-f2c.c
    src/hemm/BLAS_zhemm_z_d.cpp
    src/hemm/BLAS_zhemm_z_d_x-f2c.c
    src/hemm/BLAS_zhemm_z_d_x.cpp
    src/hemv/BLAS_chemv_c_s-f2c.c
    src/hemv/BLAS_chemv_c_s.cpp
    src/hemv/BLAS_chemv_c_s_x-f2c.c
    src/hemv/BLAS_chemv_c_s_x.cpp
    src/hemv/BLAS_chemv_x-f2c.c
    src/hemv/BLAS_chemv_x.cpp
    src/hemv/BLAS_zhemv_c_c-f2c.c
    src/hemv/BLAS_zhemv_c_c.cpp
    src/hemv/BLAS_zhemv_c_c_x-f2c.c
    src/hemv/BLAS_zhemv_c_c_x.cpp
    src/hemv/BLAS_zhemv_c_z-f2c.c
    src/hemv/BLAS_zhemv_c_z.cpp
    src/hemv/BLAS_zhemv_c_z_x-f2c.c
    src/hemv/BLAS_zhemv_c_z_x.cpp
    src/hemv/BLAS_zhemv_x-f2c.c
    src/hemv/BLAS_zhemv_x.cpp
    src/hemv/BLAS_zhemv_z_c-f2c.c
    src/hemv/BLAS_zhemv_z_c.cpp
    src/hemv/BLAS_zhemv_z_c_x-f2c.c
    src/hemv/BLAS_zhemv_z_c_x.cpp
    src/hemv/BLAS_zhemv_z_d-f2c.c
    src/hemv/BLAS_zhemv_z_d.cpp
    src/hemv/BLAS_zhemv_z_d_x-f2c.c
    src/hemv/BLAS_zhemv_z_d_x.cpp
    src/hemv2/BLAS_chemv2_c_s-f2c.c
    src/hemv2/BLAS_chemv2_c_s.cpp
    src/hemv2/BLAS_chemv2_c_s_x-f2c.c
    src/hemv2/BLAS_chemv2_c_s_x.cpp
    src/hemv2/BLAS_chemv2_x-f2c.c
    src/hemv2/BLAS_chemv2_x.cpp
    src/hemv2/BLAS_zhemv2_c_c-f2c.c
    src/hemv2/BLAS_zhemv2_c_c.cpp
    src/hemv2/BLAS_zhemv2_c_c_x-f2c.c
    src/hemv2/BLAS_zhemv2_c_c_x.cpp
    src/hemv2/BLAS_zhemv2_c_z-f2c.c
    src/hemv2/BLAS_zhemv2_c_z.cpp
    src/hemv2/BLAS_zhemv2_c_z_x-f2c.c
    src/hemv2/BLAS_zhemv2_c_z_x.cpp
    src/hemv2/BLAS_zhemv2_x-f2c.c
    src/hemv2/BLAS_zhemv2_x.cpp
    src/hemv2/BLAS_zhemv2_z_c-f2c.c
    src/hemv2/BLAS_zhemv2_z_c.cpp
    src/hemv2/BLAS_zhemv2_z_c_x-f2c.c
    src/hemv2/BLAS_zhemv2_z_c_x.cpp
    src/hemv2/BLAS_zhemv2_z_d-f2c.c
    src/hemv2/BLAS_zhemv2_z_d.cpp
    src/hemv2/BLAS_zhemv2_z_d_x-f2c.c
    src/hemv2/BLAS_zhemv2_z_d_x.cpp
    src/hpmv/BLAS_chpmv_c_s-f2c.c
    src/hpmv/BLAS_chpmv_c_s.cpp
    src/hpmv/BLAS_chpmv_c_s_x-f2c.c
    src/hpmv/BLAS_chpmv_c_s_x.cpp
    src/hpmv/BLAS_chpmv_x-f2c.c
    src/hpmv/BLAS_chpmv_x.cpp
    src/hpmv/BLAS_zhpmv_c_c-f2c.c
    src/hpmv/BLAS_zhpmv_c_c.cpp
    src/hpmv/BLAS_zhpmv_c_c_x-f2c.c
    src/hpmv/BLAS_zhpmv_c_c_x.cpp
    src/hpmv/BLAS_zhpmv_c_z-f2c.c
    src/hpmv/BLAS_zhpmv_c_z.cpp
    src/hpmv/BLAS_zhpmv_c_z_x-f2c.c
    src/hpmv/BLAS_zhpmv_c_z_x.cpp
    src/hpmv/BLAS_zhpmv_x-f2c.c
    src/hpmv/BLAS_zhpmv_x.cpp
    src/hpmv/BLAS_zhpmv_z_c-f2c.c
    src/hpmv/BLAS_zhpmv_z_c.cpp
    src/hpmv/BLAS_zhpmv_z_c_x-f2c.c
    src/hpmv/BLAS_zhpmv_z_c_x.cpp
    src/hpmv/BLAS_zhpmv_z_d-f2c.c
    src/hpmv/BLAS_zhpmv_z_d.cpp
    src/hpmv/BLAS_zhpmv_z_d_x-f2c.c
    src/hpmv/BLAS_zhpmv_z_d_x.cpp
    src/sbmv/BLAS_csbmv_c_s-f2c.c
    src/sbmv/BLAS_csbmv_c_s.cpp
    src/sbmv/BLAS_csbmv_c_s_x-f2c.c
    src/sbmv/BLAS_csbmv_c_s_x.cpp
    src/sbmv/BLAS_csbmv_s_c-f2c.c
    src/sbmv/BLAS_csbmv_s_c.cpp
    src/sbmv/BLAS_csbmv_s_c_x-f2c.c
    src/sbmv/BLAS_csbmv_s_c_x.cpp
    src/sbmv/BLAS_csbmv_s_s-f2c.c
    src/sbmv/BLAS_csbmv_s_s.cpp
    src/sbmv/BLAS_csbmv_s_s_x-f2c.c
    src/sbmv/BLAS_csbmv_s_s_x.cpp
    src/sbmv/BLAS_csbmv_x-f2c.c
    src/sbmv/BLAS_csbmv_x.cpp
    src/sbmv/BLAS_dsbmv_d_s-f2c.c
    src/sbmv/BLAS_dsbmv_d_s.cpp
    src/sbmv/BLAS_dsbmv_d_s_x-f2c.c
    src/sbmv/BLAS_dsbmv_d_s_x.cpp
    src/sbmv/BLAS_dsbmv_s_d-f2c.c
    src/sbmv/BLAS_dsbmv_s_d.cpp
    src/sbmv/BLAS_dsbmv_s_d_x-f2c.c
    src/sbmv/BLAS_dsbmv_s_d_x.cpp
    src/sbmv/BLAS_dsbmv_s_s-f2c.c
    src/sbmv/BLAS_dsbmv_s_s.cpp
    src/sbmv/BLAS_dsbmv_s_s_x-f2c.c
    src/sbmv/BLAS_dsbmv_s_s_x.cpp
    src/sbmv/BLAS_dsbmv_x-f2c.c
    src/sbmv/BLAS_dsbmv_x.cpp
    src/sbmv/BLAS_ssbmv_x-f2c.c
    src/sbmv/BLAS_ssbmv_x.cpp
    src/sbmv/BLAS_zsbmv_c_c-f2c.c
    src/sbmv/BLAS_zsbmv_c_c.cpp
    src/sbmv/BLAS_zsbmv_c_c_x-f2c.c
    src/sbmv/BLAS_zsbmv_c_c_x.cpp
    src/sbmv/BLAS_zsbmv_c_z-f2c.c
    src/sbmv/BLAS_zsbmv_c_z.cpp
    src/sbmv/BLAS_zsbmv_c_z_x-f2c.c
    src/sbmv/BLAS_zsbmv_c_z_x.cpp
    src/sbmv/BLAS_zsbmv_d_d-f2c.c
    src/sbmv/BLAS_zsbmv_d_d.cpp
    src/sbmv/BLAS_zsbmv_d_d_x-f2c.c
    src/sbmv/BLAS_zsbmv_d_d_x.cpp
    src/sbmv/BLAS_zsbmv_d_z-f2c.c
    src/sbmv/BLAS_zsbmv_d_z.cpp
    src/sbmv/BLAS_zsbmv_d_z_x-f2c.c
    src/sbmv/BLAS_zsbmv_d_z_x.cpp
    src/sbmv/BLAS_zsbmv_x-f2c.c
    src/sbmv/BLAS_zsbmv_x.cpp
    src/sbmv/BLAS_zsbmv_z_c-f2c.c
    src/sbmv/BLAS_zsbmv_z_c.cpp
    src/sbmv/BLAS_zsbmv_z_c_x-f2c.c
    src/sbmv/BLAS_zsbmv_z_c_x.cpp
    src/sbmv/BLAS_zsbmv_z_d-f2c.c
    src/sbmv/BLAS_zsbmv_z_d.cpp
    src/sbmv/BLAS_zsbmv_z_d_x-f2c.c
    src/sbmv/BLAS_zsbmv_z_d_x.cpp
    src/spmv/BLAS_cspmv_c_s-f2c.c
    src/spmv/BLAS_cspmv_c_s.cpp
    src/spmv/BLAS_cspmv_c_s_x-f2c.c
    src/spmv/BLAS_cspmv_c_s_x.cpp
    src/spmv/BLAS_cspmv_s_c-f2c.c
    src/spmv/BLAS_cspmv_s_c.cpp
    src/spmv/BLAS_cspmv_s_c_x-f2c.c
    src/spmv/BLAS_cspmv_s_c_x.cpp
    src/spmv/BLAS_cspmv_s_s-f2c.c
    src/spmv/BLAS_cspmv_s_s.cpp
    src/spmv/BLAS_cspmv_s_s_x-f2c.c
    src/spmv/BLAS_cspmv_s_s_x.cpp
    src/spmv/BLAS_cspmv_x-f2c.c
    src/spmv/BLAS_cspmv_x.cpp
    src/spmv/BLAS_dspmv_d_s-f2c.c
    src/spmv/BLAS_dspmv_d_s.cpp
    src/spmv/BLAS_dspmv_d_s_x-f2c.c
    src/spmv/BLAS_dspmv_d_s_x.cpp
    src/spmv/BLAS_dspmv_s_d-f2c.c
    src/spmv/BLAS_dspmv_s_d.cpp
    src/spmv/BLAS_dspmv_s_d_x-f2c.c
    src/spmv/BLAS_dspmv_s_d_x.cpp
    src/spmv/BLAS_dspmv_s_s-f2c.c
    src/spmv/BLAS_dspmv_s_s.cpp
    src/spmv/BLAS_dspmv_s_s_x-f2c.c
    src/spmv/BLAS_dspmv_s_s_x.cpp
    src/spmv/BLAS_dspmv_x-f2c.c
    src/spmv/BLAS_dspmv_x.cpp
    src/spmv/BLAS_sspmv_x-f2c.c
    src/spmv/BLAS_sspmv_x.cpp
    src/spmv/BLAS_zspmv_c_c-f2c.c
    src/spmv/BLAS_zspmv_c_c.cpp
    src/spmv/BLAS_zspmv_c_c_x-f2c.c
    src/spmv/BLAS_zspmv_c_c_x.cpp
    src/spmv/BLAS_zspmv_c_z-f2c.c
    src/spmv/BLAS_zspmv_c_z.cpp
    src/spmv/BLAS_zspmv_c_z_x-f2c.c
    src/spmv/BLAS_zspmv_c_z_x.cpp
    src/spmv/BLAS_zspmv_d_d-f2c.c
    src/spmv/BLAS_zspmv_d_d.cpp
    src/spmv/BLAS_zspmv_d_d_x-f2c.c
    src/spmv/BLAS_zspmv_d_d_x.cpp
    src/spmv/BLAS_zspmv_d_z-f2c.c
    src/spmv/BLAS_zspmv_d_z.cpp
    src/spmv/BLAS_zspmv_d_z_x-f2c.c
    src/spmv/BLAS_zspmv_d_z_x.cpp
    src/spmv/BLAS_zspmv_x-f2c.c
    src/spmv/BLAS_zspmv_x.cpp
    src/spmv/BLAS_zspmv_z_c-f2c.c
    src/spmv/BLAS_zspmv_z_c.cpp
    src/spmv/BLAS_zspmv_z_c_x-f2c.c
    src/spmv/BLAS_zspmv_z_c_x.cpp
    src/spmv/BLAS_zspmv_z_d-f2c.c
    src/spmv/BLAS_zspmv_z_d.cpp
    src/spmv/BLAS_zspmv_z_d_x-f2c.c
    src/spmv/BLAS_zspmv_z_d_x.cpp
    src/sum/BLAS_csum_x-f2c.c
    src/sum/BLAS_csum_x.cpp
    src/sum/BLAS_dsum_x-f2c.c
    src/sum/BLAS_dsum_x.cpp
    src/sum/BLAS_ssum_x-f2c.c
    src/sum/BLAS_ssum_x.cpp
    src/sum/BLAS_zsum_x-f2c.c
    src/sum/BLAS_zsum_x.cpp
    src/symm/BLAS_csymm_c_s-f2c.c
    src/symm/BLAS_csymm_c_s.cpp
    src/symm/BLAS_csymm_c_s_x-f2c.c
    src/symm/BLAS_csymm_c_s_x.cpp
    src/symm/BLAS_csymm_s_c-f2c.c
    src/symm/BLAS_csymm_s_c.cpp
    src/symm/BLAS_csymm_s_c_x-f2c.c
    src/symm/BLAS_csymm_s_c_x.cpp
    src/symm/BLAS_csymm_s_s-f2c.c
    src/symm/BLAS_csymm_s_s.cpp
    src/symm/BLAS_csymm_s_s_x-f2c.c
    src/symm/BLAS_csymm_s_s_x.cpp
    src/symm/BLAS_csymm_x-f2c.c
    src/symm/BLAS_csymm_x.cpp
    src/symm/BLAS_dsymm_d_s-f2c.c
    src/symm/BLAS_dsymm_d_s.cpp
    src/symm/BLAS_dsymm_d_s_x-f2c.c
    src/symm/BLAS_dsymm_d_s_x.cpp
    src/symm/BLAS_dsymm_s_d-f2c.c
    src/symm/BLAS_dsymm_s_d.cpp
    src/symm/BLAS_dsymm_s_d_x-f2c.c
    src/symm/BLAS_dsymm_s_d_x.cpp
    src/symm/BLAS_dsymm_s_s-f2c.c
    src/symm/BLAS_dsymm_s_s.cpp
    src/symm/BLAS_dsymm_s_s_x-f2c.c
    src/symm/BLAS_dsymm_s_s_x.cpp
    src/symm/BLAS_dsymm_x-f2c.c
    src/symm/BLAS_dsymm_x.cpp
    src/symm/BLAS_ssymm_x-f2c.c
    src/symm/BLAS_ssymm_x.cpp
    src/symm/BLAS_zsymm_c_c-f2c.c
    src/symm/BLAS_zsymm_c_c.cpp
    src/symm/BLAS_zsymm_c_c_x-f2c.c
    src/symm/BLAS_zsymm_c_c_x.cpp
    src/symm/BLAS_zsymm_c_z-f2c.c
    src/symm/BLAS_zsymm_c_z.cpp
    src/symm/BLAS_zsymm_c_z_x-f2c.c
    src/symm/BLAS_zsymm_c_z_x.cpp
    src/symm/BLAS_zsymm_d_d-f2c.c
    src/symm/BLAS_zsymm_d_d.cpp
    src/symm/BLAS_zsymm_d_d_x-f2c.c
    src/symm/BLAS_zsymm_d_d_x.cpp
    src/symm/BLAS_zsymm_d_z-f2c.c
    src/symm/BLAS_zsymm_d_z.cpp
    src/symm/BLAS_zsymm_d_z_x-f2c.c
    src/symm/BLAS_zsymm_d_z_x.cpp
    src/symm/BLAS_zsymm_x-f2c.c
    src/symm/BLAS_zsymm_x.cpp
    src/symm/BLAS_zsymm_z_c-f2c.c
    src/symm/BLAS_zsymm_z_c.cpp
    src/symm/BLAS_zsymm_z_c_x-f2c.c
    src/symm/BLAS_zsymm_z_c_x.cpp
    src/symm/BLAS_zsymm_z_d-f2c.c
    src/symm/BLAS_zsymm_z_d.cpp
    src/symm/BLAS_zsymm_z_d_x-f2c.c
    src/symm/BLAS_zsymm_z_d_x.cpp
    src/symv/BLAS_csymv_c_s-f2c.c
    src/symv/BLAS_csymv_c_s.cpp
    src/symv/BLAS_csymv_c_s_x-f2c.c
    src/symv/BLAS_csymv_c_s_x.cpp
    src/symv/BLAS_csymv_s_c-f2c.c
    src/symv/BLAS_csymv_s_c.cpp
    src/symv/BLAS_csymv_s_c_x-f2c.c
    src/symv/BLAS_csymv_s_c_x.cpp
    src/symv/BLAS_csymv_s_s-f2c.c
    src/symv/BLAS_csymv_s_s.cpp
    src/symv/BLAS_csymv_s_s_x-f2c.c
    src/symv/BLAS_csymv_s_s_x.cpp
    src/symv/BLAS_csymv_x-f2c.c
    src/symv/BLAS_csymv_x.cpp
    src/symv/BLAS_dsymv_d_s-f2c.c
    src/symv/BLAS_dsymv_d_s.cpp
    src/symv/BLAS_dsymv_d_s_x-f2c.c
    src/symv/BLAS_dsymv_d_s_x.cpp
    src/symv/BLAS_dsymv_s_d-f2c.c
    src/symv/BLAS_dsymv_s_d.cpp
    src/symv/BLAS_dsymv_s_d_x-f2c.c
    src/symv/BLAS_dsymv_s_d_x.cpp
    src/symv/BLAS_dsymv_s_s-f2c.c
    src/symv/BLAS_dsymv_s_s.cpp
    src/symv/BLAS_dsymv_s_s_x-f2c.c
    src/symv/BLAS_dsymv_s_s_x.cpp
    src/symv/BLAS_dsymv_x-f2c.c
    src/symv/BLAS_dsymv_x.cpp
    src/symv/BLAS_ssymv_x-f2c.c
    src/symv/BLAS_ssymv_x.cpp
    src/symv/BLAS_zsymv_c_c-f2c.c
    src/symv/BLAS_zsymv_c_c.cpp
    src/symv/BLAS_zsymv_c_c_x-f2c.c
    src/symv/BLAS_zsymv_c_c_x.cpp
    src/symv/BLAS_zsymv_c_z-f2c.c
    src/symv/BLAS_zsymv_c_z.cpp
    src/symv/BLAS_zsymv_c_z_x-f2c.c
    src/symv/BLAS_zsymv_c_z_x.cpp
    src/symv/BLAS_zsymv_d_d-f2c.c
    src/symv/BLAS_zsymv_d_d.cpp
    src/symv/BLAS_zsymv_d_d_x-f2c.c
    src/symv/BLAS_zsymv_d_d_x.cpp
    src/symv/BLAS_zsymv_d_z-f2c.c
    src/symv/BLAS_zsymv_d_z.cpp
    src/symv/BLAS_zsymv_d_z_x-f2c.c
    src/symv/BLAS_zsymv_d_z_x.cpp
    src/symv/BLAS_zsymv_x-f2c.c
    src/symv/BLAS_zsymv_x.cpp
    src/symv/BLAS_zsymv_z_c-f2c.c
    src/symv/BLAS_zsymv_z_c.cpp
    src/symv/BLAS_zsymv_z_c_x-f2c.c
    src/symv/BLAS_zsymv_z_c_x.cpp
    src/symv/BLAS_zsymv_z_d-f2c.c
    src/symv/BLAS_zsymv_z_d.cpp
    src/symv/BLAS_zsymv_z_d_x-f2c.c
    src/symv/BLAS_zsymv_z_d_x.cpp
    src/symv2/BLAS_csymv2_c_s-f2c.c
    src/symv2/BLAS_csymv2_c_s.cpp
    src/symv2/BLAS_csymv2_c_s_x-f2c.c
    src/symv2/BLAS_csymv2_c_s_x.cpp
    src/symv2/BLAS_csymv2_s_c-f2c.c
    src/symv2/BLAS_csymv2_s_c.cpp
    src/symv2/BLAS_csymv2_s_c_x-f2c.c
    src/symv2/BLAS_csymv2_s_c_x.cpp
    src/symv2/BLAS_csymv2_s_s-f2c.c
    src/symv2/BLAS_csymv2_s_s.cpp
    src/symv2/BLAS_csymv2_s_s_x-f2c.c
    src/symv2/BLAS_csymv2_s_s_x.cpp
    src/symv2/BLAS_csymv2_x-f2c.c
    src/symv2/BLAS_csymv2_x.cpp
    src/symv2/BLAS_dsymv2_d_s-f2c.c
    src/symv2/BLAS_dsymv2_d_s.cpp
    src/symv2/BLAS_dsymv2_d_s_x-f2c.c
    src/symv2/BLAS_dsymv2_d_s_x.cpp
    src/symv2/BLAS_dsymv2_s_d-f2c.c
    src/symv2/BLAS_dsymv2_s_d.cpp
    src/symv2/BLAS_dsymv2_s_d_x-f2c.c
    src/symv2/BLAS_dsymv2_s_d_x.cpp
    src/symv2/BLAS_dsymv2_s_s-f2c.c
    src/symv2/BLAS_dsymv2_s_s.cpp
    src/symv2/BLAS_dsymv2_s_s_x-f2c.c
    src/symv2/BLAS_dsymv2_s_s_x.cpp
    src/symv2/BLAS_dsymv2_x-f2c.c
    src/symv2/BLAS_dsymv2_x.cpp
    src/symv2/BLAS_ssymv2_x-f2c.c
    src/symv2/BLAS_ssymv2_x.cpp
    src/symv2/BLAS_zsymv2_c_c-f2c.c
    src/symv2/BLAS_zsymv2_c_c.cpp
    src/symv2/BLAS_zsymv2_c_c_x-f2c.c
    src/symv2/BLAS_zsymv2_c_c_x.cpp
    src/symv2/BLAS_zsymv2_c_z-f2c.c
    src/symv2/BLAS_zsymv2_c_z.cpp
    src/symv2/BLAS_zsymv2_c_z_x-f2c.c
    src/symv2/BLAS_zsymv2_c_z_x.cpp
    src/symv2/BLAS_zsymv2_d_d-f2c.c
    src/symv2/BLAS_zsymv2_d_d.cpp
    src/symv2/BLAS_zsymv2_d_d_x-f2c.c
    src/symv2/BLAS_zsymv2_d_d_x.cpp
    src/symv2/BLAS_zsymv2_d_z-f2c.c
    src/symv2/BLAS_zsymv2_d_z.cpp
    src/symv2/BLAS_zsymv2_d_z_x-f2c.c
    src/symv2/BLAS_zsymv2_d_z_x.cpp
    src/symv2/BLAS_zsymv2_x-f2c.c
    src/symv2/BLAS_zsymv2_x.cpp
    src/symv2/BLAS_zsymv2_z_c-f2c.c
    src/symv2/BLAS_zsymv2_z_c.cpp
    src/symv2/BLAS_zsymv2_z_c_x-f2c.c
    src/symv2/BLAS_zsymv2_z_c_x.cpp
    src/symv2/BLAS_zsymv2_z_d-f2c.c
    src/symv2/BLAS_zsymv2_z_d.cpp
    src/symv2/BLAS_zsymv2_z_d_x-f2c.c
    src/symv2/BLAS_zsymv2_z_d_x.cpp
    src/tbsv/BLAS_ctbsv_s-f2c.c
    src/tbsv/BLAS_ctbsv_s.cpp
    src/tbsv/BLAS_ctbsv_s_x-f2c.c
    src/tbsv/BLAS_ctbsv_s_x.cpp
    src/tbsv/BLAS_ctbsv_x-f2c.c
    src/tbsv/BLAS_ctbsv_x.cpp
    src/tbsv/BLAS_dtbsv_s-f2c.c
    src/tbsv/BLAS_dtbsv_s.cpp
    src/tbsv/BLAS_dtbsv_s_x-f2c.c
    src/tbsv/BLAS_dtbsv_s_x.cpp
    src/tbsv/BLAS_dtbsv_x-f2c.c
    src/tbsv/BLAS_dtbsv_x.cpp
    src/tbsv/BLAS_stbsv_x-f2c.c
    src/tbsv/BLAS_stbsv_x.cpp
    src/tbsv/BLAS_ztbsv_c-f2c.c
    src/tbsv/BLAS_ztbsv_c.cpp
    src/tbsv/BLAS_ztbsv_c_x-f2c.c
    src/tbsv/BLAS_ztbsv_c_x.cpp
    src/tbsv/BLAS_ztbsv_d-f2c.c
    src/tbsv/BLAS_ztbsv_d.cpp
    src/tbsv/BLAS_ztbsv_d_x-f2c.c
    src/tbsv/BLAS_ztbsv_d_x.cpp
    src/tbsv/BLAS_ztbsv_x-f2c.c
    src/tbsv/BLAS_ztbsv_x.cpp
    src/tpmv/BLAS_ctpmv_s-f2c.c
    src/tpmv/BLAS_ctpmv_s.cpp
    src/tpmv/BLAS_ctpmv_s_x-f2c.c
    src/tpmv/BLAS_ctpmv_s_x.cpp
    src/tpmv/BLAS_ctpmv_x-f2c.c
    src/tpmv/BLAS_ctpmv_x.cpp
    src/tpmv/BLAS_dtpmv_s-f2c.c
    src/tpmv/BLAS_dtpmv_s.cpp
    src/tpmv/BLAS_dtpmv_s_x-f2c.c
    src/tpmv/BLAS_dtpmv_s_x.cpp
    src/tpmv/BLAS_dtpmv_x-f2c.c
    src/tpmv/BLAS_dtpmv_x.cpp
    src/tpmv/BLAS_stpmv_x-f2c.c
    src/tpmv/BLAS_stpmv_x.cpp
    src/tpmv/BLAS_ztpmv_c-f2c.c
    src/tpmv/BLAS_ztpmv_c.cpp
    src/tpmv/BLAS_ztpmv_c_x-f2c.c
    src/tpmv/BLAS_ztpmv_c_x.cpp
    src/tpmv/BLAS_ztpmv_d-f2c.c
    src/tpmv/BLAS_ztpmv_d.cpp
    src/tpmv/BLAS_ztpmv_d_x-f2c.c
    src/tpmv/BLAS_ztpmv_d_x.cpp
    src/tpmv/BLAS_ztpmv_x-f2c.c
    src/tpmv/BLAS_ztpmv_x.cpp
    src/trmv/BLAS_ctrmv_s-f2c.c
    src/trmv/BLAS_ctrmv_s.cpp
    src/trmv/BLAS_ctrmv_s_x-f2c.c
    src/trmv/BLAS_ctrmv_s_x.cpp
    src/trmv/BLAS_ctrmv_x-f2c.c
    src/trmv/BLAS_ctrmv_x.cpp
    src/trmv/BLAS_dtrmv_s-f2c.c
    src/trmv/BLAS_dtrmv_s.cpp
    src/trmv/BLAS_dtrmv_s_x-f2c.c
    src/trmv/BLAS_dtrmv_s_x.cpp
    src/trmv/BLAS_dtrmv_x-f2c.c
    src/trmv/BLAS_dtrmv_x.cpp
    src/trmv/BLAS_strmv_x-f2c.c
    src/trmv/BLAS_strmv_x.cpp
    src/trmv/BLAS_ztrmv_c-f2c.c
    src/trmv/BLAS_ztrmv_c.cpp
    src/trmv/BLAS_ztrmv_c_x-f2c.c
    src/trmv/BLAS_ztrmv_c_x.cpp
    src/trmv/BLAS_ztrmv_d-f2c.c
    src/trmv/BLAS_ztrmv_d.cpp
    src/trmv/BLAS_ztrmv_d_x-f2c.c
    src/trmv/BLAS_ztrmv_d_x.cpp
    src/trmv/BLAS_ztrmv_x-f2c.c
    src/trmv/BLAS_ztrmv_x.cpp
    src/trsv/BLAS_ctrsv_s-f2c.c
    src/trsv/BLAS_ctrsv_s.cpp
    src/trsv/BLAS_ctrsv_s_x-f2c.c
    src/trsv/BLAS_ctrsv_s_x.cpp
    src/trsv/BLAS_ctrsv_x-f2c.c
    src/trsv/BLAS_ctrsv_x.cpp
    src/trsv/BLAS_dtrsv_s-f2c.c
    src/trsv/BLAS_dtrsv_s.cpp
    src/trsv/BLAS_dtrsv_s_x-f2c.c
    src/trsv/BLAS_dtrsv_s_x.cpp
    src/trsv/BLAS_dtrsv_x-f2c.c
    src/trsv/BLAS_dtrsv_x.cpp
    src/trsv/BLAS_strsv_x-f2c.c
    src/trsv/BLAS_strsv_x.cpp
    src/trsv/BLAS_ztrsv_c-f2c.c
    src/trsv/BLAS_ztrsv_c.cpp
    src/trsv/BLAS_ztrsv_c_x-f2c.c
    src/trsv/BLAS_ztrsv_c_x.cpp
    src/trsv/BLAS_ztrsv_d-f2c.c
    src/trsv/BLAS_ztrsv_d.cpp
    src/trsv/BLAS_ztrsv_d_x-f2c.c
    src/trsv/BLAS_ztrsv_d_x.cpp
    src/trsv/BLAS_ztrsv_x-f2c.c
    src/trsv/BLAS_ztrsv_x.cpp
    src/waxpby/BLAS_cwaxpby_c_s-f2c.c
    src/waxpby/BLAS_cwaxpby_c_s.cpp
    src/waxpby/BLAS_cwaxpby_c_s_x-f2c.c
    src/waxpby/BLAS_cwaxpby_c_s_x.cpp
    src/waxpby/BLAS_cwaxpby_s_c-f2c.c
    src/waxpby/BLAS_cwaxpby_s_c.cpp
    src/waxpby/BLAS_cwaxpby_s_c_x-f2c.c
    src/waxpby/BLAS_cwaxpby_s_c_x.cpp
    src/waxpby/BLAS_cwaxpby_s_s-f2c.c
    src/waxpby/BLAS_cwaxpby_s_s.cpp
    src/waxpby/BLAS_cwaxpby_s_s_x-f2c.c
    src/waxpby/BLAS_cwaxpby_s_s_x.cpp
    src/waxpby/BLAS_cwaxpby_x-f2c.c
    src/waxpby/BLAS_cwaxpby_x.cpp
    src/waxpby/BLAS_dwaxpby_d_s-f2c.c
    src/waxpby/BLAS_dwaxpby_d_s.cpp
    src/waxpby/BLAS_dwaxpby_d_s_x-f2c.c
    src/waxpby/BLAS_dwaxpby_d_s_x.cpp
    src/waxpby/BLAS_dwaxpby_s_d-f2c.c
    src/waxpby/BLAS_dwaxpby_s_d.cpp
    src/waxpby/BLAS_dwaxpby_s_d_x-f2c.c
    src/waxpby/BLAS_dwaxpby_s_d_x.cpp
    src/waxpby/BLAS_dwaxpby_s_s-f2c.c
    src/waxpby/BLAS_dwaxpby_s_s.cpp
    src/waxpby/BLAS_dwaxpby_s_s_x-f2c.c
    src/waxpby/BLAS_dwaxpby_s_s_x.cpp
    src/waxpby/BLAS_dwaxpby_x-f2c.c
    src/waxpby/BLAS_dwaxpby_x.cpp
    src/waxpby/BLAS_swaxpby_x-f2c.c
    src/waxpby/BLAS_swaxpby_x.cpp
    src/waxpby/BLAS_zwaxpby_c_c-f2c.c
    src/waxpby/BLAS_zwaxpby_c_c.cpp
    src/waxpby/BLAS_zwaxpby_c_c_x-f2c.c
    src/waxpby/BLAS_zwaxpby_c_c_x.cpp
    src/waxpby/BLAS_zwaxpby_c_z-f2c.c
    src/waxpby/BLAS_zwaxpby_c_z.cpp
    src/waxpby/BLAS_zwaxpby_c_z_x-f2c.c
    src/waxpby/BLAS_zwaxpby_c_z_x.cpp
    src/waxpby/BLAS_zwaxpby_d_d-f2c.c
    src/waxpby/BLAS_zwaxpby_d_d.cpp
    src/waxpby/BLAS_zwaxpby_d_d_x-f2c.c
    src/waxpby/BLAS_zwaxpby_d_d_x.cpp
    src/waxpby/BLAS_zwaxpby_d_z-f2c.c
    src/waxpby/BLAS_zwaxpby_d_z.cpp
    src/waxpby/BLAS_zwaxpby_d_z_x-f2c.c
    src/waxpby/BLAS_zwaxpby_d_z_x.cpp
    src/waxpby/BLAS_zwaxpby_x-f2c.c
    src/waxpby/BLAS_zwaxpby_x.cpp
    src/waxpby/BLAS_zwaxpby_z_c-f2c.c
    src/waxpby/BLAS_zwaxpby_z_c.cpp
    src/waxpby/BLAS_zwaxpby_z_c_x-f2c.c
    src/waxpby/BLAS_zwaxpby_z_c_x.cpp
    src/waxpby/BLAS_zwaxpby_z_d-f2c.c
    src/waxpby/BLAS_zwaxpby_z_d.cpp
    src/waxpby/BLAS_zwaxpby_z_d_x-f2c.c
    src/waxpby/BLAS_zwaxpby_z_d_x.cpp

  PUBLIC
    FILE_SET xblas_headers
  TYPE HEADERS
  BASE_DIRS
    include/XBLAS
  FILES
    include/XBLAS/axpby.hpp
    include/XBLAS/blas_dense_proto.h
    include/XBLAS/blas_enum.h
    include/XBLAS/blas_extended.h
    include/XBLAS/blas_extended_private.h
    include/XBLAS/blas_extended_proto.h
    include/XBLAS/blas_fpu.h
    include/XBLAS/blas_malloc.h
    include/XBLAS/dot.hpp
    include/XBLAS/double_double.hpp
    include/XBLAS/f2c-bridge.h
    include/XBLAS/gbmv.hpp
    include/XBLAS/gbmv2.hpp
    include/XBLAS/ge_sum_mv.hpp
    include/XBLAS/gemm.hpp
    include/XBLAS/gemv.hpp
    include/XBLAS/gemv2.hpp
    include/XBLAS/hbmv.hpp
    include/XBLAS/hemm.hpp
    include/XBLAS/hemv.hpp
    include/XBLAS/hemv2.hpp
    include/XBLAS/hpmv.hpp
    include/XBLAS/impl.hpp
    include/XBLAS/sbmv.hpp
    include/XBLAS/spmv.hpp
    include/XBLAS/sum.hpp
    include/XBLAS/symm.hpp
    include/XBLAS/symv.hpp
    include/XBLAS/symv2.hpp
    include/XBLAS/tbsv.hpp
    include/XBLAS/tpmv.hpp
    include/XBLAS/trmv.hpp
    include/XBLAS/trsv.hpp
    include/XBLAS/waxpby.hpp
)

option(MDKXBLAS_USE_FLOAT128 "use std::float128_t" OFF)
option(MDKXBLAS_USE_XERBLA "use xerbla_array" OFF)
option(MDKXBLAS_NO_BLAS_MALLOC "don't use blas_malloc" OFF)
option(MDKXBLAS_DEBUG_BLAS_MALLOC "enable debug blas_malloc" OFF)

target_compile_features(xblas PRIVATE cxx_std_23)

# This library has a C/fortran API, exceptions/rtti do not apply.
# We don't want c/fortran client code to have linking issues if
# the stdc++ library is not linked.
target_compile_options(xblas PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti>
)

target_compile_definitions(xblas PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:_HAS_EXCEPTIONS=0>
)

if (MDKXBLAS_USE_FLOAT128)
  message(STATUS "using std::float128_t for extended presicion")
  target_compile_definitions(xblas PRIVATE XBLAS_USE_FLOAT128)
else()
  message(STATUS "using XBLAS::double_double for extended presicion")
endif()

if (MDKXBLAS_USE_XERBLA)
  message(STATUS "blas_error forwards to xerbla")
  target_compile_definitions(xblas PRIVATE XBLAS_USE_XERBLA)
endif()

if (MDKXBLAS_NO_BLAS_MALLOC)
  message(STATUS "no blas_malloc wrapper around std::malloc")
  target_compile_definitions(xblas PRIVATE XBLAS_NO_BLAS_MALLOC)
elseif (MDKXBLAS_DEBUG_BLAS_MALLOC)
  message(STATUS "blas_malloc debug enabled")
  target_compile_definitions(xblas PRIVATE XBLAS_DEBUG_BLAS_MALLOC)
endif()

# ------------------------------------------------------------
# tests
# ------------------------------------------------------------

enable_testing()

option(MDKXBLAS_COMBINE_TESTS "combine all tests into one executable" OFF)

if (MDKXBLAS_COMBINE_TESTS)
  add_test(
    NAME ctests
    COMMAND ctests
  )

  add_executable(ctests)
  target_sources(ctests
    PRIVATE
      test/ctests/common/dummy_main.c
      test/ctests/main.c
      test/ctests/test-axpby/do_test_axpby.c
      test/ctests/test-dot/BLAS_dot_testgen.c
      test/ctests/test-dot/copy_vector.c
      test/ctests/test-dot/do_test_dot.c
      test/ctests/test-dot/print_vector.c
      test/ctests/test-dot/test_dot.c
      test/ctests/test-dot/testgen_BLAS_cdot.c
      test/ctests/test-dot/testgen_BLAS_ddot.c
      test/ctests/test-dot/testgen_BLAS_sdot.c
      test/ctests/test-dot/testgen_BLAS_zdot.c
      test/ctests/test-dot/testgen_aux.c
      test/ctests/test-dot2/BLAS_dot2_testgen.c
      test/ctests/test-dot2/do_test_dot2.c
      test/ctests/test-dot2/dot2.c
      test/ctests/test-dot2/r_truth2.c
      test/ctests/test-dot2/test_dot2.c
      test/ctests/test-dot2/testgen_BLAS_cdot2.c
      test/ctests/test-dot2/testgen_BLAS_ddot2.c
      test/ctests/test-dot2/testgen_BLAS_sdot2.c
      test/ctests/test-dot2/testgen_BLAS_zdot2.c
      test/ctests/test-gbmv/BLAS_gbmv_testgen.c
      test/ctests/test-gbmv/do_test_gbmv.c
      test/ctests/test-gbmv/gbmv-support.c
      test/ctests/test-gbmv2/BLAS_gbmv2_testgen.c
      test/ctests/test-gbmv2/do_test_gbmv2.c
      test/ctests/test-ge_sum_mv/BLAS_ge_sum_mv_testgen.c
      test/ctests/test-ge_sum_mv/do_test_ge_sum_mv.c
      test/ctests/test-gemm/BLAS_gemm_testgen.c
      test/ctests/test-gemm/do_test_gemm.c
      test/ctests/test-gemv/BLAS_gemv_testgen.c
      test/ctests/test-gemv/do_test_gemv.c
      test/ctests/test-gemv/gemv-support.c
      test/ctests/test-gemv2/BLAS_gemv2_testgen.c
      test/ctests/test-gemv2/do_test_gemv2.c
      test/ctests/test-hbmv/BLAS_hbmv_testgen.c
      test/ctests/test-hbmv/do_test_hbmv.c
      test/ctests/test-hbmv/hbmv-support.c
      test/ctests/test-hemm/BLAS_hemm_testgen.c
      test/ctests/test-hemm/do_test_hemm.c
      test/ctests/test-hemv/BLAS_hemv_testgen.c
      test/ctests/test-hemv/do_test_hemv.c
      test/ctests/test-hemv/hemv-support.c
      test/ctests/test-hemv2/BLAS_hemv2_testgen.c
      test/ctests/test-hemv2/do_test_hemv2.c
      test/ctests/test-hpmv/BLAS_hpmv_testgen.c
      test/ctests/test-hpmv/do_test_hpmv.c
      test/ctests/test-hpmv/hpmv-support.c
      test/ctests/test-sbmv/BLAS_sbmv_testgen.c
      test/ctests/test-sbmv/do_test_sbmv.c
      test/ctests/test-sbmv/sbmv-support.c
      test/ctests/test-spmv/BLAS_spmv_testgen.c
      test/ctests/test-spmv/do_test_spmv.c
      test/ctests/test-spmv/spmv-support.c
      test/ctests/test-sum/BLAS_sum_testgen.c
      test/ctests/test-sum/do_test_sum.c
      test/ctests/test-sum/sum-support.c
      test/ctests/test-symm/BLAS_symm_testgen.c
      test/ctests/test-symm/do_test_symm.c
      test/ctests/test-symv/BLAS_symv_testgen.c
      test/ctests/test-symv/do_test_symv.c
      test/ctests/test-symv/symv-support.c
      test/ctests/test-symv2/BLAS_symv2_testgen.c
      test/ctests/test-symv2/do_test_symv2.c
      test/ctests/test-tbsv/BLAS_tbsv_testgen.c
      test/ctests/test-tbsv/do_test_tbsv.c
      test/ctests/test-tbsv/tbsv-support.c
      test/ctests/test-tpmv/BLAS_tpmv_testgen.c
      test/ctests/test-tpmv/do_test_tpmv.c
      test/ctests/test-tpmv/tpmv-support.c
      test/ctests/test-trmv/BLAS_trmv_testgen.c
      test/ctests/test-trmv/do_test_trmv.c
      test/ctests/test-trmv/trmv-support.c
      test/ctests/test-trsv/BLAS_dot_testgen_trsv.c
      test/ctests/test-trsv/BLAS_trsv_testgen.c
      test/ctests/test-trsv/do_test_trsv.c
      test/ctests/test-trsv/testgen_BLAS_sdot_x.c
      test/ctests/test-trsv/trsv-support.c
      test/ctests/test-waxpby/do_test_waxpby.c

      test/ctests/blas_extended_test.h
  )

  target_include_directories(ctests PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )

  target_link_libraries(ctests PRIVATE xblas)

  set_target_properties(ctests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  message(STATUS "combined ctests executable")
  target_compile_definitions(ctests PRIVATE XBLAS_COMBINE_TESTS)
else()
  add_test(
    NAME ctests_axpby
    COMMAND ctests_axpby 10 1 1.0 1 1
  )
  add_executable(ctests_axpby)
  target_sources(ctests_axpby
    PRIVATE
      test/ctests/test-axpby/do_test_axpby.c

      test/ctests/test-dot/BLAS_dot_testgen.c
      test/ctests/test-dot/copy_vector.c
      test/ctests/test-dot/print_vector.c
      test/ctests/test-dot/test_dot.c
      test/ctests/test-dot/testgen_aux.c
      test/ctests/test-dot/testgen_BLAS_cdot.c
      test/ctests/test-dot/testgen_BLAS_ddot.c
      test/ctests/test-dot/testgen_BLAS_sdot.c
      test/ctests/test-dot/testgen_BLAS_zdot.c

      test/ctests/blas_extended_test.h
  )
  target_include_directories(ctests_axpby PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_axpby PRIVATE xblas)
  set_target_properties(ctests_axpby PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_dot
    COMMAND ctests_dot 20 1 1.0 0 1
  )
  add_executable(ctests_dot)
  target_sources(ctests_dot
    PRIVATE
      test/ctests/test-dot/BLAS_dot_testgen.c
      test/ctests/test-dot/copy_vector.c
      test/ctests/test-dot/do_test_dot.c
      test/ctests/test-dot/print_vector.c
      test/ctests/test-dot/test_dot.c
      test/ctests/test-dot/testgen_BLAS_cdot.c
      test/ctests/test-dot/testgen_BLAS_ddot.c
      test/ctests/test-dot/testgen_BLAS_sdot.c
      test/ctests/test-dot/testgen_BLAS_zdot.c
      test/ctests/test-dot/testgen_aux.c
      test/ctests/blas_extended_test.h
  )
  target_include_directories(ctests_dot PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_dot PRIVATE xblas)
  set_target_properties(ctests_dot PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_dot2
    COMMAND ctests_dot2 20 1 1.0 0 1
  )
  add_executable(ctests_dot2)
  target_sources(ctests_dot2
    PRIVATE
      test/ctests/test-dot2/BLAS_dot2_testgen.c
      test/ctests/test-dot2/do_test_dot2.c
      test/ctests/test-dot2/dot2.c
      test/ctests/test-dot2/r_truth2.c
      test/ctests/test-dot2/test_dot2.c
      test/ctests/test-dot2/testgen_BLAS_cdot2.c
      test/ctests/test-dot2/testgen_BLAS_ddot2.c
      test/ctests/test-dot2/testgen_BLAS_sdot2.c
      test/ctests/test-dot2/testgen_BLAS_zdot2.c

      test/ctests/test-dot/copy_vector.c
      test/ctests/test-dot/print_vector.c
      test/ctests/test-dot/testgen_aux.c
      test/ctests/test-dot/testgen_BLAS_cdot.c
      test/ctests/test-dot/testgen_BLAS_zdot.c

      test/ctests/blas_extended_test.h
  )
  target_include_directories(ctests_dot2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_dot2 PRIVATE xblas)
  set_target_properties(ctests_dot2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_gbmv
    COMMAND ctests_gbmv 12 1 1.0 0 0.1
  )
  add_executable(ctests_gbmv)
  target_sources(ctests_gbmv
    PRIVATE
      test/ctests/test-gbmv/BLAS_gbmv_testgen.c
      test/ctests/test-gbmv/do_test_gbmv.c
      test/ctests/test-gbmv/gbmv-support.c

      test/ctests/test-dot/BLAS_dot_testgen.c
      test/ctests/test-dot/copy_vector.c
      test/ctests/test-dot/print_vector.c
      test/ctests/test-dot/test_dot.c
      test/ctests/test-dot/testgen_aux.c
      test/ctests/test-dot/testgen_BLAS_cdot.c
      test/ctests/test-dot/testgen_BLAS_ddot.c
      test/ctests/test-dot/testgen_BLAS_sdot.c
      test/ctests/test-dot/testgen_BLAS_zdot.c

      test/ctests/blas_extended_test.h
  )
  target_include_directories(ctests_gbmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_gbmv PRIVATE xblas)
  set_target_properties(ctests_gbmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_gbmv2
    COMMAND ctests_gbmv2 12 1 1.0 0 0.1
  )
  add_executable(ctests_gbmv2)
  target_sources(ctests_gbmv2
    PRIVATE
      test/ctests/test-gbmv2/BLAS_gbmv2_testgen.c
      test/ctests/test-gbmv2/do_test_gbmv2.c

      test/ctests/test-dot2/BLAS_dot2_testgen.c
      test/ctests/test-dot/copy_vector.c
      test/ctests/test-dot/print_vector.c
      test/ctests/test-dot/testgen_aux.c
      test/ctests/test-dot/testgen_BLAS_cdot.c
      test/ctests/test-dot/testgen_BLAS_zdot.c
      test/ctests/test-dot2/dot2.c
      test/ctests/test-dot2/r_truth2.c
      test/ctests/test-dot2/test_dot2.c
      test/ctests/test-dot2/testgen_BLAS_cdot2.c
      test/ctests/test-dot2/testgen_BLAS_ddot2.c
      test/ctests/test-dot2/testgen_BLAS_sdot2.c
      test/ctests/test-dot2/testgen_BLAS_zdot2.c
      test/ctests/test-gbmv/gbmv-support.c

      test/ctests/blas_extended_test.h
  )
  target_include_directories(ctests_gbmv2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_gbmv2 PRIVATE xblas)
  set_target_properties(ctests_gbmv2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_ge_sum_mv
    COMMAND ctests_ge_sum_mv 7 1 1.0 3 0.1
  )
  add_executable(ctests_ge_sum_mv)
  target_sources(ctests_ge_sum_mv
    PRIVATE
      test/ctests/test-ge_sum_mv/BLAS_ge_sum_mv_testgen.c
      test/ctests/test-ge_sum_mv/do_test_ge_sum_mv.c
      
      test/ctests/test-dot/BLAS_dot_testgen.c
      test/ctests/test-dot/copy_vector.c
      test/ctests/test-dot/print_vector.c
      test/ctests/test-dot/test_dot.c
      test/ctests/test-dot/testgen_aux.c
      test/ctests/test-dot/testgen_BLAS_cdot.c
      test/ctests/test-dot/testgen_BLAS_ddot.c
      test/ctests/test-dot/testgen_BLAS_sdot.c
      test/ctests/test-dot/testgen_BLAS_zdot.c      
      test/ctests/test-gemv/gemv-support.c

      test/ctests/blas_extended_test.h
  )
  target_include_directories(ctests_ge_sum_mv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_ge_sum_mv PRIVATE xblas)
  set_target_properties(ctests_ge_sum_mv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )
  
  add_test(
    NAME ctests_gemm
    COMMAND ctests_gemm 9 1 1.0 0 0.01
  )
  add_executable(ctests_gemm)
  target_sources(ctests_gemm
    PRIVATE
      test/ctests/test-gemm/BLAS_gemm_testgen.c
      test/ctests/test-gemm/do_test_gemm.c
      
      test/ctests/test-dot/BLAS_dot_testgen.c
      test/ctests/test-dot/test_dot.c
      test/ctests/test-dot/testgen_aux.c
      test/ctests/test-dot/testgen_BLAS_cdot.c
      test/ctests/test-dot/testgen_BLAS_ddot.c
      test/ctests/test-dot/testgen_BLAS_sdot.c
      test/ctests/test-dot/testgen_BLAS_zdot.c      
      test/ctests/test-gemv/gemv-support.c
      
      test/ctests/blas_extended_test.h
  )
  target_include_directories(ctests_gemm PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_gemm PRIVATE xblas)
  set_target_properties(ctests_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )
  
  add_test(
    NAME ctests_gemv
    COMMAND ctests_gemv 12 1 1.0 0 0.1
  )
  add_executable(ctests_gemv)
  target_sources(ctests_gemv
    PRIVATE
      test/ctests/test-gemv/BLAS_gemv_testgen.c
      test/ctests/test-gemv/do_test_gemv.c
      test/ctests/test-gemv/gemv-support.c
      
      test/ctests/test-dot/BLAS_dot_testgen.c
      test/ctests/test-dot/copy_vector.c
      test/ctests/test-dot/print_vector.c
      test/ctests/test-dot/test_dot.c
      test/ctests/test-dot/testgen_aux.c
      test/ctests/test-dot/testgen_BLAS_cdot.c
      test/ctests/test-dot/testgen_BLAS_ddot.c
      test/ctests/test-dot/testgen_BLAS_sdot.c
      test/ctests/test-dot/testgen_BLAS_zdot.c         

      test/ctests/blas_extended_test.h
  )
  target_include_directories(ctests_gemv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_gemv PRIVATE xblas)
  set_target_properties(ctests_gemv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )  

  message(STATUS "separate ctests executables")
endif()
