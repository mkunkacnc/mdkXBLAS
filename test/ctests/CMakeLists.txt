# ------------------------------------------------------------
# ctests
# ------------------------------------------------------------

set(CMAKE_C_FLAGS "-O3" CACHE STRING "")
set(CMAKE_C_FLAGS_RELEASE "-O3" CACHE STRING "")

option(MDKXBLAS_COMBINE_TESTS "combine all tests into one executable" OFF)

if (MDKXBLAS_COMBINE_TESTS)
  add_test(
    NAME ctests
    COMMAND ctests
  )

  add_executable(ctests)
  target_sources(ctests
    PRIVATE
      common/dummy_main.c
      main.c
      test-axpby/do_test_axpby.c
      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/do_test_dot.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-dot/testgen_aux.c
      test-dot2/BLAS_dot2_testgen.c
      test-dot2/do_test_dot2.c
      test-dot2/dot2.c
      test-dot2/r_truth2.c
      test-dot2/test_dot2.c
      test-dot2/testgen_BLAS_cdot2.c
      test-dot2/testgen_BLAS_ddot2.c
      test-dot2/testgen_BLAS_sdot2.c
      test-dot2/testgen_BLAS_zdot2.c
      test-gbmv/BLAS_gbmv_testgen.c
      test-gbmv/do_test_gbmv.c
      test-gbmv/gbmv-support.c
      test-gbmv2/BLAS_gbmv2_testgen.c
      test-gbmv2/do_test_gbmv2.c
      test-ge_sum_mv/BLAS_ge_sum_mv_testgen.c
      test-ge_sum_mv/do_test_ge_sum_mv.c
      test-gemm/BLAS_gemm_testgen.c
      test-gemm/do_test_gemm.c
      test-gemv/BLAS_gemv_testgen.c
      test-gemv/do_test_gemv.c
      test-gemv/gemv-support.c
      test-gemv2/BLAS_gemv2_testgen.c
      test-gemv2/do_test_gemv2.c
      test-hbmv/BLAS_hbmv_testgen.c
      test-hbmv/do_test_hbmv.c
      test-hbmv/hbmv-support.c
      test-hemm/BLAS_hemm_testgen.c
      test-hemm/do_test_hemm.c
      test-hemv/BLAS_hemv_testgen.c
      test-hemv/do_test_hemv.c
      test-hemv/hemv-support.c
      test-hemv2/BLAS_hemv2_testgen.c
      test-hemv2/do_test_hemv2.c
      test-hpmv/BLAS_hpmv_testgen.c
      test-hpmv/do_test_hpmv.c
      test-hpmv/hpmv-support.c
      test-sbmv/BLAS_sbmv_testgen.c
      test-sbmv/do_test_sbmv.c
      test-sbmv/sbmv-support.c
      test-spmv/BLAS_spmv_testgen.c
      test-spmv/do_test_spmv.c
      test-spmv/spmv-support.c
      test-sum/BLAS_sum_testgen.c
      test-sum/do_test_sum.c
      test-sum/sum-support.c
      test-symm/BLAS_symm_testgen.c
      test-symm/do_test_symm.c
      test-symv/BLAS_symv_testgen.c
      test-symv/do_test_symv.c
      test-symv/symv-support.c
      test-symv2/BLAS_symv2_testgen.c
      test-symv2/do_test_symv2.c
      test-tbsv/BLAS_tbsv_testgen.c
      test-tbsv/do_test_tbsv.c
      test-tbsv/tbsv-support.c
      test-tpmv/BLAS_tpmv_testgen.c
      test-tpmv/do_test_tpmv.c
      test-tpmv/tpmv-support.c
      test-trmv/BLAS_trmv_testgen.c
      test-trmv/do_test_trmv.c
      test-trmv/trmv-support.c
      test-trsv/BLAS_dot_testgen_trsv.c
      test-trsv/BLAS_trsv_testgen.c
      test-trsv/do_test_trsv.c
      test-trsv/testgen_BLAS_sdot_x.c
      test-trsv/trsv-support.c
      test-waxpby/do_test_waxpby.c

      blas_extended_test.h
  )

  target_include_directories(ctests PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )

  target_link_libraries(ctests PRIVATE xblas)

  set_target_properties(ctests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  message(STATUS "Combined ctests executable")
  target_compile_definitions(ctests PRIVATE XBLAS_COMBINE_TESTS)
else()
  add_test(
    NAME ctests_axpby
    COMMAND ctests_axpby 10 1 1.0 1 1
  )
  add_executable(ctests_axpby)
  target_sources(ctests_axpby
    PRIVATE
      test-axpby/do_test_axpby.c

      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c

      blas_extended_test.h
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
      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/do_test_dot.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-dot/testgen_aux.c

      blas_extended_test.h
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
      test-dot2/BLAS_dot2_testgen.c
      test-dot2/do_test_dot2.c
      test-dot2/dot2.c
      test-dot2/r_truth2.c
      test-dot2/test_dot2.c
      test-dot2/testgen_BLAS_cdot2.c
      test-dot2/testgen_BLAS_ddot2.c
      test-dot2/testgen_BLAS_sdot2.c
      test-dot2/testgen_BLAS_zdot2.c

      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_zdot.c

      blas_extended_test.h
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
      test-gbmv/BLAS_gbmv_testgen.c
      test-gbmv/do_test_gbmv.c
      test-gbmv/gbmv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c

      blas_extended_test.h
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
      test-gbmv2/BLAS_gbmv2_testgen.c
      test-gbmv2/do_test_gbmv2.c

      test-dot2/BLAS_dot2_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_zdot.c
      test-dot2/dot2.c
      test-dot2/r_truth2.c
      test-dot2/test_dot2.c
      test-dot2/testgen_BLAS_cdot2.c
      test-dot2/testgen_BLAS_ddot2.c
      test-dot2/testgen_BLAS_sdot2.c
      test-dot2/testgen_BLAS_zdot2.c
      test-gbmv/gbmv-support.c

      blas_extended_test.h
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
      test-ge_sum_mv/BLAS_ge_sum_mv_testgen.c
      test-ge_sum_mv/do_test_ge_sum_mv.c

      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-gemv/gemv-support.c

      blas_extended_test.h
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
      test-gemm/BLAS_gemm_testgen.c
      test-gemm/do_test_gemm.c

      test-dot/BLAS_dot_testgen.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-gemv/gemv-support.c

      blas_extended_test.h
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
      test-gemv/BLAS_gemv_testgen.c
      test-gemv/do_test_gemv.c
      test-gemv/gemv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c

      blas_extended_test.h
  )
  target_include_directories(ctests_gemv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_gemv PRIVATE xblas)
  set_target_properties(ctests_gemv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_gemv2
    COMMAND ctests_gemv2 12 1 1.0 0 0.1
  )
  add_executable(ctests_gemv2)
  target_sources(ctests_gemv2
    PRIVATE
      test-gemv2/BLAS_gemv2_testgen.c
      test-gemv2/do_test_gemv2.c

      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_zdot.c
      test-dot2/BLAS_dot2_testgen.c
      test-dot2/dot2.c
      test-dot2/r_truth2.c
      test-dot2/test_dot2.c
      test-dot2/testgen_BLAS_cdot2.c
      test-dot2/testgen_BLAS_ddot2.c
      test-dot2/testgen_BLAS_sdot2.c
      test-dot2/testgen_BLAS_zdot2.c
      test-gemv/gemv-support.c

      blas_extended_test.h
  )
  target_include_directories(ctests_gemv2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_gemv2 PRIVATE xblas)
  set_target_properties(ctests_gemv2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_hbmv
    COMMAND ctests_hbmv 7 1 1.0 3 0.1
  )
  add_executable(ctests_hbmv)
  target_sources(ctests_hbmv
    PRIVATE
      test-hbmv/BLAS_hbmv_testgen.c
      test-hbmv/do_test_hbmv.c
      test-hbmv/hbmv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-sbmv/BLAS_sbmv_testgen.c
      test-sbmv/sbmv-support.c

      blas_extended_test.h
  )
  target_include_directories(ctests_hbmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_hbmv PRIVATE xblas)
  set_target_properties(ctests_hbmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_hemm
    COMMAND ctests_hemm 7 1 1.0 0 0.1
  )
  add_executable(ctests_hemm)
  target_sources(ctests_hemm
    PRIVATE
      test-hemm/BLAS_hemm_testgen.c
      test-hemm/do_test_hemm.c

      test-dot/BLAS_dot_testgen.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-symm/BLAS_symm_testgen.c
      test-gemv/gemv-support.c
      test-hemv/hemv-support.c
      test-symv/symv-support.c

      blas_extended_test.h
  )
  target_include_directories(ctests_hemm PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_hemm PRIVATE xblas)
  set_target_properties(ctests_hemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_hemv
    COMMAND ctests_hemv 7 1 1.0 3 0.1
  )
  add_executable(ctests_hemv)
  target_sources(ctests_hemv
    PRIVATE
      test-hemv/BLAS_hemv_testgen.c
      test-hemv/do_test_hemv.c
      test-hemv/hemv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-symv/BLAS_symv_testgen.c
      test-symv/symv-support.c

      blas_extended_test.h
  )
  target_include_directories(ctests_hemv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_hemv PRIVATE xblas)
  set_target_properties(ctests_hemv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_hemv2
    COMMAND ctests_hemv2 7 1 1.0 3 1
  )
  add_executable(ctests_hemv2)
  target_sources(ctests_hemv2
    PRIVATE
      test-hemv2/BLAS_hemv2_testgen.c
      test-hemv2/do_test_hemv2.c

      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_zdot.c
      test-dot2/BLAS_dot2_testgen.c
      test-dot2/dot2.c
      test-dot2/r_truth2.c
      test-dot2/test_dot2.c
      test-dot2/testgen_BLAS_cdot2.c
      test-dot2/testgen_BLAS_ddot2.c
      test-dot2/testgen_BLAS_sdot2.c
      test-dot2/testgen_BLAS_zdot2.c
      test-hemv/hemv-support.c
      test-symv/symv-support.c
      test-symv2/BLAS_symv2_testgen.c

      blas_extended_test.h
  )
  target_include_directories(ctests_hemv2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_hemv2 PRIVATE xblas)
  set_target_properties(ctests_hemv2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_hpmv
    COMMAND ctests_hpmv 7 1 1.0 3 0.1
  )
  add_executable(ctests_hpmv)
  target_sources(ctests_hpmv
    PRIVATE
      test-hpmv/BLAS_hpmv_testgen.c
      test-hpmv/do_test_hpmv.c
      test-hpmv/hpmv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-hemv/BLAS_hemv_testgen.c
      test-symv/BLAS_symv_testgen.c
      test-hemv/hemv-support.c
      test-symv/symv-support.c

      blas_extended_test.h
  )
  target_include_directories(ctests_hpmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_hpmv PRIVATE xblas)
  set_target_properties(ctests_hpmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_sbmv
    COMMAND ctests_sbmv 7 1 1.0 3 0.01
  )
  add_executable(ctests_sbmv)
  target_sources(ctests_sbmv
    PRIVATE
      test-sbmv/BLAS_sbmv_testgen.c
      test-sbmv/do_test_sbmv.c
      test-sbmv/sbmv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c

      blas_extended_test.h
  )
  target_include_directories(ctests_sbmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_sbmv PRIVATE xblas)
  set_target_properties(ctests_sbmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_spmv
    COMMAND ctests_spmv 7 1 1.0 3 0.1
  )
  add_executable(ctests_spmv)
  target_sources(ctests_spmv
    PRIVATE
      test-spmv/BLAS_spmv_testgen.c
      test-spmv/do_test_spmv.c
      test-spmv/spmv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-symv/BLAS_symv_testgen.c
      test-symv/symv-support.c

      blas_extended_test.h
  )
  target_include_directories(ctests_spmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_spmv PRIVATE xblas)
  set_target_properties(ctests_spmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_sum
    COMMAND ctests_sum 20 1 1.0 0 1.0
  )
  add_executable(ctests_sum)
  target_sources(ctests_sum
    PRIVATE
      test-sum/BLAS_sum_testgen.c
      test-sum/do_test_sum.c
      test-sum/sum-support.c

      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c

      blas_extended_test.h
  )
  target_include_directories(ctests_sum PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_sum PRIVATE xblas)
  set_target_properties(ctests_sum PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_symm
    COMMAND ctests_symm 7 1 1.0 3 0.01
  )
  add_executable(ctests_symm)
  target_sources(ctests_symm
    PRIVATE
      test-symm/BLAS_symm_testgen.c
      test-symm/do_test_symm.c

      test-dot/BLAS_dot_testgen.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-gemv/gemv-support.c
      test-symv/symv-support.c

      blas_extended_test.h
  )
  target_include_directories(ctests_symm PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_symm PRIVATE xblas)
  set_target_properties(ctests_symm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_symv
    COMMAND ctests_symv 7 1 1.0 3 0.1
  )
  add_executable(ctests_symv)
  target_sources(ctests_symv
    PRIVATE
      test-symv/BLAS_symv_testgen.c
      test-symv/do_test_symv.c
      test-symv/symv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c

      blas_extended_test.h
  )
  target_include_directories(ctests_symv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_symv PRIVATE xblas)
  set_target_properties(ctests_symv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_symv2
    COMMAND ctests_symv2 7 1 1.0 0 1
  )
  add_executable(ctests_symv2)
  target_sources(ctests_symv2
    PRIVATE
      test-symv2/BLAS_symv2_testgen.c
      test-symv2/do_test_symv2.c

      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_zdot.c
      test-dot2/BLAS_dot2_testgen.c
      test-dot2/dot2.c
      test-dot2/r_truth2.c
      test-dot2/test_dot2.c
      test-dot2/testgen_BLAS_cdot2.c
      test-dot2/testgen_BLAS_ddot2.c
      test-dot2/testgen_BLAS_sdot2.c
      test-dot2/testgen_BLAS_zdot2.c
      test-symv/symv-support.c

      blas_extended_test.h
  )
  target_include_directories(ctests_symv2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_symv2 PRIVATE xblas)
  set_target_properties(ctests_symv2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_tbsv
    COMMAND ctests_tbsv 9 1 1.0 3 0.1
  )
  add_executable(ctests_tbsv)
  target_sources(ctests_tbsv
    PRIVATE
      test-tbsv/BLAS_tbsv_testgen.c
      test-tbsv/do_test_tbsv.c
      test-tbsv/tbsv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/copy_vector.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-gbmv/gbmv-support.c
      test-trsv/BLAS_dot_testgen_trsv.c
      test-trsv/testgen_BLAS_sdot_x.c

      blas_extended_test.h
  )
  target_include_directories(ctests_tbsv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_tbsv PRIVATE xblas)
  set_target_properties(ctests_tbsv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_tpmv
    COMMAND ctests_tpmv 12 1 1.0 0 0.1
  )
  add_executable(ctests_tpmv)
  target_sources(ctests_tpmv
    PRIVATE
      test-tpmv/BLAS_tpmv_testgen.c
      test-tpmv/do_test_tpmv.c
      test-tpmv/tpmv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c

      blas_extended_test.h
  )
  target_include_directories(ctests_tpmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_tpmv PRIVATE xblas)
  set_target_properties(ctests_tpmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_trmv
    COMMAND ctests_trmv 12 1 1.0 0 0.1
  )
  add_executable(ctests_trmv)
  target_sources(ctests_trmv
    PRIVATE
      test-trmv/BLAS_trmv_testgen.c
      test-trmv/do_test_trmv.c
      test-trmv/trmv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c

      blas_extended_test.h
  )
  target_include_directories(ctests_trmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_trmv PRIVATE xblas)
  set_target_properties(ctests_trmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_trsv
    COMMAND ctests_trsv 10 1 1.0 3 0.1
  )
  add_executable(ctests_trsv)
  target_sources(ctests_trsv
    PRIVATE
      test-trsv/BLAS_dot_testgen_trsv.c
      test-trsv/BLAS_trsv_testgen.c
      test-trsv/do_test_trsv.c
      test-trsv/testgen_BLAS_sdot_x.c
      test-trsv/trsv-support.c

      test-dot/BLAS_dot_testgen.c
      test-dot/print_vector.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c
      test-trmv/trmv-support.c

      blas_extended_test.h
  )
  target_include_directories(ctests_trsv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_trsv PRIVATE xblas)
  set_target_properties(ctests_trsv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME ctests_waxpby
    COMMAND ctests_waxpby 20 1 1.0 0 1
  )
  add_executable(ctests_waxpby)
  target_sources(ctests_waxpby
    PRIVATE
      test-waxpby/do_test_waxpby.c

      test-dot/BLAS_dot_testgen.c
      test-dot/test_dot.c
      test-dot/testgen_aux.c
      test-dot/testgen_BLAS_cdot.c
      test-dot/testgen_BLAS_ddot.c
      test-dot/testgen_BLAS_sdot.c
      test-dot/testgen_BLAS_zdot.c

      blas_extended_test.h
  )
  target_include_directories(ctests_waxpby PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/ctests
  )
  target_link_libraries(ctests_waxpby PRIVATE xblas)
  set_target_properties(ctests_waxpby PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  message(STATUS "Separate ctests executables")
endif()
