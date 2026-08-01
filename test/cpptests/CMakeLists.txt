# ------------------------------------------------------------
# cpptests
# ------------------------------------------------------------

option(MDKXBLAS_COMBINE_TESTS "combine all tests into one executable" OFF)

if (MDKXBLAS_COMBINE_TESTS)
  add_test(
    NAME cpptests
    COMMAND cpptests
  )

  add_executable(cpptests)
  target_sources(cpptests
    PRIVATE
      common/dummy_main.cpp
      main.cpp
      test-axpby/do_test_axpby.cpp
      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/do_test_dot.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-dot/testgen_aux.cpp
      test-dot2/BLAS_dot2_testgen.cpp
      test-dot2/do_test_dot2.cpp
      test-dot2/dot2.cpp
      test-dot2/r_truth2.cpp
      test-dot2/test_dot2.cpp
      test-dot2/testgen_BLAS_cdot2.cpp
      test-dot2/testgen_BLAS_ddot2.cpp
      test-dot2/testgen_BLAS_sdot2.cpp
      test-dot2/testgen_BLAS_zdot2.cpp
      test-gbmv/BLAS_gbmv_testgen.cpp
      test-gbmv/do_test_gbmv.cpp
      test-gbmv/gbmv-support.cpp
      test-gbmv2/BLAS_gbmv2_testgen.cpp
      test-gbmv2/do_test_gbmv2.cpp
      test-ge_sum_mv/BLAS_ge_sum_mv_testgen.cpp
      test-ge_sum_mv/do_test_ge_sum_mv.cpp
      test-gemm/BLAS_gemm_testgen.cpp
      test-gemm/do_test_gemm.cpp
      test-gemv/BLAS_gemv_testgen.cpp
      test-gemv/do_test_gemv.cpp
      test-gemv/gemv-support.cpp
      test-gemv2/BLAS_gemv2_testgen.cpp
      test-gemv2/do_test_gemv2.cpp
      test-hbmv/BLAS_hbmv_testgen.cpp
      test-hbmv/do_test_hbmv.cpp
      test-hbmv/hbmv-support.cpp
      test-hemm/BLAS_hemm_testgen.cpp
      test-hemm/do_test_hemm.cpp
      test-hemv/BLAS_hemv_testgen.cpp
      test-hemv/do_test_hemv.cpp
      test-hemv/hemv-support.cpp
      test-hemv2/BLAS_hemv2_testgen.cpp
      test-hemv2/do_test_hemv2.cpp
      test-hpmv/BLAS_hpmv_testgen.cpp
      test-hpmv/do_test_hpmv.cpp
      test-hpmv/hpmv-support.cpp
      test-sbmv/BLAS_sbmv_testgen.cpp
      test-sbmv/do_test_sbmv.cpp
      test-sbmv/sbmv-support.cpp
      test-spmv/BLAS_spmv_testgen.cpp
      test-spmv/do_test_spmv.cpp
      test-spmv/spmv-support.cpp
      test-sum/BLAS_sum_testgen.cpp
      test-sum/do_test_sum.cpp
      test-sum/sum-support.cpp
      test-symm/BLAS_symm_testgen.cpp
      test-symm/do_test_symm.cpp
      test-symv/BLAS_symv_testgen.cpp
      test-symv/do_test_symv.cpp
      test-symv/symv-support.cpp
      test-symv2/BLAS_symv2_testgen.cpp
      test-symv2/do_test_symv2.cpp
      test-tbsv/BLAS_tbsv_testgen.cpp
      test-tbsv/do_test_tbsv.cpp
      test-tbsv/tbsv-support.cpp
      test-tpmv/BLAS_tpmv_testgen.cpp
      test-tpmv/do_test_tpmv.cpp
      test-tpmv/tpmv-support.cpp
      test-trmv/BLAS_trmv_testgen.cpp
      test-trmv/do_test_trmv.cpp
      test-trmv/trmv-support.cpp
      test-trsv/BLAS_dot_testgen_trsv.cpp
      test-trsv/BLAS_trsv_testgen.cpp
      test-trsv/do_test_trsv.cpp
      test-trsv/testgen_BLAS_sdot_x.cpp
      test-trsv/trsv-support.cpp
      test-waxpby/do_test_waxpby.cpp

      blas_extended_test.hpp
  )

  target_include_directories(cpptests PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )

  target_link_libraries(cpptests PRIVATE xblas)

  set_target_properties(cpptests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  message(STATUS "Combined cpptests executable")
  target_compile_definitions(cpptests PRIVATE XBLAS_COMBINE_TESTS)
else()
  add_test(
    NAME cpptests_axpby
    COMMAND cpptests_axpby 10 1 1.0 1 1
  )
  add_executable(cpptests_axpby)
  target_sources(cpptests_axpby
    PRIVATE
      test-axpby/do_test_axpby.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_axpby PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_axpby PRIVATE xblas)
  set_target_properties(cpptests_axpby PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_dot
    COMMAND cpptests_dot 20 1 1.0 0 1
  )
  add_executable(cpptests_dot)
  target_sources(cpptests_dot
    PRIVATE
      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/do_test_dot.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-dot/testgen_aux.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_dot PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_dot PRIVATE xblas)
  set_target_properties(cpptests_dot PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_dot2
    COMMAND cpptests_dot2 20 1 1.0 0 1
  )
  add_executable(cpptests_dot2)
  target_sources(cpptests_dot2
    PRIVATE
      test-dot2/BLAS_dot2_testgen.cpp
      test-dot2/do_test_dot2.cpp
      test-dot2/dot2.cpp
      test-dot2/r_truth2.cpp
      test-dot2/test_dot2.cpp
      test-dot2/testgen_BLAS_cdot2.cpp
      test-dot2/testgen_BLAS_ddot2.cpp
      test-dot2/testgen_BLAS_sdot2.cpp
      test-dot2/testgen_BLAS_zdot2.cpp

      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_zdot.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_dot2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_dot2 PRIVATE xblas)
  set_target_properties(cpptests_dot2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_gbmv
    COMMAND cpptests_gbmv 12 1 1.0 0 0.1
  )
  add_executable(cpptests_gbmv)
  target_sources(cpptests_gbmv
    PRIVATE
      test-gbmv/BLAS_gbmv_testgen.cpp
      test-gbmv/do_test_gbmv.cpp
      test-gbmv/gbmv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_gbmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_gbmv PRIVATE xblas)
  set_target_properties(cpptests_gbmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_gbmv2
    COMMAND cpptests_gbmv2 12 1 1.0 0 0.1
  )
  add_executable(cpptests_gbmv2)
  target_sources(cpptests_gbmv2
    PRIVATE
      test-gbmv2/BLAS_gbmv2_testgen.cpp
      test-gbmv2/do_test_gbmv2.cpp

      test-dot2/BLAS_dot2_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-dot2/dot2.cpp
      test-dot2/r_truth2.cpp
      test-dot2/test_dot2.cpp
      test-dot2/testgen_BLAS_cdot2.cpp
      test-dot2/testgen_BLAS_ddot2.cpp
      test-dot2/testgen_BLAS_sdot2.cpp
      test-dot2/testgen_BLAS_zdot2.cpp
      test-gbmv/gbmv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_gbmv2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_gbmv2 PRIVATE xblas)
  set_target_properties(cpptests_gbmv2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_ge_sum_mv
    COMMAND cpptests_ge_sum_mv 7 1 1.0 3 0.1
  )
  add_executable(cpptests_ge_sum_mv)
  target_sources(cpptests_ge_sum_mv
    PRIVATE
      test-ge_sum_mv/BLAS_ge_sum_mv_testgen.cpp
      test-ge_sum_mv/do_test_ge_sum_mv.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-gemv/gemv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_ge_sum_mv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_ge_sum_mv PRIVATE xblas)
  set_target_properties(cpptests_ge_sum_mv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_gemm
    COMMAND cpptests_gemm 9 1 1.0 0 0.01
  )
  add_executable(cpptests_gemm)
  target_sources(cpptests_gemm
    PRIVATE
      test-gemm/BLAS_gemm_testgen.cpp
      test-gemm/do_test_gemm.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-gemv/gemv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_gemm PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_gemm PRIVATE xblas)
  set_target_properties(cpptests_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_gemv
    COMMAND cpptests_gemv 12 1 1.0 0 0.1
  )
  add_executable(cpptests_gemv)
  target_sources(cpptests_gemv
    PRIVATE
      test-gemv/BLAS_gemv_testgen.cpp
      test-gemv/do_test_gemv.cpp
      test-gemv/gemv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_gemv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_gemv PRIVATE xblas)
  set_target_properties(cpptests_gemv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_gemv2
    COMMAND cpptests_gemv2 12 1 1.0 0 0.1
  )
  add_executable(cpptests_gemv2)
  target_sources(cpptests_gemv2
    PRIVATE
      test-gemv2/BLAS_gemv2_testgen.cpp
      test-gemv2/do_test_gemv2.cpp

      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-dot2/BLAS_dot2_testgen.cpp
      test-dot2/dot2.cpp
      test-dot2/r_truth2.cpp
      test-dot2/test_dot2.cpp
      test-dot2/testgen_BLAS_cdot2.cpp
      test-dot2/testgen_BLAS_ddot2.cpp
      test-dot2/testgen_BLAS_sdot2.cpp
      test-dot2/testgen_BLAS_zdot2.cpp
      test-gemv/gemv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_gemv2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_gemv2 PRIVATE xblas)
  set_target_properties(cpptests_gemv2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_hbmv
    COMMAND cpptests_hbmv 7 1 1.0 3 0.1
  )
  add_executable(cpptests_hbmv)
  target_sources(cpptests_hbmv
    PRIVATE
      test-hbmv/BLAS_hbmv_testgen.cpp
      test-hbmv/do_test_hbmv.cpp
      test-hbmv/hbmv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-sbmv/BLAS_sbmv_testgen.cpp
      test-sbmv/sbmv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_hbmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_hbmv PRIVATE xblas)
  set_target_properties(cpptests_hbmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_hemm
    COMMAND cpptests_hemm 7 1 1.0 0 0.1
  )
  add_executable(cpptests_hemm)
  target_sources(cpptests_hemm
    PRIVATE
      test-hemm/BLAS_hemm_testgen.cpp
      test-hemm/do_test_hemm.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-symm/BLAS_symm_testgen.cpp
      test-gemv/gemv-support.cpp
      test-hemv/hemv-support.cpp
      test-symv/symv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_hemm PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_hemm PRIVATE xblas)
  set_target_properties(cpptests_hemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_hemv
    COMMAND cpptests_hemv 7 1 1.0 3 0.1
  )
  add_executable(cpptests_hemv)
  target_sources(cpptests_hemv
    PRIVATE
      test-hemv/BLAS_hemv_testgen.cpp
      test-hemv/do_test_hemv.cpp
      test-hemv/hemv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-symv/BLAS_symv_testgen.cpp
      test-symv/symv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_hemv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_hemv PRIVATE xblas)
  set_target_properties(cpptests_hemv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_hemv2
    COMMAND cpptests_hemv2 7 1 1.0 3 1
  )
  add_executable(cpptests_hemv2)
  target_sources(cpptests_hemv2
    PRIVATE
      test-hemv2/BLAS_hemv2_testgen.cpp
      test-hemv2/do_test_hemv2.cpp

      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-dot2/BLAS_dot2_testgen.cpp
      test-dot2/dot2.cpp
      test-dot2/r_truth2.cpp
      test-dot2/test_dot2.cpp
      test-dot2/testgen_BLAS_cdot2.cpp
      test-dot2/testgen_BLAS_ddot2.cpp
      test-dot2/testgen_BLAS_sdot2.cpp
      test-dot2/testgen_BLAS_zdot2.cpp
      test-hemv/hemv-support.cpp
      test-symv/symv-support.cpp
      test-symv2/BLAS_symv2_testgen.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_hemv2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_hemv2 PRIVATE xblas)
  set_target_properties(cpptests_hemv2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_hpmv
    COMMAND cpptests_hpmv 7 1 1.0 3 0.1
  )
  add_executable(cpptests_hpmv)
  target_sources(cpptests_hpmv
    PRIVATE
      test-hpmv/BLAS_hpmv_testgen.cpp
      test-hpmv/do_test_hpmv.cpp
      test-hpmv/hpmv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-hemv/BLAS_hemv_testgen.cpp
      test-symv/BLAS_symv_testgen.cpp
      test-hemv/hemv-support.cpp
      test-symv/symv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_hpmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_hpmv PRIVATE xblas)
  set_target_properties(cpptests_hpmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_sbmv
    COMMAND cpptests_sbmv 7 1 1.0 3 0.01
  )
  add_executable(cpptests_sbmv)
  target_sources(cpptests_sbmv
    PRIVATE
      test-sbmv/BLAS_sbmv_testgen.cpp
      test-sbmv/do_test_sbmv.cpp
      test-sbmv/sbmv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_sbmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_sbmv PRIVATE xblas)
  set_target_properties(cpptests_sbmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_spmv
    COMMAND cpptests_spmv 7 1 1.0 3 0.1
  )
  add_executable(cpptests_spmv)
  target_sources(cpptests_spmv
    PRIVATE
      test-spmv/BLAS_spmv_testgen.cpp
      test-spmv/do_test_spmv.cpp
      test-spmv/spmv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-symv/BLAS_symv_testgen.cpp
      test-symv/symv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_spmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_spmv PRIVATE xblas)
  set_target_properties(cpptests_spmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_sum
    COMMAND cpptests_sum 20 1 1.0 0 1.0
  )
  add_executable(cpptests_sum)
  target_sources(cpptests_sum
    PRIVATE
      test-sum/BLAS_sum_testgen.cpp
      test-sum/do_test_sum.cpp
      test-sum/sum-support.cpp

      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_sum PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_sum PRIVATE xblas)
  set_target_properties(cpptests_sum PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_symm
    COMMAND cpptests_symm 7 1 1.0 3 0.01
  )
  add_executable(cpptests_symm)
  target_sources(cpptests_symm
    PRIVATE
      test-symm/BLAS_symm_testgen.cpp
      test-symm/do_test_symm.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-gemv/gemv-support.cpp
      test-symv/symv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_symm PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_symm PRIVATE xblas)
  set_target_properties(cpptests_symm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_symv
    COMMAND cpptests_symv 7 1 1.0 3 0.1
  )
  add_executable(cpptests_symv)
  target_sources(cpptests_symv
    PRIVATE
      test-symv/BLAS_symv_testgen.cpp
      test-symv/do_test_symv.cpp
      test-symv/symv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_symv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_symv PRIVATE xblas)
  set_target_properties(cpptests_symv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_symv2
    COMMAND cpptests_symv2 7 1 1.0 0 1
  )
  add_executable(cpptests_symv2)
  target_sources(cpptests_symv2
    PRIVATE
      test-symv2/BLAS_symv2_testgen.cpp
      test-symv2/do_test_symv2.cpp

      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-dot2/BLAS_dot2_testgen.cpp
      test-dot2/dot2.cpp
      test-dot2/r_truth2.cpp
      test-dot2/test_dot2.cpp
      test-dot2/testgen_BLAS_cdot2.cpp
      test-dot2/testgen_BLAS_ddot2.cpp
      test-dot2/testgen_BLAS_sdot2.cpp
      test-dot2/testgen_BLAS_zdot2.cpp
      test-symv/symv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_symv2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_symv2 PRIVATE xblas)
  set_target_properties(cpptests_symv2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_tbsv
    COMMAND cpptests_tbsv 9 1 1.0 3 0.1
  )
  add_executable(cpptests_tbsv)
  target_sources(cpptests_tbsv
    PRIVATE
      test-tbsv/BLAS_tbsv_testgen.cpp
      test-tbsv/do_test_tbsv.cpp
      test-tbsv/tbsv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/copy_vector.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-gbmv/gbmv-support.cpp
      test-trsv/BLAS_dot_testgen_trsv.cpp
      test-trsv/testgen_BLAS_sdot_x.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_tbsv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_tbsv PRIVATE xblas)
  set_target_properties(cpptests_tbsv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_tpmv
    COMMAND cpptests_tpmv 12 1 1.0 0 0.1
  )
  add_executable(cpptests_tpmv)
  target_sources(cpptests_tpmv
    PRIVATE
      test-tpmv/BLAS_tpmv_testgen.cpp
      test-tpmv/do_test_tpmv.cpp
      test-tpmv/tpmv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_tpmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_tpmv PRIVATE xblas)
  set_target_properties(cpptests_tpmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_trmv
    COMMAND cpptests_trmv 12 1 1.0 0 0.1
  )
  add_executable(cpptests_trmv)
  target_sources(cpptests_trmv
    PRIVATE
      test-trmv/BLAS_trmv_testgen.cpp
      test-trmv/do_test_trmv.cpp
      test-trmv/trmv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_trmv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_trmv PRIVATE xblas)
  set_target_properties(cpptests_trmv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_trsv
    COMMAND cpptests_trsv 10 1 1.0 3 0.1
  )
  add_executable(cpptests_trsv)
  target_sources(cpptests_trsv
    PRIVATE
      test-trsv/BLAS_dot_testgen_trsv.cpp
      test-trsv/BLAS_trsv_testgen.cpp
      test-trsv/do_test_trsv.cpp
      test-trsv/testgen_BLAS_sdot_x.cpp
      test-trsv/trsv-support.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/print_vector.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp
      test-trmv/trmv-support.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_trsv PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_trsv PRIVATE xblas)
  set_target_properties(cpptests_trsv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  add_test(
    NAME cpptests_waxpby
    COMMAND cpptests_waxpby 20 1 1.0 0 1
  )
  add_executable(cpptests_waxpby)
  target_sources(cpptests_waxpby
    PRIVATE
      test-waxpby/do_test_waxpby.cpp

      test-dot/BLAS_dot_testgen.cpp
      test-dot/test_dot.cpp
      test-dot/testgen_aux.cpp
      test-dot/testgen_BLAS_cdot.cpp
      test-dot/testgen_BLAS_ddot.cpp
      test-dot/testgen_BLAS_sdot.cpp
      test-dot/testgen_BLAS_zdot.cpp

      blas_extended_test.hpp
  )
  target_include_directories(cpptests_waxpby PRIVATE
    ${CMAKE_SOURCE_DIR}/include/XBLAS
    ${CMAKE_SOURCE_DIR}/test/cpptests
  )
  target_link_libraries(cpptests_waxpby PRIVATE xblas)
  set_target_properties(cpptests_waxpby PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
  )

  message(STATUS "Separate cpptests executables")
endif()
