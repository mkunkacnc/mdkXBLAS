#include "blas_extended.h"
#include "blas_extended_private.h"
#include "hbmv.hpp"

void BLAS_chbmv_c_s(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    int n,
                    int k,
                    const void *alpha,
                    const void *a,
                    int lda,
                    const float *x,
                    int incx,
                    const void *beta,
                    void *y,
                    int incy)
/*
 * Purpose
 * =======
 *
 * This routines computes the matrix product:
 *
 *     y  <-  alpha * A * x  +  beta * y
 *
 * where A is a hermitian band matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Storage format of input hermitian matrix A.
 *
 * uplo   (input) enum blas_uplo_type
 *        Determines which half of matrix A (upper or lower triangle)
 *          is accessed.
 *
 * n      (input) int
 *        Dimension of A and size of vectors x, y.
 *
 * k      (input) int
 *        Number of subdiagonals ( = number of superdiagonals)
 *
 * alpha  (input) const void*
 *
 * a      (input) const void*
 *        Matrix A.
 *
 * lda    (input) int
 *        Leading dimension of matrix A.
 *
 * x      (input) const float*
 *        Vector x.
 *
 * incx   (input) int
 *        Stride for vector x.
 *
 * beta   (input) const void*
 *
 * y      (input/output) void*
 *        Vector y.
 *
 * incy   (input) int
 *        Stride for vector y.
 *
 *
 *  Notes on storing a hermitian band matrix:
 *
 *      Integers in the below arrays represent values of
 *              type complex float.
 *
 *    if we have a hermitian matrix:
 *
 *      1d  2   3   0   0
 *      2#  4d  5   6   0
 *      3#  5#  7d  8   9
 *      0   6#  8#  10d 11
 *      0   0   9#  11# 12d
 *
 *     This matrix has n == 5, and k == 2. It can be stored in the
 *      following ways:
 *
 *      Notes for the examples:
 *      Each column below represents a contiguous vector.
 *      Columns are strided by lda.
 *      An asterisk (*) represents a position in the
 *       matrix that is not used.
 *      A pound sign (#) represents the conjugated form is stored
 *      A d following an integer indicates that the imaginary
 *       part of the number is assumed to be zero.
 *      Note that the minimum lda (size of column) is 3 (k+1).
 *       lda may be arbitrarily large; an lda > 3 would mean
 *       there would be unused data at the bottom of the below
 *       columns.
 *
 *    blas_colmajor and blas_upper:
 *      *   *   3   6   9
 *      *   2   5   8   11
 *      1d  4d  7d  10d 12d
 *
 *
 *    blas_colmajor and blas_lower
 *      1d   4d   7d   10d  12d
 *      2#   5#   8#   11#  *
 *      3#   6#   9#   *    *
 *
 *
 *    blas_rowmajor and blas_upper
 *      Columns here also represent contiguous arrays.
 *      1d  4d  7d  10d  12d
 *      2   5   8   11   *
 *      3   6   9   *    *
 *
 *
 *    blas_rowmajor and blas_lower
 *      Columns here also represent contiguous arrays.
 *      *   *   3#  6#   9#
 *      *   2#  5#  8#   11#
 *      1d  4d  7d  10d  12d
 *
 */
{
//static const char *routine_name = "BLAS_chbmv_c_s";
  XBLAS::hbmv(order,
              uplo,
              n,
              k,
              *static_cast<const std::complex<float> *>(alpha),
              static_cast<const std::complex<float> *>(a),
              lda,
              x,
              incx,
              *static_cast<const std::complex<float> *>(beta),
              static_cast<std::complex<float> *>(y),
              incy);
} /* end BLAS_chbmv_c_s */
