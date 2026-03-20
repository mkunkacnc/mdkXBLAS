#include "blas_extended.h"
#include "blas_extended_private.h"
#include "sbmv/XBLAS_sbmv.hpp"

void BLAS_zsbmv_d_d(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    int n,
                    int k,
                    const void *alpha,
                    const double *a,
                    int lda,
                    const double *x,
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
 * where A is a symmetric band matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Storage format of input symmetric matrix A.
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
 * a      (input) const double*
 *        Matrix A.
 *
 * lda    (input) int
 *        Leading dimension of matrix A.
 *
 * x      (input) const double*
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
 *  Notes on storing a symmetric band matrix:
 *
 *      Integers in the below arrays represent values of
 *              type double.
 *
 *    if we have a symettric matrix:
 *
 *      1  2  3  0  0
 *      2  4  5  6  0
 *      3  5  7  8  9
 *      0  6  8  10 11
 *      0  0  9  11 12
 *
 *     This matrix has n == 5, and k == 2. It can be stored in the
 *      following ways:
 *
 *      Notes for the examples:
 *      Each column below represents a contiguous vector.
 *      Columns are strided by lda.
 *      An asterisk (*) represents a position in the
 *       matrix that is not used.
 *      Note that the minimum lda (size of column) is 3 (k+1).
 *       lda may be arbitrarily large; an lda > 3 would mean
 *       there would be unused data at the bottom of the below
 *       columns.
 *
 *    blas_colmajor and blas_upper:
 *      *  *  3  6  9
 *      *  2  5  8  11
 *      1  4  7  10 12
 *
 *
 *    blas_colmajor and blas_lower
 *      1  4  7  10  12
 *      2  5  8  11  *
 *      3  6  9  *   *
 *
 *
 *    blas_rowmajor and blas_upper
 *      Columns here also represent contiguous arrays.
 *      1  4  7  10  12
 *      2  5  8  11  *
 *      3  6  9  *   *
 *
 *
 *    blas_rowmajor and blas_lower
 *      Columns here also represent contiguous arrays.
 *      *  *  3  6   9
 *      *  2  5  8   11
 *      1  4  7  10  12
 *
 */
{
//static const char *routine_name = "BLAS_zsbmv_d_d";
  XBLAS::sbmv(order,
              uplo,
              n,
              k,
              *static_cast<const std::complex<double> *>(alpha),
              a,
              lda,
              x,
              incx,
              *static_cast<const std::complex<double> *>(beta),
              static_cast<std::complex<double> *>(y),
              incy);
} /* end BLAS_zsbmv_d_d */
