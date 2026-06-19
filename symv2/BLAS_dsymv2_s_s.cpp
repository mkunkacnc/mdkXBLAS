#include <blas_extended.h>
#include <blas_extended_private.h>
#include <blas_fpu.h>
#include "symv2.hpp"

void BLAS_dsymv2_s_s(enum blas_order_type order,
                     enum blas_uplo_type uplo,
                     int n,
                     double alpha,
                     const float *a,
                     int lda,
                     const float *x_head,
                     const float *x_tail,
                     int incx,
                     double beta,
                     double *y,
                     int incy)
/*
 * Purpose
 * =======
 *
 * This routines computes the matrix product:
 *
 *     y  <-  alpha * A * (x_head + x_tail) + beta * y
 *
 * where A is a symmetric matrix.
 *
 * Arguments
 * =========
 *
 * order   (input) enum blas_order_type
 *         Storage format of input symmetric matrix A.
 *
 * uplo    (input) enum blas_uplo_type
 *         Determines which half of matrix A (upper or lower triangle)
 *           is accessed.
 *
 * n       (input) int
 *         Dimension of A and size of vectors x, y.
 *
 * alpha   (input) double
 *
 * a       (input) const float*
 *         Matrix A.
 *
 * lda     (input) int
 *         Leading dimension of matrix A.
 *
 * x_head  (input) const float*
 *         Vector x_head
 *
 * x_tail  (input) const float*
 *         Vector x_tail
 *
 * incx    (input) int
 *         Stride for vector x.
 *
 * beta    (input) double
 *
 * y       (input/output) double*
 *         Vector y.
 *
 * incy    (input) int
 *         Stride for vector y.
 *
 */
{
//static const char *routine_name = "BLAS_dsymv2_s_s";
  XBLAS::symv2(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
} /* end BLAS_dsymv2_s_s */
