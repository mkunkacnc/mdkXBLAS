#include <blas_extended.h>
#include <blas_extended_private.h>
#include <blas_fpu.h>
#include "symv2/XBLAS_symv2.hpp"

void BLAS_ssymv2_x(enum blas_order_type order,
                   enum blas_uplo_type uplo,
                   int n,
                   float alpha,
                   const float *a,
                   int lda,
                   const float *x_head,
                   const float *x_tail,
                   int incx,
                   float beta,
                   float *y,
                   int incy,
                   enum blas_prec_type prec)
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
 * alpha   (input) float
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
 * beta    (input) float
 *
 * y       (input/output) float*
 *         Vector y.
 *
 * incy    (input) int
 *         Stride for vector y.
 *
 * prec    (input) enum blas_prec_type
 *         Specifies the internal precision to be used.
 *         = blas_prec_single: single precision.
 *         = blas_prec_double: double precision.
 *         = blas_prec_extra : anything at least 1.5 times as accurate
 *                             than double, and wider than 80-bits.
 *                             We use double-double in our implementation.
 *
 */
{
//static const char routine_name[] = "BLAS_ssymv2_x";
  XBLAS::symv2_x(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy, prec);
} /* end BLAS_ssymv2_x */
