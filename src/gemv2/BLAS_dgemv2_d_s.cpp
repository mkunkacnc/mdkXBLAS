#include "blas_extended.h"
#include "blas_extended_private.h"
#include "gemv2.hpp"

void BLAS_dgemv2_d_s(enum blas_order_type order,
                     enum blas_trans_type trans,
                     int m,
                     int n,
                     double alpha,
                     const double *a,
                     int lda,
                     const float *head_x,
                     const float *tail_x,
                     int incx,
                     double beta,
                     double *y,
                     int incy)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * op(A) * head_x + alpha * op(A) * tail_x + beta * y,
 * where A is a general matrix.
 *
 * Arguments
 * =========
 *
 * order   (input) enum blas_order_type
 *         Order of A; row or column major
 *
 * trans   (input) enum blas_trans_type
 *         Transpose of A: no trans, trans, or conjugate trans
 *
 * m       (input) int
 *         Dimension of A
 *
 * n       (input) int
 *         Dimension of A and the length of vector x and z
 *
 * alpha   (input) double
 *
 * A       (input) const double*
 *
 * lda     (input) int
 *         Leading dimension of A
 *
 * head_x
 * tail_x  (input) const float*
 *
 * incx    (input) int
 *         The stride for vector x.
 *
 * beta    (input) double
 *
 * y       (input/output) double*
 *
 * incy    (input) int
 *         The stride for vector y.
 *
 */
{
//static const char *routine_name = "BLAS_dgemv2_d_s";
  XBLAS::gemv2(order, trans, m, n, alpha, a, lda, head_x, tail_x, incx, beta, y, incy);
} /* end BLAS_dgemv2_d_s */
