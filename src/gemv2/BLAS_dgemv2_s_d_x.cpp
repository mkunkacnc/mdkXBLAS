#include "blas_extended.h"
#include "gemv2.hpp"

void BLAS_dgemv2_s_d_x(enum blas_order_type order,
                       enum blas_trans_type trans,
                       int m,
                       int n,
                       double alpha,
                       const float *a,
                       int lda,
                       const double *head_x,
                       const double *tail_x,
                       int incx,
                       double beta,
                       double *y,
                       int incy,
                       enum blas_prec_type prec)
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
 * A       (input) const float*
 *
 * lda     (input) int
 *         Leading dimension of A
 *
 * head_x
 * tail_x  (input) const double*
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
//static const char *routine_name = "BLAS_dgemv2_s_d_x";
  XBLAS::gemv2_x(order, trans, m, n, alpha, a, lda, head_x, tail_x, incx, beta, y, incy, prec);
} /* end BLAS_dgemv2_s_d_x */
