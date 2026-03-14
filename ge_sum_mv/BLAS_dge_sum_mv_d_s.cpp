#include "blas_extended.h"
#include "blas_extended_private.h"
#include "ge_sum_mv/XBLAS_ge_sum_mv.hpp"

void BLAS_dge_sum_mv_d_s(enum blas_order_type order,
                         int m,
                         int n,
                         double alpha,
                         const double *a,
                         int lda,
                         const float *x,
                         int incx,
                         double beta,
                         const double *b,
                         int ldb,
                         double *y,
                         int incy)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * A * x + beta * B * y,
 *     where A, B are general matrices.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Order of A; row or column major
 *
 * m      (input) int
 *        Row Dimension of A, B, length of output vector y
 *
 * n      (input) int
 *        Column Dimension of A, B and the length of vector x
 *
 * alpha  (input) double
 *
 * A      (input) const double*
 *
 * lda    (input) int
 *        Leading dimension of A
 *
 * x      (input) const float*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) double
 *
 * b      (input) const double*
 *
 * ldb    (input) int
 *        Leading dimension of B
 *
 * y      (input/output) double*
 *
 * incy   (input) int
 *        The stride for vector y.
 *
 */
{
//static const char routine_name[] = "BLAS_dge_sum_mv_d_s";
  XBLAS::ge_sum_mv(order, m, n, alpha, a, lda, x, incx, beta, b, ldb, y, incy);
} /* end BLAS_dge_sum_mv_d_s */
