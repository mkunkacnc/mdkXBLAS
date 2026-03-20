#include "blas_extended.h"
#include "blas_extended_private.h"
#include "ge_sum_mv/XBLAS_ge_sum_mv.hpp"

void BLAS_zge_sum_mv_d_z(enum blas_order_type order,
                         int m,
                         int n,
                         const void *alpha,
                         const double *a,
                         int lda,
                         const void *x,
                         int incx,
                         const void *beta,
                         const double *b,
                         int ldb,
                         void *y,
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
 * alpha  (input) const void*
 *
 * A      (input) const double*
 *
 * lda    (input) int
 *        Leading dimension of A
 *
 * x      (input) const void*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) const void*
 *
 * b      (input) const double*
 *
 * ldb    (input) int
 *        Leading dimension of B
 *
 * y      (input/output) void*
 *
 * incy   (input) int
 *        The stride for vector y.
 *
 */
{
//static const char *routine_name = "BLAS_zge_sum_mv_d_z";
  XBLAS::ge_sum_mv(order,
                   m,
                   n,
                   *static_cast<const std::complex<double> *>(alpha),
                   a,
                   lda,
                   static_cast<const std::complex<double> *>(x),
                   incx,
                   *static_cast<const std::complex<double> *>(beta),
                   b,
                   ldb,
                   static_cast<std::complex<double> *>(y),
                   incy);
} /* end BLAS_zge_sum_mv_d_z */
