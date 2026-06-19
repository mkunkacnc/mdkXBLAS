#include "blas_extended.h"
#include "blas_extended_private.h"
#include "ge_sum_mv.hpp"

void BLAS_zge_sum_mv_z_d(enum blas_order_type order,
                         int m,
                         int n,
                         const void *alpha,
                         const void *a,
                         int lda,
                         const double *x,
                         int incx,
                         const void *beta,
                         const void *b,
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
 * A      (input) const void*
 *
 * lda    (input) int
 *        Leading dimension of A
 *
 * x      (input) const double*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) const void*
 *
 * b      (input) const void*
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
//static const char *routine_name = "BLAS_zge_sum_mv_z_d";
  XBLAS::ge_sum_mv(order,
                   m,
                   n,
                   *static_cast<const std::complex<double> *>(alpha),
                   static_cast<const std::complex<double> *>(a),
                   lda,
                   x,
                   incx,
                   *static_cast<const std::complex<double> *>(beta),
                   static_cast<const std::complex<double> *>(b),
                   ldb,
                   static_cast<std::complex<double> *>(y),
                   incy);
} /* end BLAS_zge_sum_mv_z_d */
