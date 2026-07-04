#include "blas_extended.h"
#include "ge_sum_mv.hpp"

void BLAS_cge_sum_mv_s_s(enum blas_order_type order,
                         int m,
                         int n,
                         const void *alpha,
                         const float *a,
                         int lda,
                         const float *x,
                         int incx,
                         const void *beta,
                         const float *b,
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
 * A      (input) const float*
 *
 * lda    (input) int
 *        Leading dimension of A
 *
 * x      (input) const float*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) const void*
 *
 * b      (input) const float*
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
//static const char *routine_name = "BLAS_cge_sum_mv_s_s";
  XBLAS::ge_sum_mv(order,
                   m,
                   n,
                   *static_cast<const std::complex<float> *>(alpha),
                   a,
                   lda,
                   x,
                   incx,
                   *static_cast<const std::complex<float> *>(beta),
                   b,
                   ldb,
                   static_cast<std::complex<float> *>(y),
                   incy);
} /* end BLAS_cge_sum_mv_s_s */
