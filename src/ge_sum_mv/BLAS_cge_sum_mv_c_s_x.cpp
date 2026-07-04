#include "blas_extended.h"
#include "ge_sum_mv.hpp"

void BLAS_cge_sum_mv_c_s_x(enum blas_order_type order,
                           int m,
                           int n,
                           const void *alpha,
                           const void *a,
                           int lda,
                           const float *x,
                           int incx,
                           const void *beta,
                           const void *b,
                           int ldb,
                           void *y,
                           int incy,
                           enum blas_prec_type prec)
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
 * x      (input) const float*
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
 * prec   (input) enum blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 */
{
//static const char *routine_name = "BLAS_cge_sum_mv_c_s_x";
  XBLAS::ge_sum_mv_x(order,
                     m,
                     n,
                     *static_cast<const std::complex<float> *>(alpha),
                     static_cast<const std::complex<float> *>(a),
                     lda,
                     x,
                     incx,
                     *static_cast<const std::complex<float> *>(beta),
                     static_cast<const std::complex<float> *>(b),
                     ldb,
                     static_cast<std::complex<float> *>(y),
                     incy,
                     prec);
} /* end BLAS_cge_sum_mv_c_s_x */
