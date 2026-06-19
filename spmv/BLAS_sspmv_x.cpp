#include "blas_extended.h"
#include "blas_extended_private.h"
#include "spmv.hpp"

void BLAS_sspmv_x(enum blas_order_type order,
                  enum blas_uplo_type uplo,
                  int n,
                  float alpha,
                  const float *ap,
                  const float *x,
                  int incx,
                  float beta,
                  float *y,
                  int incy,
                  enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * ap * x + beta * y, where ap is a symmetric
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Order of ap; row or column major
 *
 * uplo   (input) enum blas_uplo_type
 *        Whether ap is upper or lower
 *
 * n      (input) int
 *        Dimension of ap and the length of vector x
 *
 * alpha  (input) float
 *
 * ap     (input) const float*
 *
 * x      (input) const float*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) float
 *
 * y      (input/output) float*
 *
 * incy   (input) int
 *        The stride for vector y.
 * prec   (input) enum blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 *
 */
{
//static const char *routine_name = "BLAS_sspmv_x";
  XBLAS::spmv_x(order, uplo, n, alpha, ap, x, incx, beta, y, incy, prec);
} /* end BLAS_sspmv_x */
