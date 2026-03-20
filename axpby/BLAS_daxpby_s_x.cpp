#include "blas_extended.h"
#include "blas_extended_private.h"
#include "axpby/XBLAS_axpby.hpp"

void BLAS_daxpby_s_x(int n,
                     double alpha,
                     const float *x,
                     int incx,
                     double beta,
                     double *y,
                     int incy,
                     enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine computes:
 *
 *      y <- alpha * x + beta * y.
 *
 * Arguments
 * =========
 *
 * n      (input) int
 *        The length of vectors x and y.
 *
 * alpha  (input) double
 *
 * x      (input) const float*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 * beta   (input) double
 *
 * y      (input/output) double*
 *        Array of length n.
 *
 * incy   (input) int
 *        The stride used to access components y[i].
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
  //static const char *routine_name = "BLAS_daxpby_s_x";
  XBLAS::axpby_x(n, alpha, x, incx, beta, y, incy, prec);
} /* end BLAS_daxpby_s_x */
