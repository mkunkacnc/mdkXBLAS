#include "blas_extended.h"
#include "blas_extended_private.h"
#include "axpby/XBLAS_axpby.hpp"

void BLAS_daxpby_s(int n,
                   double alpha,
                   const float *x,
                   int incx,
                   double beta,
                   double *y,
                   int incy)
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
 */
{
  //static const char *routine_name = "BLAS_daxpby_s";
  XBLAS::axpby(n, alpha, x, incx, beta, y, incy);
} /* end BLAS_daxpby_s */
