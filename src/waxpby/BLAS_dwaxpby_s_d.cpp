#include "blas_extended.h"
#include "waxpby.hpp"

void BLAS_dwaxpby_s_d(int n,
                      double alpha,
                      const float *x,
                      int incx,
                      double beta,
                      const double *y,
                      int incy,
                      double *w,
                      int incw)
/*
 * Purpose
 * =======
 *
 * This routine computes:
 *
 *     w <- alpha * x + beta * y
 *
 * Arguments
 * =========
 *
 * n      (input) int
 *        The length of vectors x, y, and w.
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
 * y      (input) const double*
 *        Array of length n.
 *
 * incy   (input) int
 *        The stride used to access components y[i].
 *
 * w      (output) double*
 *        Array of length n.
 *
 * incw   (input) int
 *        The stride used to write components w[i].
 *
 */
{
//static const char *routine_name = "BLAS_dwaxpby_s_d";
  XBLAS::waxpby(n, alpha, x, incx, beta, y, incy, w, incw);
} /* end BLAS_dwaxpby_s_d */
