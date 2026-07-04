#include "blas_extended.h"
#include "waxpby.hpp"

void BLAS_dwaxpby_s_s_x(int n,
                        double alpha,
                        const float *x,
                        int incx,
                        double beta,
                        const float *y,
                        int incy,
                        double *w,
                        int incw,
                        enum blas_prec_type prec)
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
 * y      (input) const float*
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
//static const char *routine_name = "BLAS_dwaxpby_s_s_x";
  XBLAS::waxpby_x(n, alpha, x, incx, beta, y, incy, w, incw, prec);
} /* end BLAS_dwaxpby_s_s_x */
