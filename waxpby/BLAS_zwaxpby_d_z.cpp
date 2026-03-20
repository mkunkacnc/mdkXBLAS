#include "blas_extended.h"
#include "blas_extended_private.h"
#include "waxpby/XBLAS_waxpby.hpp"

void BLAS_zwaxpby_d_z(int n,
                      const void *alpha,
                      const double *x,
                      int incx,
                      const void *beta,
                      const void *y,
                      int incy,
                      void *w,
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
 * alpha  (input) const void*
 *
 * x      (input) const double*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 * beta   (input) const void*
 *
 * y      (input) const void*
 *        Array of length n.
 *
 * incy   (input) int
 *        The stride used to access components y[i].
 *
 * w      (output) void*
 *        Array of length n.
 *
 * incw   (input) int
 *        The stride used to write components w[i].
 *
 */
{
//static const char *routine_name = "BLAS_zwaxpby_d_z";
  XBLAS::waxpby(n,
                *static_cast<const std::complex<double> *>(alpha),
                x,
                incx,
                *static_cast<const std::complex<double> *>(beta),
                static_cast<const std::complex<double> *>(y),
                incy,
                static_cast<std::complex<double> *>(w),
                incw);
} /* end BLAS_zwaxpby_d_z */
