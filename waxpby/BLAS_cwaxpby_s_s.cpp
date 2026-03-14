#include "blas_extended.h"
#include "blas_extended_private.h"
#include "waxpby/XBLAS_waxpby.hpp"

void BLAS_cwaxpby_s_s(int n,
                      const void *alpha,
                      const float *x,
                      int incx,
                      const void *beta,
                      const float *y,
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
 * x      (input) const float*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 * beta   (input) const void*
 *
 * y      (input) const float*
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
//static const char routine_name[] = "BLAS_cwaxpby_s_s";
  XBLAS::waxpby(n,
                *static_cast<const std::complex<float> *>(alpha),
                x,
                incx,
                *static_cast<const std::complex<float> *>(beta),
                y,
                incy,
                static_cast<std::complex<float> *>(w),
                incw);
} /* end BLAS_cwaxpby_s_s */
