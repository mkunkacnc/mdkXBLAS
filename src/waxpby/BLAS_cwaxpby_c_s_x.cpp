#include "blas_extended.h"
#include "waxpby.hpp"

void BLAS_cwaxpby_c_s_x(int n,
                        const void *alpha,
                        const void *x,
                        int incx,
                        const void *beta,
                        const float *y,
                        int incy,
                        void *w,
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
 * alpha  (input) const void*
 *
 * x      (input) const void*
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
//static const char *routine_name = "BLAS_cwaxpby_c_s_x";
  XBLAS::waxpby_x(n,
                  *static_cast<const std::complex<float> *>(alpha),
                  static_cast<const std::complex<float> *>(x),
                  incx,
                  *static_cast<const std::complex<float> *>(beta),
                  y,
                  incy,
                  static_cast<std::complex<float> *>(w),
                  incw,
                  prec);
} /* end BLAS_cwaxpby_c_s_x */
