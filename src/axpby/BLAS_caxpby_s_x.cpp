#include "blas_extended.h"
#include "axpby.hpp"

void BLAS_caxpby_s_x(int n,
                     const void *alpha,
                     const float *x,
                     int incx,
                     const void *beta,
                     void *y,
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
 * y      (input/output) void*
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
//static const char *routine_name = "BLAS_caxpby_s_x";
  XBLAS::axpby_x(n,
                 *static_cast<const std::complex<float> *>(alpha),
                 x,
                 incx,
                 *static_cast<const std::complex<float> *>(beta),
                 static_cast<std::complex<float> *>(y),
                 incy,
                 prec);
} /* end BLAS_caxpby_s_x */
