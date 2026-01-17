#include "blas_extended.h"
#include "blas_extended_private.h"
#include "axpby/XBLAS_axpby.hpp"

void BLAS_caxpby_s(int n,
                   const void *alpha,
                   const float *x,
                   int incx,
                   const void *beta,
                   void *y,
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
 * n         (input) int
 *           The length of vectors x and y.
 *
 * alpha     (input) const void*
 *
 * x         (input) const float*
 *           Array of length n.
 *
 * incx      (input) int
 *           The stride used to access components x[i].
 *
 * beta      (input) const void*
 *
 * y         (input) void*
 *           Array of length n.
 *
 * incy      (input) int
 *           The stride used to access components y[i].
 *
 */
{
//static const char routine_name[] = "BLAS_caxpby_s";
  XBLAS::axpby(n,
               *static_cast<const std::complex<float> *>(alpha),
               x,
               incx,
               *static_cast<const std::complex<float> *>(beta),
               static_cast<std::complex<float> *>(y),
               incy);
}
