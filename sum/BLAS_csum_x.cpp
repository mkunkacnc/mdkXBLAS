#include "blas_extended.h"
#include "blas_extended_private.h"
#include "sum/XBLAS_sum.hpp"

void BLAS_csum_x(int n,
                 const void *x,
                 int incx,
                 void *sum,
                 enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine computes the summation:
 *
 *     sum <- SUM_{i=0, n-1} x[i].
 *
 * Arguments
 * =========
 *
 * n     (input) int
 *       The length of vector x.
 *
 * x     (input) const void*
 *       Array of length n.
 *
 * incx  (input) int
 *       The stride used to access components x[i].
 *
 * sum   (output) void*
 *
 * prec  (input) enum blas_prec_type
 *       Specifies the internal precision to be used.
 *       = blas_prec_single: single precision.
 *       = blas_prec_double: double precision.
 *       = blas_prec_extra : anything at least 1.5 times as accurate
 *                           than double, and wider than 80-bits.
 *                           We use double-double in our implementation.
 *
 */
{
//static const char routine_name[] = "BLAS_csum_x";
  XBLAS::sum_x(n,
               static_cast<const std::complex<float> *>(x),
               incx,
               static_cast<std::complex<float> *>(sum),
               prec);
} /* end BLAS_csum_x */
