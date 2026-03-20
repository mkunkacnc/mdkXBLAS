#include "blas_extended.h"
#include "blas_extended_private.h"
#include "sum/XBLAS_sum.hpp"

void BLAS_ssum_x(int n,
                 const float *x,
                 int incx,
                 float *sum,
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
 * x     (input) const float*
 *       Array of length n.
 *
 * incx  (input) int
 *       The stride used to access components x[i].
 *
 * sum   (output) float*
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
//static const char *routine_name = "BLAS_ssum_x";
  XBLAS::sum_x(n, x, incx, sum, prec);
} /* end BLAS_ssum_x */
