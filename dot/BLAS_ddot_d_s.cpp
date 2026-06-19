#include "blas_extended.h"
#include "blas_extended_private.h"
#include "dot.hpp"

void BLAS_ddot_d_s(enum blas_conj_type conj,
                   int n,
                   double alpha,
                   const double *x,
                   int incx,
                   double beta,
                   const float *y,
                   int incy,
                   double *r)
/*
 * Purpose
 * =======
 *
 * This routine computes the inner product:
 *
 *     r <- beta * r + alpha * SUM_{i=0, n-1} x[i] * y[i].
 *
 * Arguments
 * =========
 *
 * conj   (input) enum blas_conj_type
 *        When x and y are complex vectors, specifies whether vector
 *          components x[i] are used unconjugated or conjugated.
 *
 * n      (input) int
 *        The length of vectors x and y.
 *
 * alpha  (input) double
 *
 * x      (input) const double*
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
 * r      (input/output) double*
 *
 */
{
//static const char *routine_name = "BLAS_ddot_d_s";
  XBLAS::dot(conj, n, alpha, x, incx, beta, y, incy, r);
} /* end BLAS_ddot_d_s */
