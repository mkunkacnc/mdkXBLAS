#include "blas_extended.h"
#include "blas_extended_private.h"
#include "dot/XBLAS_dot.hpp"

void BLAS_zdot_z_d(enum blas_conj_type conj,
                   int n,
                   const void *alpha,
                   const void *x,
                   int incx,
                   const void *beta,
                   const double *y,
                   int incy,
                   void *r)
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
 *        components x[i] are used unconjugated or conjugated.
 *
 * n      (input) int
 *        The length of vectors x and y.
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
 * y      (input) const double*
 *        Array of length n.
 *
 * incy   (input) int
 *        The stride used to access components y[i].
 *
 * r      (input/output) void*
 *
 */
{
//static const char routine_name[] = "BLAS_zdot_z_d";
  XBLAS::dot(conj,
             n,
             *static_cast<const std::complex<double> *>(alpha),
             static_cast<const std::complex<double> *>(x),
             incx,
             *static_cast<const std::complex<double> *>(beta),
             y,
             incy,
             static_cast<std::complex<double> *>(r));
} /* end BLAS_zdot_z_d */
