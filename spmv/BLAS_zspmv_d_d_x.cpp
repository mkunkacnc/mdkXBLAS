#include "blas_extended.h"
#include "blas_extended_private.h"
#include "spmv/XBLAS_spmv.hpp"

void BLAS_zspmv_d_d_x(enum blas_order_type order,
                      enum blas_uplo_type uplo,
                      int n,
                      const void *alpha,
                      const double *ap,
                      const double *x,
                      int incx,
                      const void *beta,
                      void *y,
                      int incy,
                      enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * ap * x + beta * y, where ap is a symmetric
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Order of ap; row or column major
 *
 * uplo   (input) enum blas_uplo_type
 *        Whether ap is upper or lower
 *
 * n      (input) int
 *        Dimension of ap and the length of vector x
 *
 * alpha  (input) const void*
 *
 * ap     (input) const double*
 *
 * x      (input) const double*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) const void*
 *
 * y      (input/output) void*
 *
 * incy   (input) int
 *        The stride for vector y.
 * prec   (input) enum blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 *
 */
{
//static const char *routine_name = "BLAS_zspmv_d_d_x";
  XBLAS::spmv_x(order,
                uplo,
                n,
                *static_cast<const std::complex<double> *>(alpha),
                ap,
                x,
                incx,
                *static_cast<const std::complex<double> *>(beta),
                static_cast<std::complex<double> *>(y),
                incy,
                prec);
} /* end BLAS_zspmv_d_d_x */
