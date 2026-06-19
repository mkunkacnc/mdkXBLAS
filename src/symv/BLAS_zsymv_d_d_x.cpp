#include "blas_extended.h"
#include "blas_extended_private.h"
#include "symv.hpp"

void BLAS_zsymv_d_d_x(enum blas_order_type order,
                      enum blas_uplo_type uplo,
                      int n,
                      const void *alpha,
                      const double *a,
                      int lda,
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
 * This routines computes the matrix product:
 *
 *     y  <-  alpha * A * x  +  beta * y
 *
 * where A is a Symmetric matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Storage format of input symmetric matrix A.
 *
 * uplo   (input) enum blas_uplo_type
 *        Determines which half of matrix A (upper or lower triangle)
 *          is accessed.
 *
 * n      (input) int
 *        Dimension of A and size of vectors x, y.
 *
 * alpha  (input) const void*
 *
 * a      (input) const double*
 *        Matrix A.
 *
 * lda    (input) int
 *        Leading dimension of matrix A.
 *
 * x      (input) const double*
 *        Vector x.
 *
 * incx   (input) int
 *        Stride for vector x.
 *
 * beta   (input) const void*
 *
 * y      (input/output) void*
 *        Vector y.
 *
 * incy   (input) int
 *        Stride for vector y.
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
//static const char *routine_name = "BLAS_zsymv_d_d_x";
  XBLAS::symv_x(order,
                uplo,
                n,
                *static_cast<const std::complex<double> *>(alpha),
                a,
                lda,
                x,
                incx,
                *static_cast<const std::complex<double> *>(beta),
                static_cast<std::complex<double> *>(y),
                incy,
                prec);
} /* end BLAS_zsymv_d_d_x */
