#include <blas_extended.h>
#include <blas_extended_private.h>
#include <blas_fpu.h>
#include "symv2.hpp"

void BLAS_zsymv2_z_d_x(enum blas_order_type order,
                       enum blas_uplo_type uplo,
                       int n,
                       const void *alpha,
                       const void *a,
                       int lda,
                       const double *x_head,
                       const double *x_tail,
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
 *     y  <-  alpha * A * (x_head + x_tail) + beta * y
 *
 * where A is a symmetric matrix.
 *
 * Arguments
 * =========
 *
 * order   (input) enum blas_order_type
 *         Storage format of input symmetric matrix A.
 *
 * uplo    (input) enum blas_uplo_type
 *         Determines which half of matrix A (upper or lower triangle)
 *           is accessed.
 *
 * n       (input) int
 *         Dimension of A and size of vectors x, y.
 *
 * alpha   (input) const void*
 *
 * a       (input) const void*
 *         Matrix A.
 *
 * lda     (input) int
 *         Leading dimension of matrix A.
 *
 * x_head  (input) const double*
 *         Vector x_head
 *
 * x_tail  (input) const double*
 *         Vector x_tail
 *
 * incx    (input) int
 *         Stride for vector x.
 *
 * beta    (input) const void*
 *
 * y       (input/output) void*
 *         Vector y.
 *
 * incy    (input) int
 *         Stride for vector y.
 *
 * prec    (input) enum blas_prec_type
 *         Specifies the internal precision to be used.
 *         = blas_prec_single: single precision.
 *         = blas_prec_double: double precision.
 *         = blas_prec_extra : anything at least 1.5 times as accurate
 *                             than double, and wider than 80-bits.
 *                             We use double-double in our implementation.
 *
 */
{
//static const char *routine_name = "BLAS_zsymv2_z_d_x";
  XBLAS::symv2_x(order,
                 uplo,
                 n,
                 *static_cast<const std::complex<double> *>(alpha),
                 static_cast<const std::complex<double> *>(a),
                 lda,
                 x_head,
                 x_tail,
                 incx,
                 *static_cast<const std::complex<double> *>(beta),
                 static_cast<std::complex<double> *>(y),
                 incy,
                 prec);
} /* end BLAS_zsymv2_z_d_x */
