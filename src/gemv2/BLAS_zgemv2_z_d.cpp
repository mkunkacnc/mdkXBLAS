#include "blas_extended.h"
#include "gemv2.hpp"

void BLAS_zgemv2_z_d(enum blas_order_type order,
                     enum blas_trans_type trans,
                     int m,
                     int n,
                     const void *alpha,
                     const void *a,
                     int lda,
                     const double *head_x,
                     const double *tail_x,
                     int incx,
                     const void *beta,
                     void *y,
                     int incy)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * op(A) * head_x + alpha * op(A) * tail_x + beta * y,
 * where A is a general matrix.
 *
 * Arguments
 * =========
 *
 * order   (input) enum blas_order_type
 *         Order of A; row or column major
 *
 * trans   (input) enum blas_trans_type
 *         Transpose of A: no trans, trans, or conjugate trans
 *
 * m       (input) int
 *         Dimension of A
 *
 * n       (input) int
 *         Dimension of A and the length of vector x and z
 *
 * alpha   (input) const void*
 *
 * A       (input) const void*
 *
 * lda     (input) int
 *         Leading dimension of A
 *
 * head_x
 * tail_x  (input) const double*
 *
 * incx    (input) int
 *         The stride for vector x.
 *
 * beta    (input) const void*
 *
 * y       (input/output) void*
 *
 * incy    (input) int
 *         The stride for vector y.
 *
 */
{
//static const char *routine_name = "BLAS_zgemv2_z_d";
  XBLAS::gemv2(order,
               trans,
               m,
               n,
               *static_cast<const std::complex<double> *>(alpha),
               static_cast<const std::complex<double> *>(a),
               lda,
               head_x,
               tail_x,
               incx,
               *static_cast<const std::complex<double> *>(beta),
               static_cast<std::complex<double> *>(y),
               incy);
} /* end BLAS_zgemv2_z_d */
