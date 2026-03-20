#include "blas_extended.h"
#include "blas_extended_private.h"
#include "gemv2/XBLAS_gemv2.hpp"

void BLAS_zgemv2_c_z_x(enum blas_order_type order,
                       enum blas_trans_type trans,
                       int m,
                       int n,
                       const void *alpha,
                       const void *a,
                       int lda,
                       const void *head_x,
                       const void *tail_x,
                       int incx,
                       const void *beta,
                       void *y,
                       int incy,
                       enum blas_prec_type prec)
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
 * tail_x  (input) const void*
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
//static const char *routine_name = "BLAS_zgemv2_c_z_x";
  XBLAS::gemv2_x(order,
                 trans,
                 m,
                 n,
                 *static_cast<const std::complex<double> *>(alpha),
                 static_cast<const std::complex<float> *>(a),
                 lda,
                 static_cast<const std::complex<double> *>(head_x),
                 static_cast<const std::complex<double> *>(tail_x),
                 incx,
                 *static_cast<const std::complex<double> *>(beta),
                 static_cast<std::complex<double> *>(y),
                 incy,
                 prec);
} /* end BLAS_zgemv2_c_z_x */
