#include "blas_extended.h"
#include "blas_extended_private.h"
#include "gemv/XBLAS_gemv.hpp"

void BLAS_zgemv_d_d_x(enum blas_order_type order,
                      enum blas_trans_type trans,
                      int m,
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
 * Computes y = alpha * A * x + beta * y, where A is a general matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Order of AP; row or column major
 *
 * trans  (input) enum blas_trans_type
 *        Transpose of AB; no trans,
 *          trans, or conjugate trans
 *
 * m      (input) int
 *        Dimension of AB
 *
 * n      (input) int
 *        Dimension of AB and the length of vector x
 *
 * alpha  (input) const void*
 *
 * A      (input) const double*
 *
 * lda    (input) int
 *        Leading dimension of A
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
//static const char *routine_name = "BLAS_zgemv_d_d_x";
  XBLAS::gemv_x(order,
                trans,
                m,
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
} /* end BLAS_zgemv_d_d_x */
