#include "blas_extended.h"
#include "blas_extended_private.h"
#include "gemv/XBLAS_gemv.hpp"

void BLAS_cgemv_c_s(enum blas_order_type order,
                    enum blas_trans_type trans,
                    int m,
                    int n,
                    const void *alpha,
                    const void *a,
                    int lda,
                    const float *x,
                    int incx,
                    const void *beta,
                    void *y,
                    int incy)
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
 * A      (input) const void*
 *
 * lda    (input) int
 *        Leading dimension of A
 *
 * x      (input) const float*
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
 */
{
//static const char *routine_name = "BLAS_cgemv_c_s";
  XBLAS::gemv(order,
              trans,
              m,
              n,
              *static_cast<const std::complex<float> *>(alpha),
              static_cast<const std::complex<float> *>(a),
              lda,
              x,
              incx,
              *static_cast<const std::complex<float> *>(beta),
              static_cast<std::complex<float> *>(y),
              incy);
} /* end BLAS_cgemv_c_s */
