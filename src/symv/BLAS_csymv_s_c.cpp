#include "blas_extended.h"
#include "symv.hpp"

void BLAS_csymv_s_c(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    int n,
                    const void *alpha,
                    const float *a,
                    int lda,
                    const void *x,
                    int incx,
                    const void *beta,
                    void *y,
                    int incy)
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
 * a      (input) const float*
 *        Matrix A.
 *
 * lda    (input) int
 *        Leading dimension of matrix A.
 *
 * x      (input) const void*
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
 */
{
//static const char *routine_name = "BLAS_csymv_s_c";
  XBLAS::symv(order,
              uplo,
              n,
              *static_cast<const std::complex<float> *>(alpha),
              a,
              lda,
              static_cast<const std::complex<float> *>(x),
              incx,
              *static_cast<const std::complex<float> *>(beta),
              static_cast<std::complex<float> *>(y),
              incy);
} /* end BLAS_csymv_s_c */
