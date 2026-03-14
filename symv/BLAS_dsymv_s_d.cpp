#include "blas_extended.h"
#include "blas_extended_private.h"
#include "symv/XBLAS_symv.hpp"

void BLAS_dsymv_s_d(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    int n,
                    double alpha,
                    const float *a,
                    int lda,
                    const double *x,
                    int incx,
                    double beta,
                    double *y,
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
 * alpha  (input) double
 *
 * a      (input) const float*
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
 * beta   (input) double
 *
 * y      (input/output) double*
 *        Vector y.
 *
 * incy   (input) int
 *        Stride for vector y.
 *
 */
{
//static const char routine_name[] = "BLAS_dsymv_s_d";
  XBLAS::symv(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
} /* end BLAS_dsymv_s_d */
