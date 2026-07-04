#include "blas_extended.h"
#include "symm.hpp"

void BLAS_dsymm_s_s_x(enum blas_order_type order,
                      enum blas_side_type side,
                      enum blas_uplo_type uplo,
                      int m,
                      int n,
                      double alpha,
                      const float *a,
                      int lda,
                      const float *b,
                      int ldb,
                      double beta,
                      double *c,
                      int ldc,
                      enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routines computes one of the matrix product:
 *
 *     C  <-  alpha * A * B  +  beta * C
 *     C  <-  alpha * B * A  +  beta * C
 *
 * where A is a symmetric matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Storage format of input matrices A, B, and C.
 *
 * side   (input) enum blas_side_type
 *        Determines which side of matrix B is matrix A is multiplied.
 *
 * uplo   (input) enum blas_uplo_type
 *        Determines which half of matrix A (upper or lower triangle)
 *          is accessed.
 *
 * m n     (input) int
 *         Size of matrices A, B, and C.
 *         Matrix A is m-by-m if it is multiplied on the left,
 *                     n-by-n otherwise.
 *         Matrices B and C are m-by-n.
 *
 * alpha  (input) double
 *
 * a      (input) const float*
 *        Matrix A.
 *
 * lda    (input) int
 *        Leading dimension of matrix A.
 *
 * b      (input) const float*
 *        Matrix B.
 *
 * ldb    (input) int
 *        Leading dimension of matrix B.
 *
 * beta   (input) double
 *
 * c      (input/output) double*
 *        Matrix C.
 *
 * ldc    (input) int
 *        Leading dimension of matrix C.
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
//static const char *routine_name = "BLAS_dsymm_s_s_x";
  XBLAS::symm_x(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc, prec);
} /* end BLAS_dsymm_s_s_x */
