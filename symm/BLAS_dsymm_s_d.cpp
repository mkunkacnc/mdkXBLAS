#include "blas_extended.h"
#include "blas_extended_private.h"
#include "symm/XBLAS_symm.hpp"

void BLAS_dsymm_s_d(enum blas_order_type order,
                    enum blas_side_type side,
                    enum blas_uplo_type uplo,
                    int m,
                    int n,
                    double alpha,
                    const float *a,
                    int lda,
                    const double *b,
                    int ldb,
                    double beta,
                    double *c,
                    int ldc)
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
 * b      (input) const double*
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
 */
{
//static const char *routine_name = "BLAS_dsymm_s_d";
  XBLAS::symm(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
} /* end BLAS_dsymm_s_d */
