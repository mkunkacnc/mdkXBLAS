#include "blas_extended.h"
#include "blas_extended_private.h"
#include "gemm/XBLAS_gemm.hpp"

void BLAS_dgemm_s_s(enum blas_order_type order,
                    enum blas_trans_type transa,
                    enum blas_trans_type transb,
                    int m,
                    int n,
                    int k,
                    double alpha,
                    const float *a,
                    int lda,
                    const float *b,
                    int ldb,
                    double beta,
                    double *c,
                    int ldc)
/*
 * Purpose
 * =======
 *
 * This routine computes the matrix product:
 *
 *      C   <-  alpha * op(A) * op(B)  +  beta * C .
 *
 * where op(M) represents either M, M transpose,
 * or M conjugate transpose.
 *
 * Arguments
 * =========
 *
 * order   (input) enum blas_order_type
 *         Storage format of input matrices A, B, and C.
 *
 * transa  (input) enum blas_trans_type
 *         Operation to be done on matrix A before multiplication.
 *           Can be no operation, transposition, or conjugate transposition.
 *
 * transb  (input) enum blas_trans_type
 *         Operation to be done on matrix B before multiplication.
 *           Can be no operation, transposition, or conjugate transposition.
 *
 * m n k   (input) int
 *         The dimensions of matrices A, B, and C.
 *         Matrix C is m-by-n matrix.
 *         Matrix A is m-by-k if A is not transposed,
 *                     k-by-m otherwise.
 *         Matrix B is k-by-n if B is not transposed,
 *                     n-by-k otherwise.
 *
 * alpha   (input) double
 *
 * a       (input) const float*
 *         matrix A.
 *
 * lda     (input) int
 *         leading dimension of A.
 *
 * b       (input) const float*
 *         matrix B
 *
 * ldb     (input) int
 *         leading dimension of B.
 *
 * beta    (input) double
 *
 * c       (input/output) double*
 *         matrix C
 *
 * ldc     (input) int
 *         leading dimension of C.
 *
 */
{
//static const char routine_name[] = "BLAS_dgemm_s_s";
  XBLAS::gemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
} /* end BLAS_dgemm_s_s */
