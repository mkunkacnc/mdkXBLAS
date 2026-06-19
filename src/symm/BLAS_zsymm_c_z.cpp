#include "blas_extended.h"
#include "blas_extended_private.h"
#include "symm.hpp"

void BLAS_zsymm_c_z(enum blas_order_type order,
                    enum blas_side_type side,
                    enum blas_uplo_type uplo,
                    int m,
                    int n,
                    const void *alpha,
                    const void *a,
                    int lda,
                    const void *b,
                    int ldb,
                    const void *beta,
                    void *c,
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
 * alpha  (input) const void*
 *
 * a      (input) const void*
 *        Matrix A.
 *
 * lda    (input) int
 *        Leading dimension of matrix A.
 *
 * b      (input) const void*
 *        Matrix B.
 *
 * ldb    (input) int
 *        Leading dimension of matrix B.
 *
 * beta   (input) const void*
 *
 * c      (input/output) void*
 *        Matrix C.
 *
 * ldc    (input) int
 *        Leading dimension of matrix C.
 *
 */
{
//static const char *routine_name = "BLAS_zsymm_c_z";
  XBLAS::symm(order,
              side,
              uplo,
              m,
              n,
              *static_cast<const std::complex<double> *>(alpha),
              static_cast<const std::complex<float> *>(a),
              lda,
              static_cast<const std::complex<double> *>(b),
              ldb,
              *static_cast<const std::complex<double> *>(beta),
              static_cast<std::complex<double> *>(c),
              ldc);
} /* end BLAS_zsymm_c_z */
