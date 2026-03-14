#include "blas_extended.h"
#include "blas_extended_private.h"
#include "hemm/XBLAS_hemm.hpp"

void BLAS_chemm_c_s_x(enum blas_order_type order,
                      enum blas_side_type side,
                      enum blas_uplo_type uplo,
                      int m,
                      int n,
                      const void *alpha,
                      const void *a,
                      int lda,
                      const float *b,
                      int ldb,
                      const void *beta,
                      void *c,
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
 * where A is a hermitian matrix.
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
 * b      (input) const float*
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
//static const char routine_name[] = "BLAS_chemm_c_s_x";
  XBLAS::hemm_x(order,
                side,
                uplo,
                m,
                n,
                *static_cast<const std::complex<float> *>(alpha),
                static_cast<const std::complex<float> *>(a),
                lda,
                b,
                ldb,
                *static_cast<const std::complex<float> *>(beta),
                static_cast<std::complex<float> *>(c),
                ldc,
                prec);
} /* end BLAS_chemm_c_s_x */
