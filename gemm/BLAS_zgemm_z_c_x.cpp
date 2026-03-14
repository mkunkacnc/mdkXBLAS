#include "blas_extended.h"
#include "blas_extended_private.h"
#include "gemm/XBLAS_gemm.hpp"

void BLAS_zgemm_z_c_x(enum blas_order_type order,
                      enum blas_trans_type transa,
                      enum blas_trans_type transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *a,
                      int lda,
                      const void *b,
                      int ldb,
                      const void *beta,
                      void *c,
                      int ldc,
                      enum blas_prec_type prec)
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
 * alpha   (input) const void*
 *
 * a       (input) const void*
 *         matrix A.
 *
 * lda     (input) int
 *         leading dimension of A.
 *
 * b       (input) const void*
 *         matrix B
 *
 * ldb     (input) int
 *         leading dimension of B.
 *
 * beta    (input) const void*
 *
 * c       (input/output) void*
 *         matrix C
 *
 * ldc     (input) int
 *         leading dimension of C.
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
//static const char routine_name[] = "BLAS_zgemm_z_c_x";
  XBLAS::gemm_x(order,
                transa,
                transb,
                m,
                n,
                k,
                *static_cast<const std::complex<double> *>(alpha),
                static_cast<const std::complex<double> *>(a),
                lda,
                static_cast<const std::complex<float> *>(b),
                ldb,
                *static_cast<const std::complex<double> *>(beta),
                static_cast<std::complex<double> *>(c),
                ldc,
                prec);
} /* end BLAS_zgemm_z_c_x */
