#ifndef XBLAS_SYMM_HPP
#define XBLAS_SYMM_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename A,
         typename B,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<B, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void symm(blas_order_type order,
                    blas_side_type side,
                    blas_uplo_type uplo,
                    N m,
                    N n,
                    T alpha,
                    const A *a,
                    N lda,
                    const B *b,
                    N ldb,
                    T beta,
                    T *c,
                    N ldc)
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
 * order  (input) blas_order_type
 *        Storage format of input matrices A, B, and C.
 *
 * side   (input) blas_side_type
 *        Determines which side of matrix B is matrix A is multiplied.
 *
 * uplo   (input) blas_uplo_type
 *        Determines which half of matrix A (upper or lower triangle)
 *          is accessed.
 *
 * m n     (input) N
 *         Size of matrices A, B, and C.
 *         Matrix A is m-by-m if it is multiplied on the left,
 *                     n-by-n otherwise.
 *         Matrices B and C are m-by-n.
 *
 * alpha  (input) T
 *
 * a      (input) const A*
 *        Matrix A.
 *
 * lda    (input) N
 *        Leading dimension of matrix A.
 *
 * b      (input) const B*
 *        Matrix B.
 *
 * ldb    (input) N
 *        Leading dimension of matrix B.
 *
 * beta   (input) T
 *
 * c      (input/output) T*
 *        Matrix C.
 *
 * ldc    (input) N
 *        Leading dimension of matrix C.
 *
 */
{
//static const char *routine_name = "XBLAS::symm";

  using PrdType = impl::get_inner_type_t<A, B, TmpType>;

  FPU_FIX_DECL;

  /* Integer Index Variables */
  IdxType i, j, k;

  IdxType ai, bj, ci;
  IdxType aik, bkj, cij;

  IdxType incai, incbj, incci;
  IdxType incaik1, incaik2, incbkj, inccij;

  IdxType m_i, n_i;

  /* Input Matrices */

  /* Output Matrix */

  /* Input Scalars */

  /* Temporary Floating-Point Variables */

  /* Check for error conditions. */
  if (m <= 0 || n <= 0) {
    return;
  }

  if (order == blas_colmajor && (ldb < m || ldc < m)) {
    return;
  }
  if (order == blas_rowmajor && (ldb < n || ldc < n)) {
    return;
  }
  if (side == blas_left_side && lda < m) {
    return;
  }
  if (side == blas_right_side && lda < n) {
    return;
  }

  /* Test for no-op */
  if (alpha == T(0) && beta == T(1)) {
    return;
  }

  /* Set Index Parameters */
  if (side == blas_left_side) {
    m_i = m;
    n_i = n;
  } else {
    m_i = n;
    n_i = m;
  }

  if ((order == blas_colmajor && side == blas_left_side) ||
      (order == blas_rowmajor && side == blas_right_side)) {
    incbj = ldb;
    incbkj = 1;
    incci = 1;
    inccij = ldc;
  } else {
    incbj = 1;
    incbkj = ldb;
    incci = ldc;
    inccij = 1;
  }

  if ((order == blas_colmajor && uplo == blas_upper) ||
      (order == blas_rowmajor && uplo == blas_lower)) {
    incai = lda;
    incaik1 = 1;
    incaik2 = lda;
  } else {
    incai = 1;
    incaik1 = lda;
    incaik2 = 1;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  /* alpha = 0.  In this case, just return beta * C */
  if (alpha == T(0)) {
    for (i = 0, ci = 0; i < m_i; i++, ci += incci) {
      for (j = 0, cij = ci; j < n_i; j++, cij += inccij) {
        T c_elem = c[cij];
        TmpType tmp1 = impl::mul<TmpType>(c_elem, beta);
        c[cij] = impl::to<T>(tmp1);
      }
    }
  } else if (alpha == T(1)) {
    /* Case alpha == 1. */
    if (beta == T(0)) {
      /* Case alpha = 1, beta = 0.  We compute  C <--- A * B   or  B * A */
      for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
        for (j = 0, cij = ci, bj = 0; j < n_i;
             j++, cij += inccij, bj += incbj) {
          PrdType sum = impl::zero_v<PrdType>;
          for (k = 0, aik = ai, bkj = bj; k < i;
               k++, aik += incaik1, bkj += incbkj) {
            A a_elem = a[aik];
            B b_elem = b[bkj];
            PrdType prod = impl::mul<PrdType>(a_elem, b_elem);
            sum = sum + prod;
          }
          for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
            A a_elem = a[aik];
            B b_elem = b[bkj];
            PrdType prod = impl::mul<PrdType>(a_elem, b_elem);
            sum = sum + prod;
          }
          c[cij] = impl::to<T>(sum);
        }
      }
    } else {
      /* Case alpha = 1, but beta != 0.
         We compute  C  <--- A * B + beta * C
         or  C  <--- B * A + beta * C  */
      for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
        for (j = 0, cij = ci, bj = 0; j < n_i;
             j++, cij += inccij, bj += incbj) {
          PrdType sum = impl::zero_v<PrdType>;
          for (k = 0, aik = ai, bkj = bj; k < i;
               k++, aik += incaik1, bkj += incbkj) {
            A a_elem = a[aik];
            B b_elem = b[bkj];
            PrdType prod = impl::mul<PrdType>(a_elem, b_elem);
            sum = sum + prod;
          }
          for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
            A a_elem = a[aik];
            B b_elem = b[bkj];
            PrdType prod = impl::mul<PrdType>(a_elem, b_elem);
            sum = sum + prod;
          }
          T c_elem = c[cij];
          TmpType tmp2 = impl::mul<TmpType>(c_elem, beta);
          TmpType tmp1 = sum;
          tmp1 = tmp2 + tmp1;
          c[cij] = impl::to<T>(tmp1);
        }
      }
    }

  } else {
    /* The most general form,   C <--- alpha * A * B + beta * C
       or   C <--- alpha * B * A + beta * C  */

    for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
      for (j = 0, cij = ci, bj = 0; j < n_i; j++, cij += inccij, bj += incbj) {

        PrdType sum = impl::zero_v<PrdType>;

        for (k = 0, aik = ai, bkj = bj; k < i;
             k++, aik += incaik1, bkj += incbkj) {
          A a_elem = a[aik];
          B b_elem = b[bkj];
          PrdType prod = impl::mul<PrdType>(a_elem, b_elem);
          sum = sum + prod;
        }
        for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
          A a_elem = a[aik];
          B b_elem = b[bkj];
          PrdType prod = impl::mul<PrdType>(a_elem, b_elem);
          sum = sum + prod;
        }
        TmpType tmp1 = impl::mul<TmpType>(sum, alpha);
        T c_elem = c[cij];
        TmpType tmp2 = impl::mul<TmpType>(c_elem, beta);
        tmp1 = tmp1 + tmp2;
        c[cij] = impl::to<T>(tmp1);
      }
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::symm */

//-----------------

template<typename T,
         typename A,
         typename B,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<B, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void symm_x(blas_order_type order,
                      blas_side_type side,
                      blas_uplo_type uplo,
                      N m,
                      N n,
                      T alpha,
                      const A *a,
                      N lda,
                      const B *b,
                      N ldb,
                      T beta,
                      T *c,
                      N ldc,
                      blas_prec_type prec)
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
 * order  (input) blas_order_type
 *        Storage format of input matrices A, B, and C.
 *
 * side   (input) blas_side_type
 *        Determines which side of matrix B is matrix A is multiplied.
 *
 * uplo   (input) blas_uplo_type
 *        Determines which half of matrix A (upper or lower triangle)
 *          is accessed.
 *
 * m n     (input) N
 *         Size of matrices A, B, and C.
 *         Matrix A is m-by-m if it is multiplied on the left,
 *                     n-by-n otherwise.
 *         Matrices B and C are m-by-n.
 *
 * alpha  (input) T
 *
 * a      (input) const A*
 *        Matrix A.
 *
 * lda    (input) N
 *        Leading dimension of matrix A.
 *
 * b      (input) const B*
 *        Matrix B.
 *
 * ldb    (input) N
 *        Leading dimension of matrix B.
 *
 * beta   (input) T
 *
 * c      (input/output) T*
 *        Matrix C.
 *
 * ldc    (input) N
 *        Leading dimension of matrix C.
 *
 * prec   (input) blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 */
{
  static const char *routine_name = "XBLAS::symm_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::symm<T, A, B, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  case blas_prec_double:
    XBLAS::symm<T, A, B, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  case blas_prec_indigenous:
    XBLAS::symm<T, A, B, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  case blas_prec_extra:
    XBLAS::symm<T, A, B, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  default:
    BLAS_error(routine_name, -14, prec, nullptr);
    break;
  }
} /* end XBLAS::symm_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_SYMM_HPP
