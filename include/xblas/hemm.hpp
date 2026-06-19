#ifndef XBLAS_HEMM_HPP
#define XBLAS_HEMM_HPP

#include "blas_enum.h"
#include "impl.hpp"

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

template<int need_alpha,
         int need_beta,
         typename TmpType,
         typename PrdType,
         typename T,
         typename A,
         typename B,
         typename IdxType>
constexpr void hemm_impl(IdxType m_i,
                         IdxType n_i,
                         T alpha,
                         const A *a,
                         const B *b,
                         T beta,
                         T *c,
                         int conj_flag,
                         IdxType incai,
                         IdxType incaik1,
                         IdxType incaik2,
                         IdxType incbj,
                         IdxType incbkj,
                         IdxType incci,
                         IdxType inccij)
{
  IdxType ai = 0;
  IdxType ci = 0;
  for (IdxType i = 0; i < m_i; i++) {
    IdxType bj = 0;
    IdxType cij = ci;
    for (IdxType j = 0; j < n_i; j++) {
      IdxType aik = ai;
      IdxType bkj = bj;
      PrdType sum = impl::zero_v<PrdType>;

      IdxType k = 0;
      for (; k < i; k++) {
        A a_elem = a[aik];
        if (conj_flag == 1) {
          a_elem = impl::Conj::func(a_elem);
        }
        PrdType prod = impl::mul<PrdType>(a_elem, b[bkj]);
        sum += prod;

        aik += incaik1;
        bkj += incbkj;
      }

      for (; k < m_i; k++) {
        A a_elem = a[aik];
        if (conj_flag == 0) {
          a_elem = impl::Conj::func(a_elem);
        }
        PrdType prod = impl::mul<PrdType>(a_elem, b[bkj]);
        sum += prod;

        aik += incaik2;
        bkj += incbkj;
      }

      if constexpr (need_alpha == 1) {
        if constexpr (need_beta == 0) {
          c[cij] = impl::to<T>(sum);
        } else {
          TmpType tmp1 = sum;
          TmpType tmp2 = impl::mul<TmpType>(c[cij], beta);
          tmp1 += tmp2;
          c[cij] = impl::to<T>(tmp1);
        }
      } else {
        TmpType tmp1 = impl::mul<TmpType>(sum, alpha);
        TmpType tmp2 = impl::mul<TmpType>(c[cij], beta);
        tmp1 += tmp2;
        c[cij] = impl::to<T>(tmp1);
      }

      bj += incbj;
      cij += inccij;
    }

    ai += incai;
    ci += incci;
  }
} /* end XBLAS::impl::hemm_impl */

//-----------------
} // namespace impl
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
constexpr void hemm(blas_order_type order,
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
 * where A is a hermitian matrix.
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
  static const char *routine_name = "XBLAS::hemm";

  using PrdType = impl::get_inner_type_t<A, B, TmpType>;

  FPU_FIX_DECL;

  /* Check for error conditions. */
  if (m < 0)
    BLAS_error(routine_name, -4, m, nullptr);
  if (n < 0)
    BLAS_error(routine_name, -5, n, nullptr);

  if ((side == blas_left_side && lda < m) ||
      (side == blas_right_side && lda < n)) {
    BLAS_error(routine_name, -8, lda, nullptr);
  }
  if ((order == blas_colmajor && ldb < m) ||
      (order == blas_rowmajor && ldb < n)){
    BLAS_error(routine_name, -10, ldb, nullptr);
  }
  if ((order == blas_colmajor && ldc < m) ||
      (order == blas_rowmajor && ldc < n)) {
    BLAS_error(routine_name, -13, ldc, nullptr);
  }

  /* Test for no-op */
  if (m == 0 || n == 0) {
    return;
  }
  if (alpha == T(0) && beta == T(1)) {
    return;
  }

  /* Set Index Parameters */
  IdxType m_i, n_i;
  if (side == blas_left_side) {
    m_i = m;
    n_i = n;
  } else {
    m_i = n;
    n_i = m;
  }

  IdxType incbj, incci;
  IdxType incbkj, inccij;
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

  IdxType incai;
  IdxType incaik1, incaik2;
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

  IdxType conj_flag;
  if ((side == blas_left_side && uplo == blas_upper) ||
      (side == blas_right_side && uplo == blas_lower))
    conj_flag = 1;
  else
    conj_flag = 0;

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  /* alpha = 0.  In this case, just return beta * C */
  if (alpha == T(0)) {
    IdxType ci = 0;
    for (IdxType i = 0; i < m_i; i++) {
      IdxType cij = ci;
      for (IdxType j = 0; j < n_i; j++) {
        TmpType tmp1 = impl::mul<TmpType>(c[cij], beta);
        c[cij] = impl::to<T>(tmp1);
        cij += inccij;
      }
      ci += incci;
    }
  } else if (alpha == T(1)) {
    /* Case alpha == 1. */
    if (beta == T(0)) {
      /* Case alpha = 1, beta = 0.  We compute  C <--- A * B   or  B * A */
      impl::hemm_impl< 1,  0, TmpType, PrdType>(m_i, n_i, alpha, a, b, beta, c,
                                                conj_flag, incai, incaik1, incaik2, incbj, incbkj, incci, inccij);
    } else {
      /* Case alpha = 1, but beta != 0.
         We compute  C  <--- A * B + beta * C
         or  C  <--- B * A + beta * C  */
      impl::hemm_impl< 1, -1, TmpType, PrdType>(m_i, n_i, alpha, a, b, beta, c,
                                                conj_flag, incai, incaik1, incaik2, incbj, incbkj, incci, inccij);
    }
  } else {
    /* The most general form,   C <--- alpha * A * B + beta * C
       or   C <--- alpha * B * A + beta * C  */
    impl::hemm_impl<-1, -1, TmpType, PrdType>(m_i, n_i, alpha, a, b, beta, c,
                                              conj_flag, incai, incaik1, incaik2, incbj, incbkj, incci, inccij);
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::hemm */

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
constexpr void hemm_x(blas_order_type order,
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
 * where A is a hermitian matrix.
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
  static const char *routine_name = "XBLAS::hemm_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::hemm<T, A, B, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  case blas_prec_double:
    XBLAS::hemm<T, A, B, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  case blas_prec_indigenous:
    XBLAS::hemm<T, A, B, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  case blas_prec_extra:
    XBLAS::hemm<T, A, B, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  default:
    BLAS_error(routine_name, -14, prec, nullptr);
    break;
  }
} /* end XBLAS::hemm_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_HEMM_HPP
