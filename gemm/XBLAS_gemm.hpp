#ifndef XBLAS_GEMM_HPP
#define XBLAS_GEMM_HPP

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
constexpr void gemm(blas_order_type order,
                    blas_trans_type transa,
                    blas_trans_type transb,
                    N m,
                    N n,
                    N k,
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
 * order   (input) blas_order_type
 *         Storage format of input matrices A, B, and C.
 *
 * transa  (input) blas_trans_type
 *         Operation to be done on matrix A before multiplication.
 *           Can be no operation, transposition, or conjugate transposition.
 *
 * transb  (input) blas_trans_type
 *         Operation to be done on matrix B before multiplication.
 *           Can be no operation, transposition, or conjugate transposition.
 *
 * m n k   (input) N
 *         The dimensions of matrices A, B, and C.
 *         Matrix C is m-by-n matrix.
 *         Matrix A is m-by-k if A is not transposed,
 *                     k-by-m otherwise.
 *         Matrix B is k-by-n if B is not transposed,
 *                     n-by-k otherwise.
 *
 * alpha   (input) T
 *
 * a       (input) const A*
 *         matrix A.
 *
 * lda     (input) N
 *         leading dimension of A.
 *
 * b       (input) const B*
 *         matrix B
 *
 * ldb     (input) N
 *         leading dimension of B.
 *
 * beta    (input) T
 *
 * c       (input/output) T*
 *         matrix C
 *
 * ldc     (input) N
 *         leading dimension of C.
 *
 */
{
  static const char *routine_name = "XBLAS::gemm";

  using PrdType = impl::get_inner_type_t<A, B, TmpType>;

  FPU_FIX_DECL;

  /* Integer Index Variables */
  IdxType i, j, h;

  IdxType ai, bj, ci;
  IdxType aih, bhj, cij;                /* Index into matrices a, b, c during multiply */

  IdxType incai, incaih;                /* Index increments for matrix a */
  IdxType incbj, incbhj;                /* Index increments for matrix b */
  IdxType incci, inccij;                /* Index increments for matrix c */

  /* Input Matrices */

  /* Output Matrix */

  /* Input Scalars */

  /* Temporary Floating-Point Variables */
  A a_elem;
  B b_elem;
  T c_elem;
  PrdType prod;
  PrdType sum;
  TmpType tmp1;
  TmpType tmp2;

  /* Test for error conditions */
  if (m < 0)
    BLAS_error(routine_name, -4, m, nullptr);
  if (n < 0)
    BLAS_error(routine_name, -5, n, nullptr);
  if (k < 0)
    BLAS_error(routine_name, -6, k, nullptr);

  if (order == blas_colmajor) {
    if (ldc < m)
      BLAS_error(routine_name, -14, ldc, nullptr);

    if (transa == blas_no_trans) {
      if (lda < m)
        BLAS_error(routine_name, -9, lda, nullptr);
    } else {
      if (lda < k)
        BLAS_error(routine_name, -9, lda, nullptr);
    }

    if (transb == blas_no_trans) {
      if (ldb < k)
        BLAS_error(routine_name, -11, ldb, nullptr);
    } else {
      if (ldb < n)
        BLAS_error(routine_name, -11, ldb, nullptr);
    }
  } else {
    /* row major */
    if (ldc < n)
      BLAS_error(routine_name, -14, ldc, nullptr);

    if (transa == blas_no_trans) {
      if (lda < k)
        BLAS_error(routine_name, -9, lda, nullptr);
    } else {
      if (lda < m)
        BLAS_error(routine_name, -9, lda, nullptr);
    }

    if (transb == blas_no_trans) {
      if (ldb < n)
        BLAS_error(routine_name, -11, ldb, nullptr);
    } else {
      if (ldb < k)
        BLAS_error(routine_name, -11, ldb, nullptr);
    }
  }

  /* Test for no-op */
  if (n == 0 || m == 0 || k == 0)
    return;
  if (alpha == T(0) && beta == T(1)) {
    return;
  }

  /* Set Index Parameters */
  if (order == blas_colmajor) {
    incci = 1;
    inccij = ldc;

    if (transa == blas_no_trans) {
      incai = 1;
      incaih = lda;
    } else {
      incai = lda;
      incaih = 1;
    }

    if (transb == blas_no_trans) {
      incbj = ldb;
      incbhj = 1;
    } else {
      incbj = 1;
      incbhj = ldb;
    }
  } else {
    /* row major */
    incci = ldc;
    inccij = 1;

    if (transa == blas_no_trans) {
      incai = lda;
      incaih = 1;
    } else {
      incai = 1;
      incaih = lda;
    }

    if (transb == blas_no_trans) {
      incbj = 1;
      incbhj = ldb;
    } else {
      incbj = ldb;
      incbhj = 1;
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  /* alpha = 0.  In this case, just return beta * C */
  if (alpha == T(0)) {
    ci = 0;
    for (i = 0; i < m; i++, ci += incci) {
      cij = ci;
      for (j = 0; j < n; j++, cij += inccij) {
        c_elem = c[cij];
        tmp1 = impl::mul<TmpType>(c_elem, beta);
        c[cij] = impl::to<T>(tmp1);
      }
    }
  } else if (alpha == T(1)) {
    /* Case alpha == 1. */
    if (beta == T(0)) {
      /* Case alpha == 1, beta == 0.   We compute  C <--- A * B */
      ci = 0;
      ai = 0;
      for (i = 0; i < m; i++, ci += incci, ai += incai) {
        cij = ci;
        bj = 0;
        for (j = 0; j < n; j++, cij += inccij, bj += incbj) {
          aih = ai;
          bhj = bj;
          sum = impl::zero_v<PrdType>;
          for (h = 0; h < k; h++, aih += incaih, bhj += incbhj) {
            a_elem = a[aih];
            b_elem = b[bhj];
            if constexpr (impl::is_complex_v<A>) {
              if (transa == blas_conj_trans)
                a_elem = impl::Conj::func(a_elem);
            }
            if constexpr (impl::is_complex_v<B>) {
              if (transb == blas_conj_trans)
                b_elem = impl::Conj::func(b_elem);
            }
            prod = impl::mul<PrdType>(a_elem, b_elem);
            sum = sum + prod;
          }
          c[cij] = impl::to<T>(sum);
        }
      }
    } else {
      /* Case alpha == 1, but beta != 0.
         We compute   C <--- A * B + beta * C   */
      ci = 0;
      ai = 0;
      for (i = 0; i < m; i++, ci += incci, ai += incai) {
        cij = ci;
        bj = 0;
        for (j = 0; j < n; j++, cij += inccij, bj += incbj) {
          aih = ai;
          bhj = bj;
          sum = impl::zero_v<PrdType>;
          for (h = 0; h < k; h++, aih += incaih, bhj += incbhj) {
            a_elem = a[aih];
            b_elem = b[bhj];
            if constexpr (impl::is_complex_v<A>) {
              if (transa == blas_conj_trans)
                a_elem = impl::Conj::func(a_elem);
            }
            if constexpr (impl::is_complex_v<B>) {
              if (transb == blas_conj_trans)
                b_elem = impl::Conj::func(b_elem);
            }
            prod = impl::mul<PrdType>(a_elem, b_elem);
            sum = sum + prod;
          }
          c_elem = c[cij];
          tmp2 = impl::mul<TmpType>(c_elem, beta);
          tmp1 = sum;
          tmp1 = tmp2 + tmp1;
          c[cij] = impl::to<T>(tmp1);
        }
      }
    }
  } else {
    /* The most general form,   C <-- alpha * A * B + beta * C  */
    ci = 0;
    ai = 0;
    for (i = 0; i < m; i++, ci += incci, ai += incai) {
      cij = ci;
      bj = 0;
      for (j = 0; j < n; j++, cij += inccij, bj += incbj) {
        aih = ai;
        bhj = bj;
        sum = impl::zero_v<PrdType>;
        for (h = 0; h < k; h++, aih += incaih, bhj += incbhj) {
          a_elem = a[aih];
          b_elem = b[bhj];
          if constexpr (impl::is_complex_v<A>) {
            if (transa == blas_conj_trans)
              a_elem = impl::Conj::func(a_elem);
          }
          if constexpr (impl::is_complex_v<B>) {
            if (transb == blas_conj_trans)
              b_elem = impl::Conj::func(b_elem);
          }
          prod = impl::mul<PrdType>(a_elem, b_elem);
          sum = sum + prod;
        }
        tmp1 = impl::mul<TmpType>(sum, alpha);
        c_elem = c[cij];
        tmp2 = impl::mul<TmpType>(c_elem, beta);
        tmp1 = tmp1 + tmp2;
        c[cij] = impl::to<T>(tmp1);
      }
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::gemm */

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
constexpr void gemm_x(blas_order_type order,
                      blas_trans_type transa,
                      blas_trans_type transb,
                      N m,
                      N n,
                      N k,
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
 * order   (input) blas_order_type
 *         Storage format of input matrices A, B, and C.
 *
 * transa  (input) blas_trans_type
 *         Operation to be done on matrix A before multiplication.
 *           Can be no operation, transposition, or conjugate transposition.
 *
 * transb  (input) blas_trans_type
 *         Operation to be done on matrix B before multiplication.
 *           Can be no operation, transposition, or conjugate transposition.
 *
 * m n k   (input) N
 *         The dimensions of matrices A, B, and C.
 *         Matrix C is m-by-n matrix.
 *         Matrix A is m-by-k if A is not transposed,
 *                     k-by-m otherwise.
 *         Matrix B is k-by-n if B is not transposed,
 *                     n-by-k otherwise.
 *
 * alpha   (input) T
 *
 * a       (input) const A*
 *         matrix A.
 *
 * lda     (input) N
 *         leading dimension of A.
 *
 * b       (input) const B*
 *         matrix B
 *
 * ldb     (input) N
 *         leading dimension of B.
 *
 * beta    (input) T
 *
 * c       (input/output) T*
 *         matrix C
 *
 * ldc     (input) N
 *         leading dimension of C.
 *
 * prec    (input) blas_prec_type
 *         Specifies the internal precision to be used.
 *         = blas_prec_single: single precision.
 *         = blas_prec_double: double precision.
 *         = blas_prec_extra : anything at least 1.5 times as accurate
 *                             than double, and wider than 80-bits.
 *                             We use double-double in our implementation.
 *
 */
{
  static const char *routine_name = "XBLAS::gemm_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::gemm<T, A, B, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  case blas_prec_double:
    XBLAS::gemm<T, A, B, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  case blas_precndigenous:
    XBLAS::gemm<T, A, B, N, impl::internal_precision_t<T, blas_precndigenous>, IdxType>(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  case blas_prec_extra:
    XBLAS::gemm<T, A, B, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  default:
    BLAS_error(routine_name, -15, prec, nullptr);
    break;
  }
} /* end XBLAS::gemm_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_GEMM_HPP
