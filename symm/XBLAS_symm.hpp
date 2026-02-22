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
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<B, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void symm(blas_order_type order,
                    blas_side_type side,
                    blas_uplo_type uplo,
                    IdxType m,
                    IdxType n,
                    T alpha,
                    const A *a,
                    IdxType lda,
                    const B *b,
                    IdxType ldb,
                    T beta,
                    T *c,
                    IdxType ldc)
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
 * m n     (input) IdxType
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
 * lda    (input) IdxType
 *        Leading dimension of matrix A.
 *
 * b      (input) const B*
 *        Matrix B.
 *
 * ldb    (input) IdxType
 *        Leading dimension of matrix B.
 *
 * beta   (input) T
 *
 * c      (input/output) T*
 *        Matrix C.
 *
 * ldc    (input) IdxType
 *        Leading dimension of matrix C.
 *
 */
{
//static const char routine_name[] = "XBLAS::symm";

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
  const A *a_i = a;
  const B *b_i = b;

  /* Output Matrix */
  T *c_i = c;

  /* Input Scalars */
  T alpha_i = alpha;
  T beta_i = beta;

  /* Temporary Floating-Point Variables */
  A a_elem;
  B b_elem;
  T c_elem;
  PrdType prod;
  PrdType sum;
  TmpType tmp1;
  TmpType tmp2;

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
  if (alpha_i == T(0) && beta_i == T(1)) {
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
  if (alpha_i == T(0)) {
    for (i = 0, ci = 0; i < m_i; i++, ci += incci) {
      for (j = 0, cij = ci; j < n_i; j++, cij += inccij) {
        c_elem = c_i[cij];
        tmp1 = impl::mul<TmpType>(c_elem, beta_i);
        c_i[cij] = impl::to<T>(tmp1);
      }
    }
  } else if (alpha_i == T(1)) {
    /* Case alpha == 1. */
    if (beta_i == T(0)) {
      /* Case alpha = 1, beta = 0.  We compute  C <--- A * B   or  B * A */
      for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
        for (j = 0, cij = ci, bj = 0; j < n_i;
             j++, cij += inccij, bj += incbj) {
          sum = impl::zero_v<PrdType>;
          for (k = 0, aik = ai, bkj = bj; k < i;
               k++, aik += incaik1, bkj += incbkj) {
            a_elem = a_i[aik];
            b_elem = b_i[bkj];
            prod = impl::mul<PrdType>(a_elem, b_elem);
            sum = sum + prod;
          }
          for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
            a_elem = a_i[aik];
            b_elem = b_i[bkj];
            prod = impl::mul<PrdType>(a_elem, b_elem);
            sum = sum + prod;
          }
          c_i[cij] = impl::to<T>(sum);
        }
      }
    } else {
      /* Case alpha = 1, but beta != 0.
         We compute  C  <--- A * B + beta * C
         or  C  <--- B * A + beta * C  */
      for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
        for (j = 0, cij = ci, bj = 0; j < n_i;
             j++, cij += inccij, bj += incbj) {
          sum = impl::zero_v<PrdType>;
          for (k = 0, aik = ai, bkj = bj; k < i;
               k++, aik += incaik1, bkj += incbkj) {
            a_elem = a_i[aik];
            b_elem = b_i[bkj];
            prod = impl::mul<PrdType>(a_elem, b_elem);
            sum = sum + prod;
          }
          for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
            a_elem = a_i[aik];
            b_elem = b_i[bkj];
            prod = impl::mul<PrdType>(a_elem, b_elem);
            sum = sum + prod;
          }
          c_elem = c_i[cij];
          tmp2 = impl::mul<TmpType>(c_elem, beta_i);
          tmp1 = sum;
          tmp1 = tmp2 + tmp1;
          c_i[cij] = impl::to<T>(tmp1);
        }
      }
    }

  } else {
    /* The most general form,   C <--- alpha * A * B + beta * C
       or   C <--- alpha * B * A + beta * C  */

    for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
      for (j = 0, cij = ci, bj = 0; j < n_i; j++, cij += inccij, bj += incbj) {

        sum = impl::zero_v<PrdType>;

        for (k = 0, aik = ai, bkj = bj; k < i;
             k++, aik += incaik1, bkj += incbkj) {
          a_elem = a_i[aik];
          b_elem = b_i[bkj];
          prod = impl::mul<PrdType>(a_elem, b_elem);
          sum = sum + prod;
        }
        for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
          a_elem = a_i[aik];
          b_elem = b_i[bkj];
          prod = impl::mul<PrdType>(a_elem, b_elem);
          sum = sum + prod;
        }
        tmp1 = impl::mul<TmpType>(sum, alpha_i);
        c_elem = c_i[cij];
        tmp2 = impl::mul<TmpType>(c_elem, beta_i);
        tmp1 = tmp1 + tmp2;
        c_i[cij] = impl::to<T>(tmp1);
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
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<B, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void symm_x(blas_order_type order,
                      blas_side_type side,
                      blas_uplo_type uplo,
                      IdxType m,
                      IdxType n,
                      T alpha,
                      const A *a,
                      IdxType lda,
                      const B *b,
                      IdxType ldb,
                      T beta,
                      T *c,
                      IdxType ldc,
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
 * a      (input) const double*
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
  switch (prec) {

  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      /* Integer Index Variables */
      int i, j, k;

      int ai, bj, ci;
      int aik, bkj, cij;

      int incai, incbj, incci;
      int incaik1, incaik2, incbkj, inccij;

      int m_i, n_i;

      /* Input Matrices */
      const double *a_i = a;
      const float *b_i = b;

      /* Output Matrix */
      double *c_i = c;

      /* Input Scalars */
      double alpha_i = alpha;
      double beta_i = beta;

      /* Temporary Floating-Point Variables */
      double a_elem;
      float b_elem;
      double c_elem;
      double prod;
      double sum;
      double tmp1;
      double tmp2;



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
      if (alpha_i == 0.0 && beta_i == 1.0) {
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



      /* Adjustment to increments (if any) */








      /* alpha = 0.  In this case, just return beta * C */
      if (alpha_i == 0.0) {
        for (i = 0, ci = 0; i < m_i; i++, ci += incci) {
          for (j = 0, cij = ci; j < n_i; j++, cij += inccij) {
            c_elem = c_i[cij];
            tmp1 = c_elem * beta_i;
            c_i[cij] = tmp1;
          }
        }
      } else if (alpha_i == 1.0) {

        /* Case alpha == 1. */

        if (beta_i == 0.0) {
          /* Case alpha = 1, beta = 0.  We compute  C <--- A * B   or  B * A */
          for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
            for (j = 0, cij = ci, bj = 0; j < n_i;
                 j++, cij += inccij, bj += incbj) {

              sum = 0.0;

              for (k = 0, aik = ai, bkj = bj; k < i;
                   k++, aik += incaik1, bkj += incbkj) {
                a_elem = a_i[aik];
                b_elem = b_i[bkj];
                prod = a_elem * b_elem;
                sum = sum + prod;
              }
              for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
                a_elem = a_i[aik];
                b_elem = b_i[bkj];
                prod = a_elem * b_elem;
                sum = sum + prod;
              }
              c_i[cij] = sum;
            }
          }
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  C  <--- A * B + beta * C
             or  C  <--- B * A + beta * C  */

          for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
            for (j = 0, cij = ci, bj = 0; j < n_i;
                 j++, cij += inccij, bj += incbj) {

              sum = 0.0;

              for (k = 0, aik = ai, bkj = bj; k < i;
                   k++, aik += incaik1, bkj += incbkj) {
                a_elem = a_i[aik];
                b_elem = b_i[bkj];
                prod = a_elem * b_elem;
                sum = sum + prod;
              }
              for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
                a_elem = a_i[aik];
                b_elem = b_i[bkj];
                prod = a_elem * b_elem;
                sum = sum + prod;
              }
              c_elem = c_i[cij];
              tmp2 = c_elem * beta_i;
              tmp1 = sum;
              tmp1 = tmp2 + tmp1;
              c_i[cij] = tmp1;
            }
          }
        }

      } else {
        /* The most general form,   C <--- alpha * A * B + beta * C
           or   C <--- alpha * B * A + beta * C  */

        for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
          for (j = 0, cij = ci, bj = 0; j < n_i;
               j++, cij += inccij, bj += incbj) {

            sum = 0.0;

            for (k = 0, aik = ai, bkj = bj; k < i;
                 k++, aik += incaik1, bkj += incbkj) {
              a_elem = a_i[aik];
              b_elem = b_i[bkj];
              prod = a_elem * b_elem;
              sum = sum + prod;
            }
            for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
              a_elem = a_i[aik];
              b_elem = b_i[bkj];
              prod = a_elem * b_elem;
              sum = sum + prod;
            }
            tmp1 = sum * alpha_i;
            c_elem = c_i[cij];
            tmp2 = c_elem * beta_i;
            tmp1 = tmp1 + tmp2;
            c_i[cij] = tmp1;
          }
        }
      }



      break;
    }

  case blas_prec_extra:{

      /* Integer Index Variables */
      int i, j, k;

      int ai, bj, ci;
      int aik, bkj, cij;

      int incai, incbj, incci;
      int incaik1, incaik2, incbkj, inccij;

      int m_i, n_i;

      /* Input Matrices */
      const double *a_i = a;
      const float *b_i = b;

      /* Output Matrix */
      double *c_i = c;

      /* Input Scalars */
      double alpha_i = alpha;
      double beta_i = beta;

      /* Temporary Floating-Point Variables */
      double a_elem;
      float b_elem;
      double c_elem;
      double head_prod, tail_prod;
      double head_sum, tail_sum;
      double head_tmp1, tail_tmp1;
      double head_tmp2, tail_tmp2;

      FPU_FIX_DECL;

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
      if (alpha_i == 0.0 && beta_i == 1.0) {
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

      FPU_FIX_START;

      /* Adjustment to increments (if any) */








      /* alpha = 0.  In this case, just return beta * C */
      if (alpha_i == 0.0) {
        for (i = 0, ci = 0; i < m_i; i++, ci += incci) {
          for (j = 0, cij = ci; j < n_i; j++, cij += inccij) {
            c_elem = c_i[cij];
            compute_doubledouble_eq_double_mul_double(&head_tmp1, &tail_tmp1, c_elem, beta_i);
            c_i[cij] = head_tmp1;
          }
        }
      } else if (alpha_i == 1.0) {

        /* Case alpha == 1. */

        if (beta_i == 0.0) {
          /* Case alpha = 1, beta = 0.  We compute  C <--- A * B   or  B * A */
          for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
            for (j = 0, cij = ci, bj = 0; j < n_i;
                 j++, cij += inccij, bj += incbj) {

              head_sum = tail_sum = 0.0;

              for (k = 0, aik = ai, bkj = bj; k < i;
                   k++, aik += incaik1, bkj += incbkj) {
                a_elem = a_i[aik];
                b_elem = b_i[bkj];
                {
                  double dt = (double) b_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
              }
              for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
                a_elem = a_i[aik];
                b_elem = b_i[bkj];
                {
                  double dt = (double) b_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
              }
              c_i[cij] = head_sum;
            }
          }
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  C  <--- A * B + beta * C
             or  C  <--- B * A + beta * C  */

          for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
            for (j = 0, cij = ci, bj = 0; j < n_i;
                 j++, cij += inccij, bj += incbj) {

              head_sum = tail_sum = 0.0;

              for (k = 0, aik = ai, bkj = bj; k < i;
                   k++, aik += incaik1, bkj += incbkj) {
                a_elem = a_i[aik];
                b_elem = b_i[bkj];
                {
                  double dt = (double) b_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
              }
              for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
                a_elem = a_i[aik];
                b_elem = b_i[bkj];
                {
                  double dt = (double) b_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
              }
              c_elem = c_i[cij];
              compute_doubledouble_eq_double_mul_double(&head_tmp2, &tail_tmp2, c_elem, beta_i);
              head_tmp1 = head_sum;
              tail_tmp1 = tail_sum;
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp2, tail_tmp2, head_tmp1, tail_tmp1);
              c_i[cij] = head_tmp1;
            }
          }
        }

      } else {
        /* The most general form,   C <--- alpha * A * B + beta * C
           or   C <--- alpha * B * A + beta * C  */

        for (i = 0, ci = 0, ai = 0; i < m_i; i++, ci += incci, ai += incai) {
          for (j = 0, cij = ci, bj = 0; j < n_i;
               j++, cij += inccij, bj += incbj) {

            head_sum = tail_sum = 0.0;

            for (k = 0, aik = ai, bkj = bj; k < i;
                 k++, aik += incaik1, bkj += incbkj) {
              a_elem = a_i[aik];
              b_elem = b_i[bkj];
              {
                double dt = (double) b_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
            }
            for (; k < m_i; k++, aik += incaik2, bkj += incbkj) {
              a_elem = a_i[aik];
              b_elem = b_i[bkj];
              {
                double dt = (double) b_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
            }
            compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_sum, tail_sum, alpha_i);
            c_elem = c_i[cij];
            compute_doubledouble_eq_double_mul_double(&head_tmp2, &tail_tmp2, c_elem, beta_i);
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
            c_i[cij] = head_tmp1;
          }
        }
      }

      FPU_FIX_STOP;

      break;
    }
  }
} /* end XBLAS::symm_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_SYMM_HPP
