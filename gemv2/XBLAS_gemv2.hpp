#ifndef XBLAS_GEMV2_HPP
#define XBLAS_GEMV2_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename A,
         typename X,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void gemv2(blas_order_type order,
                     blas_trans_type trans,
                     IdxType m,
                     IdxType n,
                     T alpha,
                     const A *a,
                     IdxType lda,
                     const X *head_x,
                     const X *tail_x,
                     IdxType incx,
                     T beta,
                     T *y,
                     IdxType incy)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * op(A) * head_x + alpha * op(A) * tail_x + beta * y,
 * where A is a general matrix.
 *
 * Arguments
 * =========
 *
 * order   (input) blas_order_type
 *         Order of A; row or column major
 *
 * trans   (input) blas_trans_type
 *         Transpose of A: no trans, trans, or conjugate trans
 *
 * m       (input) IdxType
 *         Dimension of A
 *
 * n       (input) IdxType
 *         Dimension of A and the length of vector x and z
 *
 * alpha   (input) T
 *
 * A       (input) const A*
 *
 * lda     (input) IdxType
 *         Leading dimension of A
 *
 * head_x
 * tail_x  (input) const X*
 *
 * incx    (input) IdxType
 *         The stride for vector x.
 *
 * beta    (input) T
 *
 * y       (input/output) T*
 *
 * incy    (input) IdxType
 *         The stride for vector y.
 *
 */
{
  static const char routine_name[] = "XBLAS::gemv2";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  IdxType i, j;
  IdxType iy, jx, kx, ky;
  IdxType lenx, leny;
  IdxType ai, aij;
  IdxType incai, incaij;

  const A *a_i = a;
  const X *head_x_i = head_x;
  const X *tail_x_i = tail_x;
  T *y_i = y;
  T alpha_i = alpha;
  T beta_i = beta;
  A a_elem;
  X x_elem;
  T y_elem;
  PrdType prod;
  PrdType sum;
  PrdType sum2;
  TmpType tmp1;
  TmpType tmp2;

  /* all error calls */
  if (m < 0)
    BLAS_error(routine_name, -3, m, 0);
  else if (n <= 0)
    BLAS_error(routine_name, -4, n, 0);
  else if (incx == 0)
    BLAS_error(routine_name, -10, incx, 0);
  else if (incy == 0)
    BLAS_error(routine_name, -13, incy, 0);

  if ((order == blas_rowmajor) && (trans == blas_no_trans)) {
    lenx = n;
    leny = m;
    incai = lda;
    incaij = 1;
  } else if ((order == blas_rowmajor) && (trans != blas_no_trans)) {
    lenx = m;
    leny = n;
    incai = 1;
    incaij = lda;
  } else if ((order == blas_colmajor) && (trans == blas_no_trans)) {
    lenx = n;
    leny = m;
    incai = 1;
    incaij = lda;
  } else { /* colmajor and blas_trans */
    lenx = m;
    leny = n;
    incai = lda;
    incaij = 1;
  }

  if (lda < leny)
    BLAS_error(routine_name, -7, lda, NULL);

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if (incx > 0)
    kx = 0;
  else
    kx = (1 - lenx) * incx;
  if (incy > 0)
    ky = 0;
  else
    ky = (1 - leny) * incy;

  /* No extra-precision needed for alpha = 0 */
  if (alpha_i == T(0)) {
    if (beta_i == T(0)) {
      iy = ky;
      for (i = 0; i < leny; i++) {
        y_i[iy] = T(0);
        iy += incy;
      }
    } else if (beta_i != T(0)) {
      iy = ky;
      for (i = 0; i < leny; i++) {
        y_elem = y_i[iy];
        tmp1 = impl::mul<TmpType>(y_elem, beta_i);
        y_i[iy] = impl::to<T>(tmp1);
        iy += incy;
      }
    }
  } else { /* alpha != 0 */
    if constexpr (impl::is_complex_v<A>) {
      if (trans == blas_conj_trans) {
        /* if beta = 0, we can save m multiplies:
           y = alpha*A*head_x + alpha*A*tail_x  */
        if (beta_i == T(0)) {
          if (alpha_i == T(1)) {
            /* save m more multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = impl::zero_v<PrdType>;
              sum2 = impl::zero_v<PrdType>;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = impl::Conj::func(a_i[aij]);
                x_elem = head_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              sum = sum + sum2;
              y_i[iy] = impl::to<T>(sum);
              ai += incai;
              iy += incy;
            } /* end for */
          } else {
            /* alpha != 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = impl::zero_v<PrdType>;
              sum2 = impl::zero_v<PrdType>;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = impl::Conj::func(a_i[aij]);
                x_elem = head_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              tmp1 = impl::mul<TmpType>(sum, alpha_i);
              tmp2 = impl::mul<TmpType>(sum2, alpha_i);
              tmp1 = tmp1 + tmp2;
              y_i[iy] = impl::to<T>(tmp1);
              ai += incai;
              iy += incy;
            }
          }
        } else { /* beta != 0 */
          if (alpha_i == T(1)) {
            /* save m multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = impl::zero_v<PrdType>;
              sum2 = impl::zero_v<PrdType>;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = impl::Conj::func(a_i[aij]);
                x_elem = head_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              sum = sum + sum2;
              y_elem = y_i[iy];
              tmp1 = impl::mul<TmpType>(y_elem, beta_i);
              tmp2 = sum + tmp1;
              y_i[iy] = impl::to<T>(tmp2);
              ai += incai;
              iy += incy;
            }
          } else {
            /* alpha != 1, the most general form:
               y = alpha*A*head_x + alpha*A*tail_x + beta*y */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = impl::zero_v<PrdType>;
              sum2 = impl::zero_v<PrdType>;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = impl::Conj::func(a_i[aij]);
                x_elem = head_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              tmp1 = impl::mul<TmpType>(sum, alpha_i);
              tmp2 = impl::mul<TmpType>(sum2, alpha_i);
              tmp1 = tmp1 + tmp2;
              y_elem = y_i[iy];
              tmp2 = impl::mul<TmpType>(y_elem, beta_i);
              tmp1 = tmp1 + tmp2;
              y_i[iy] = impl::to<T>(tmp1);
              ai += incai;
              iy += incy;
            }
          }
        }
      } else { // non-conj
        /* if beta = 0, we can save m multiplies:
           y = alpha*A*head_x + alpha*A*tail_x  */
        if (beta_i == T(0)) {
          if (alpha_i == T(1)) {
            /* save m more multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = impl::zero_v<PrdType>;
              sum2 = impl::zero_v<PrdType>;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];
                x_elem = head_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              sum = sum + sum2;
              y_i[iy] = impl::to<T>(sum);
              ai += incai;
              iy += incy;
            } /* end for */
          } else {
            /* alpha != 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = impl::zero_v<PrdType>;
              sum2 = impl::zero_v<PrdType>;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];
                x_elem = head_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              tmp1 = impl::mul<TmpType>(sum, alpha_i);
              tmp2 = impl::mul<TmpType>(sum2, alpha_i);
              tmp1 = tmp1 + tmp2;
              y_i[iy] = impl::to<T>(tmp1);
              ai += incai;
              iy += incy;
            }
          }
        } else { /* beta != 0 */
          if (alpha_i == T(1)) {
            /* save m multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = impl::zero_v<PrdType>;
              sum2 = impl::zero_v<PrdType>;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];
                x_elem = head_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              sum = sum + sum2;
              y_elem = y_i[iy];
              tmp1 = impl::mul<TmpType>(y_elem, beta_i);
              tmp2 = sum + tmp1;
              y_i[iy] = impl::to<T>(tmp2);
              ai += incai;
              iy += incy;
            }
          } else {
            /* alpha != 1, the most general form:
               y = alpha*A*head_x + alpha*A*tail_x + beta*y */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = impl::zero_v<PrdType>;
              sum2 = impl::zero_v<PrdType>;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];
                x_elem = head_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = impl::mul<PrdType>(a_elem, x_elem);
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              tmp1 = impl::mul<TmpType>(sum, alpha_i);
              tmp2 = impl::mul<TmpType>(sum2, alpha_i);
              tmp1 = tmp1 + tmp2;
              y_elem = y_i[iy];
              tmp2 = impl::mul<TmpType>(y_elem, beta_i);
              tmp1 = tmp1 + tmp2;
              y_i[iy] = impl::to<T>(tmp1);
              ai += incai;
              iy += incy;
            }
          }
        }
      }
    } else { // non-complex
      /* if beta = 0, we can save m multiplies:
         y = alpha*A*head_x + alpha*A*tail_x  */
      if (beta_i == T(0)) {
        if (alpha_i == T(1)) {
          /* save m more multiplies if alpha = 1 */
          ai = 0;
          iy = ky;
          for (i = 0; i < leny; i++) {
            sum = impl::zero_v<PrdType>;
            sum2 = impl::zero_v<PrdType>;
            aij = ai;
            jx = kx;
            for (j = 0; j < lenx; j++) {
              a_elem = a_i[aij];
              x_elem = head_x_i[jx];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
              x_elem = tail_x_i[jx];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum2 = sum2 + prod;
              aij += incaij;
              jx += incx;
            }
            sum = sum + sum2;
            y_i[iy] = impl::to<T>(sum);
            ai += incai;
            iy += incy;
          } /* end for */
        } else {
          /* alpha != 1 */
          ai = 0;
          iy = ky;
          for (i = 0; i < leny; i++) {
            sum = impl::zero_v<PrdType>;
            sum2 = impl::zero_v<PrdType>;
            aij = ai;
            jx = kx;
            for (j = 0; j < lenx; j++) {
              a_elem = a_i[aij];
              x_elem = head_x_i[jx];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
              x_elem = tail_x_i[jx];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum2 = sum2 + prod;
              aij += incaij;
              jx += incx;
            }
            tmp1 = impl::mul<TmpType>(sum, alpha_i);
            tmp2 = impl::mul<TmpType>(sum2, alpha_i);
            tmp1 = tmp1 + tmp2;
            y_i[iy] = impl::to<T>(tmp1);
            ai += incai;
            iy += incy;
          }
        }
      } else { /* beta != 0 */
        if (alpha_i == T(1)) {
          /* save m multiplies if alpha = 1 */
          ai = 0;
          iy = ky;
          for (i = 0; i < leny; i++) {
            sum = impl::zero_v<PrdType>;
            sum2 = impl::zero_v<PrdType>;
            aij = ai;
            jx = kx;
            for (j = 0; j < lenx; j++) {
              a_elem = a_i[aij];
              x_elem = head_x_i[jx];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
              x_elem = tail_x_i[jx];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum2 = sum2 + prod;
              aij += incaij;
              jx += incx;
            }
            sum = sum + sum2;
            y_elem = y_i[iy];
            tmp1 = impl::mul<TmpType>(y_elem, beta_i);
            tmp2 = sum + tmp1;
            y_i[iy] = impl::to<T>(tmp2);
            ai += incai;
            iy += incy;
          }
        } else {
          /* alpha != 1, the most general form:
             y = alpha*A*head_x + alpha*A*tail_x + beta*y */
          ai = 0;
          iy = ky;
          for (i = 0; i < leny; i++) {
            sum = impl::zero_v<PrdType>;
            sum2 = impl::zero_v<PrdType>;
            aij = ai;
            jx = kx;
            for (j = 0; j < lenx; j++) {
              a_elem = a_i[aij];
              x_elem = head_x_i[jx];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
              x_elem = tail_x_i[jx];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum2 = sum2 + prod;
              aij += incaij;
              jx += incx;
            }
            tmp1 = impl::mul<TmpType>(sum, alpha_i);
            tmp2 = impl::mul<TmpType>(sum2, alpha_i);
            tmp1 = tmp1 + tmp2;
            y_elem = y_i[iy];
            tmp2 = impl::mul<TmpType>(y_elem, beta_i);
            tmp1 = tmp1 + tmp2;
            y_i[iy] = impl::to<T>(tmp1);
            ai += incai;
            iy += incy;
          }
        }
      }
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::gemv2 */

//-----------------

template<typename T,
         typename A,
         typename X,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void gemv2_x(blas_order_type order,
                       blas_trans_type trans,
                       IdxType m,
                       IdxType n,
                       T alpha,
                       const A *a,
                       IdxType lda,
                       const X *head_x,
                       const X *tail_x,
                       IdxType incx,
                       T beta,
                       T *y,
                       IdxType incy,
                       blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * op(A) * head_x + alpha * op(A) * tail_x + beta * y,
 * where A is a general matrix.
 *
 * Arguments
 * =========
 *
 * order   (input) blas_order_type
 *         Order of A; row or column major
 *
 * trans   (input) blas_trans_type
 *         Transpose of A: no trans, trans, or conjugate trans
 *
 * m       (input) IdxType
 *         Dimension of A
 *
 * n       (input) IdxType
 *         Dimension of A and the length of vector x and z
 *
 * alpha   (input) T
 *
 * A       (input) const A*
 *
 * lda     (input) IdxType
 *         Leading dimension of A
 *
 * head_x
 * tail_x  (input) const X*
 *
 * incx    (input) IdxType
 *         The stride for vector x.
 *
 * beta    (input) T
 *
 * y       (input/output) T*
 *
 * incy    (input) IdxType
 *         The stride for vector y.
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
  static const char routine_name[] = "BLAS_dgemv2_d_s_x";
  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, j;
      int iy, jx, kx, ky;
      int lenx, leny;
      int ai, aij;
      int incai, incaij;

      const double *a_i = a;
      const float *head_x_i = head_x;
      const float *tail_x_i = tail_x;
      double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      double a_elem;
      float x_elem;
      double y_elem;
      double prod;
      double sum;
      double sum2;
      double tmp1;
      double tmp2;


      /* all error calls */
      if (m < 0)
        BLAS_error(routine_name, -3, m, 0);
      else if (n <= 0)
        BLAS_error(routine_name, -4, n, 0);
      else if (incx == 0)
        BLAS_error(routine_name, -10, incx, 0);
      else if (incy == 0)
        BLAS_error(routine_name, -13, incy, 0);

      if ((order == blas_rowmajor) && (trans == blas_no_trans)) {
        lenx = n;
        leny = m;
        incai = lda;
        incaij = 1;
      } else if ((order == blas_rowmajor) && (trans != blas_no_trans)) {
        lenx = m;
        leny = n;
        incai = 1;
        incaij = lda;
      } else if ((order == blas_colmajor) && (trans == blas_no_trans)) {
        lenx = n;
        leny = m;
        incai = 1;
        incaij = lda;
      } else {                        /* colmajor and blas_trans */
        lenx = m;
        leny = n;
        incai = lda;
        incaij = 1;
      }

      if (lda < leny)
        BLAS_error(routine_name, -7, lda, NULL);








      if (incx > 0)
        kx = 0;
      else
        kx = (1 - lenx) * incx;
      if (incy > 0)
        ky = 0;
      else
        ky = (1 - leny) * incy;

      /* No extra-precision needed for alpha = 0 */
      if (alpha_i == 0.0) {
        if (beta_i == 0.0) {
          iy = ky;
          for (i = 0; i < leny; i++) {
            y_i[iy] = 0.0;
            iy += incy;
          }
        } else if (!(beta_i == 0.0)) {
          iy = ky;
          for (i = 0; i < leny; i++) {
            y_elem = y_i[iy];
            tmp1 = y_elem * beta_i;
            y_i[iy] = tmp1;
            iy += incy;
          }
        }
      } else {                        /* alpha != 0 */

        /* if beta = 0, we can save m multiplies:
           y = alpha*A*head_x + alpha*A*tail_x  */
        if (beta_i == 0.0) {
          if (alpha_i == 1.0) {
            /* save m more multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = 0.0;
              sum2 = 0.0;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                prod = a_elem * x_elem;
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = a_elem * x_elem;
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              sum = sum + sum2;
              y_i[iy] = sum;
              ai += incai;
              iy += incy;
            }                        /* end for */
          } else {                /* alpha != 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = 0.0;
              sum2 = 0.0;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                prod = a_elem * x_elem;
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = a_elem * x_elem;
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              tmp1 = sum * alpha_i;
              tmp2 = sum2 * alpha_i;
              tmp1 = tmp1 + tmp2;
              y_i[iy] = tmp1;
              ai += incai;
              iy += incy;
            }
          }
        } else {                /* beta != 0 */
          if (alpha_i == 1.0) {
            /* save m multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = 0.0;;
              sum2 = 0.0;;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                prod = a_elem * x_elem;
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = a_elem * x_elem;
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              sum = sum + sum2;
              y_elem = y_i[iy];
              tmp1 = y_elem * beta_i;
              tmp2 = sum + tmp1;
              y_i[iy] = tmp2;
              ai += incai;
              iy += incy;
            }
          } else {                /* alpha != 1, the most general form:
                                   y = alpha*A*head_x + alpha*A*tail_x + beta*y */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = 0.0;;
              sum2 = 0.0;;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                prod = a_elem * x_elem;
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = a_elem * x_elem;
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              tmp1 = sum * alpha_i;
              tmp2 = sum2 * alpha_i;
              tmp1 = tmp1 + tmp2;
              y_elem = y_i[iy];
              tmp2 = y_elem * beta_i;
              tmp1 = tmp1 + tmp2;
              y_i[iy] = tmp1;
              ai += incai;
              iy += incy;
            }
          }
        }

      }



      break;
    }
  case blas_prec_extra:{

      int i, j;
      int iy, jx, kx, ky;
      int lenx, leny;
      int ai, aij;
      int incai, incaij;

      const double *a_i = a;
      const float *head_x_i = head_x;
      const float *tail_x_i = tail_x;
      double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      double a_elem;
      float x_elem;
      double y_elem;
      double head_prod, tail_prod;
      double head_sum, tail_sum;
      double head_sum2, tail_sum2;
      double head_tmp1, tail_tmp1;
      double head_tmp2, tail_tmp2;
      FPU_FIX_DECL;

      /* all error calls */
      if (m < 0)
        BLAS_error(routine_name, -3, m, 0);
      else if (n <= 0)
        BLAS_error(routine_name, -4, n, 0);
      else if (incx == 0)
        BLAS_error(routine_name, -10, incx, 0);
      else if (incy == 0)
        BLAS_error(routine_name, -13, incy, 0);

      if ((order == blas_rowmajor) && (trans == blas_no_trans)) {
        lenx = n;
        leny = m;
        incai = lda;
        incaij = 1;
      } else if ((order == blas_rowmajor) && (trans != blas_no_trans)) {
        lenx = m;
        leny = n;
        incai = 1;
        incaij = lda;
      } else if ((order == blas_colmajor) && (trans == blas_no_trans)) {
        lenx = n;
        leny = m;
        incai = 1;
        incaij = lda;
      } else {                        /* colmajor and blas_trans */
        lenx = m;
        leny = n;
        incai = lda;
        incaij = 1;
      }

      if (lda < leny)
        BLAS_error(routine_name, -7, lda, NULL);

      FPU_FIX_START;






      if (incx > 0)
        kx = 0;
      else
        kx = (1 - lenx) * incx;
      if (incy > 0)
        ky = 0;
      else
        ky = (1 - leny) * incy;

      /* No extra-precision needed for alpha = 0 */
      if (alpha_i == 0.0) {
        if (beta_i == 0.0) {
          iy = ky;
          for (i = 0; i < leny; i++) {
            y_i[iy] = 0.0;
            iy += incy;
          }
        } else if (!(beta_i == 0.0)) {
          iy = ky;
          for (i = 0; i < leny; i++) {
            y_elem = y_i[iy];
            compute_doubledouble_eq_double_mul_double(&head_tmp1, &tail_tmp1, y_elem, beta_i);
            y_i[iy] = head_tmp1;
            iy += incy;
          }
        }
      } else {                        /* alpha != 0 */

        /* if beta = 0, we can save m multiplies:
           y = alpha*A*head_x + alpha*A*tail_x  */
        if (beta_i == 0.0) {
          if (alpha_i == 1.0) {
            /* save m more multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              head_sum = tail_sum = 0.0;
              head_sum2 = tail_sum2 = 0.0;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                {
                  double dt = (double) x_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
                x_elem = tail_x_i[jx];
                {
                  double dt = (double) x_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod, tail_prod);
                aij += incaij;
                jx += incx;
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_sum2, tail_sum2);
              y_i[iy] = head_sum;
              ai += incai;
              iy += incy;
            }                        /* end for */
          } else {                /* alpha != 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              head_sum = tail_sum = 0.0;
              head_sum2 = tail_sum2 = 0.0;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                {
                  double dt = (double) x_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
                x_elem = tail_x_i[jx];
                {
                  double dt = (double) x_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod, tail_prod);
                aij += incaij;
                jx += incx;
              }
              compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_sum, tail_sum, alpha_i);
              compute_doubledouble_eq_doubledouble_mul_double(&head_tmp2, &tail_tmp2, head_sum2, tail_sum2, alpha_i);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
              y_i[iy] = head_tmp1;
              ai += incai;
              iy += incy;
            }
          }
        } else {                /* beta != 0 */
          if (alpha_i == 1.0) {
            /* save m multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              head_sum = tail_sum = 0.0;;
              head_sum2 = tail_sum2 = 0.0;;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                {
                  double dt = (double) x_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
                x_elem = tail_x_i[jx];
                {
                  double dt = (double) x_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod, tail_prod);
                aij += incaij;
                jx += incx;
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_sum2, tail_sum2);
              y_elem = y_i[iy];
              compute_doubledouble_eq_double_mul_double(&head_tmp1, &tail_tmp1, y_elem, beta_i);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp2, &tail_tmp2, head_sum, tail_sum, head_tmp1, tail_tmp1);
              y_i[iy] = head_tmp2;
              ai += incai;
              iy += incy;
            }
          } else {                /* alpha != 1, the most general form:
                                   y = alpha*A*head_x + alpha*A*tail_x + beta*y */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              head_sum = tail_sum = 0.0;;
              head_sum2 = tail_sum2 = 0.0;;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                {
                  double dt = (double) x_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
                x_elem = tail_x_i[jx];
                {
                  double dt = (double) x_elem;
                  compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
                }
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod, tail_prod);
                aij += incaij;
                jx += incx;
              }
              compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_sum, tail_sum, alpha_i);
              compute_doubledouble_eq_doubledouble_mul_double(&head_tmp2, &tail_tmp2, head_sum2, tail_sum2, alpha_i);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
              y_elem = y_i[iy];
              compute_doubledouble_eq_double_mul_double(&head_tmp2, &tail_tmp2, y_elem, beta_i);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
              y_i[iy] = head_tmp1;
              ai += incai;
              iy += incy;
            }
          }
        }

      }

      FPU_FIX_STOP;
    }
    break;
  }
} /* end XBLAS::gemv2_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_GEMV2_HPP
