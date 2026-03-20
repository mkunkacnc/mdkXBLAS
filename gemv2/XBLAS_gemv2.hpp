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
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void gemv2(blas_order_type order,
                     blas_trans_type trans,
                     N m,
                     N n,
                     T alpha,
                     const A *a,
                     N lda,
                     const X *head_x,
                     const X *tail_x,
                     N incx,
                     T beta,
                     T *y,
                     N incy)
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
 * m       (input) N
 *         Dimension of A
 *
 * n       (input) N
 *         Dimension of A and the length of vector x and z
 *
 * alpha   (input) T
 *
 * A       (input) const A*
 *
 * lda     (input) N
 *         Leading dimension of A
 *
 * head_x
 * tail_x  (input) const X*
 *
 * incx    (input) N
 *         The stride for vector x.
 *
 * beta    (input) T
 *
 * y       (input/output) T*
 *
 * incy    (input) N
 *         The stride for vector y.
 *
 */
{
  static const char *routine_name = "XBLAS::gemv2";

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
    BLAS_error(routine_name, -3, m, nullptr);
  else if (n <= 0)
    BLAS_error(routine_name, -4, n, nullptr);
  else if (incx == 0)
    BLAS_error(routine_name, -10, incx, nullptr);
  else if (incy == 0)
    BLAS_error(routine_name, -13, incy, nullptr);

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
    BLAS_error(routine_name, -7, lda, nullptr);

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
  if (alpha == T(0)) {
    if (beta == T(0)) {
      iy = ky;
      for (i = 0; i < leny; i++) {
        y_i[iy] = T(0);
        iy += incy;
      }
    } else if (beta != T(0)) {
      iy = ky;
      for (i = 0; i < leny; i++) {
        y_elem = y_i[iy];
        tmp1 = impl::mul<TmpType>(y_elem, beta);
        y_i[iy] = impl::to<T>(tmp1);
        iy += incy;
      }
    }
  } else { /* alpha != 0 */
    if constexpr (impl::is_complex_v<A>) {
      if (trans == blas_conj_trans) {
        /* if beta = 0, we can save m multiplies:
           y = alpha*A*head_x + alpha*A*tail_x  */
        if (beta == T(0)) {
          if (alpha == T(1)) {
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
              tmp1 = impl::mul<TmpType>(sum, alpha);
              tmp2 = impl::mul<TmpType>(sum2, alpha);
              tmp1 = tmp1 + tmp2;
              y_i[iy] = impl::to<T>(tmp1);
              ai += incai;
              iy += incy;
            }
          }
        } else { /* beta != 0 */
          if (alpha == T(1)) {
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
              tmp1 = impl::mul<TmpType>(y_elem, beta);
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
              tmp1 = impl::mul<TmpType>(sum, alpha);
              tmp2 = impl::mul<TmpType>(sum2, alpha);
              tmp1 = tmp1 + tmp2;
              y_elem = y_i[iy];
              tmp2 = impl::mul<TmpType>(y_elem, beta);
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
        if (beta == T(0)) {
          if (alpha == T(1)) {
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
              tmp1 = impl::mul<TmpType>(sum, alpha);
              tmp2 = impl::mul<TmpType>(sum2, alpha);
              tmp1 = tmp1 + tmp2;
              y_i[iy] = impl::to<T>(tmp1);
              ai += incai;
              iy += incy;
            }
          }
        } else { /* beta != 0 */
          if (alpha == T(1)) {
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
              tmp1 = impl::mul<TmpType>(y_elem, beta);
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
              tmp1 = impl::mul<TmpType>(sum, alpha);
              tmp2 = impl::mul<TmpType>(sum2, alpha);
              tmp1 = tmp1 + tmp2;
              y_elem = y_i[iy];
              tmp2 = impl::mul<TmpType>(y_elem, beta);
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
      if (beta == T(0)) {
        if (alpha == T(1)) {
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
            tmp1 = impl::mul<TmpType>(sum, alpha);
            tmp2 = impl::mul<TmpType>(sum2, alpha);
            tmp1 = tmp1 + tmp2;
            y_i[iy] = impl::to<T>(tmp1);
            ai += incai;
            iy += incy;
          }
        }
      } else { /* beta != 0 */
        if (alpha == T(1)) {
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
            tmp1 = impl::mul<TmpType>(y_elem, beta);
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
            tmp1 = impl::mul<TmpType>(sum, alpha);
            tmp2 = impl::mul<TmpType>(sum2, alpha);
            tmp1 = tmp1 + tmp2;
            y_elem = y_i[iy];
            tmp2 = impl::mul<TmpType>(y_elem, beta);
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
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void gemv2_x(blas_order_type order,
                       blas_trans_type trans,
                       N m,
                       N n,
                       T alpha,
                       const A *a,
                       N lda,
                       const X *head_x,
                       const X *tail_x,
                       N incx,
                       T beta,
                       T *y,
                       N incy,
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
 * m       (input) N
 *         Dimension of A
 *
 * n       (input) N
 *         Dimension of A and the length of vector x and z
 *
 * alpha   (input) T
 *
 * A       (input) const A*
 *
 * lda     (input) N
 *         Leading dimension of A
 *
 * head_x
 * tail_x  (input) const X*
 *
 * incx    (input) N
 *         The stride for vector x.
 *
 * beta    (input) T
 *
 * y       (input/output) T*
 *
 * incy    (input) N
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
  static const char *routine_name = "XBLAS::gemv2_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::gemv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, trans, m, n, alpha, a, lda, head_x, tail_x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::gemv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, trans, m, n, alpha, a, lda, head_x, tail_x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::gemv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, trans, m, n, alpha, a, lda, head_x, tail_x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::gemv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, trans, m, n, alpha, a, lda, head_x, tail_x, incx, beta, y, incy);
    break;
  default:
    BLAS_error(routine_name, -14, prec, nullptr);
    break;
  }
} /* end XBLAS::gemv2_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_GEMV2_HPP
