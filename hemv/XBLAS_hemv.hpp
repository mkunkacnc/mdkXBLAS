#ifndef XBLAS_HEMV_HPP
#define XBLAS_HEMV_HPP

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
constexpr void hemv(blas_order_type order,
                    blas_uplo_type uplo,
                    IdxType n,
                    T alpha,
                    const A *a,
                    IdxType lda,
                    const X *x,
                    IdxType incx,
                    T beta,
                    T *y,
                    IdxType incy)
/*
 * Purpose
 * =======
 *
 * This routines computes the matrix product:
 *
 *     y  <-  alpha * A * x  +  beta * y
 *
 * where A is a Hermitian matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Storage format of input hermitian matrix A.
 *
 * uplo   (input) blas_uplo_type
 *        Determines which half of matrix A (upper or lower triangle)
 *          is accessed.
 *
 * n      (input) IdxType
 *        Dimension of A and size of vectors x, y.
 *
 * alpha  (input) T
 *
 * a      (input) const A*
 *        Matrix A.
 *
 * lda    (input) IdxType
 *        Leading dimension of matrix A.
 *
 * x      (input) const X*
 *        Vector x.
 *
 * incx   (input) IdxType
 *        Stride for vector x.
 *
 * beta   (input) T
 *
 * y      (input/output) T*
 *        Vector y.
 *
 * incy   (input) IdxType
 *        Stride for vector y.
 *
 */
{
  /* Routine name */
  static const char routine_name[] = "XBLAS::hemv";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  /* Integer Index Variables */
  IdxType i, k;

  IdxType xi, yi;
  IdxType aik, astarti, x_starti, y_starti;

  //IdxType incai;
  IdxType incaik, incaik2;

  IdxType n_i;

  /* Input Matrices */
  const A *a_i = a;
  const X *x_i = x;

  /* Output Vector */
  T *y_i = y;

  /* Input Scalars */
  T alpha_i = alpha;
  T beta_i = beta;

  /* Temporary Floating-Point Variables */
  A a_elem;
  X x_elem;
  T y_elem;
  PrdType prod;
  PrdType sum;
  TmpType tmp1;
  TmpType tmp2;

  /* Test for no-op */
  if (n <= 0) {
    return;
  }
  if (alpha_i == T(0) && beta_i == T(1)) {
    return;
  }

  /* Check for error conditions. */
  if (order != blas_colmajor && order != blas_rowmajor) {
    BLAS_error(routine_name, -1, order, nullptr);
  }
  if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, nullptr);
  }
  if (lda < n) {
    BLAS_error(routine_name, -3, n, nullptr);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -8, incx, nullptr);
  }
  if (incy == 0) {
    BLAS_error(routine_name, -11, incy, nullptr);
  }

  /* Set Index Parameters */
  n_i = n;

  if ((order == blas_colmajor && uplo == blas_upper) ||
      (order == blas_rowmajor && uplo == blas_lower)) {
    //incai = lda;
    incaik = 1;
    incaik2 = lda;
  } else {
    //incai = 1;
    incaik = lda;
    incaik2 = 1;
  }

  if (incx < 0) {
    x_starti = (-n_i + 1) * incx;
  } else {
    x_starti = 0;
  }
  if (incy < 0) {
    y_starti = (-n_i + 1) * incy;
  } else {
    y_starti = 0;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  /* alpha = 0.  In this case, just return beta * y */
  if (alpha_i == T(0)) {
    for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
      y_elem = y_i[yi];
      tmp1 = impl::mul<TmpType>(y_elem, beta_i);
      y_i[yi] = impl::to<T>(tmp1);
    }
  } else {
    /*  determine whether we conjugate in first loop or second loop */
    if (uplo == blas_lower) {
      /*  conjugate second */

      /* Case alpha == 1. */
      if (alpha_i == T(1)) {
        if (beta_i == T(0)) {
          /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incaik2) {
            sum = impl::zero_v<PrdType>;
            for (k = 0, aik = astarti, xi = x_starti;
                 k < i; k++, aik += incaik, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            auto a_elem_r = std::real(a_i[aik]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem_r, x_elem);
            sum = sum + prod;
            k++;
            aik += incaik2;
            xi += incx;
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = impl::Conj::func(a_i[aik]);
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            y_i[yi] = impl::to<T>(sum);
          }
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  y  <--- A * x + beta * y */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incaik2) {
            sum = impl::zero_v<PrdType>;
            for (k = 0, aik = astarti, xi = x_starti;
                 k < i; k++, aik += incaik, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            auto a_elem_r = std::real(a_i[aik]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem_r, x_elem);
            sum = sum + prod;
            k++;
            aik += incaik2;
            xi += incx;
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = impl::Conj::func(a_i[aik]);
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            y_elem = y_i[yi];
            tmp2 = impl::mul<TmpType>(y_elem, beta_i);
            tmp1 = sum;
            tmp1 = tmp2 + tmp1;
            y_i[yi] = impl::to<T>(tmp1);
          }
        }
      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        for (i = 0, yi = y_starti, astarti = 0;
             i < n_i; i++, yi += incy, astarti += incaik2) {
          sum = impl::zero_v<PrdType>;
          for (k = 0, aik = astarti, xi = x_starti;
               k < i; k++, aik += incaik, xi += incx) {
            a_elem = a_i[aik];
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          auto a_elem_r = std::real(a_i[aik]);
          x_elem = x_i[xi];
          prod = impl::mul<PrdType>(a_elem_r, x_elem);
          sum = sum + prod;
          k++;
          aik += incaik2;
          xi += incx;
          for (; k < n_i; k++, aik += incaik2, xi += incx) {
            a_elem = impl::Conj::func(a_i[aik]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          y_elem = y_i[yi];
          tmp2 = impl::mul<TmpType>(y_elem, beta_i);
          tmp1 = impl::mul<TmpType>(sum, alpha_i);
          tmp1 = tmp1 + tmp2;
          y_i[yi] = impl::to<T>(tmp1);
        }
      }
    } else {
      /*  conjugate first loop */

      /* Case alpha == 1. */
      if (alpha_i == T(1)) {
        if (beta_i == T(0)) {
          /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incaik2) {
            sum = impl::zero_v<PrdType>;
            for (k = 0, aik = astarti, xi = x_starti;
                 k < i; k++, aik += incaik, xi += incx) {
              a_elem = impl::Conj::func(a_i[aik]);
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            auto a_elem_r = std::real(a_i[aik]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem_r, x_elem);
            sum = sum + prod;
            k++;
            aik += incaik2;
            xi += incx;
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            y_i[yi] = impl::to<T>(sum);
          }
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  y  <--- A * x + beta * y */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incaik2) {
            sum = impl::zero_v<PrdType>;
            for (k = 0, aik = astarti, xi = x_starti;
                 k < i; k++, aik += incaik, xi += incx) {
              a_elem = impl::Conj::func(a_i[aik]);
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            auto a_elem_r = std::real(a_i[aik]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem_r, x_elem);
            sum = sum + prod;
            k++;
            aik += incaik2;
            xi += incx;
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            y_elem = y_i[yi];
            tmp2 = impl::mul<TmpType>(y_elem, beta_i);
            tmp1 = sum;
            tmp1 = tmp2 + tmp1;
            y_i[yi] = impl::to<T>(tmp1);
          }
        }
      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        for (i = 0, yi = y_starti, astarti = 0;
             i < n_i; i++, yi += incy, astarti += incaik2) {
          sum = impl::zero_v<PrdType>;
          for (k = 0, aik = astarti, xi = x_starti;
               k < i; k++, aik += incaik, xi += incx) {
            a_elem = impl::Conj::func(a_i[aik]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          auto a_elem_r = std::real(a_i[aik]);
          x_elem = x_i[xi];
          prod = impl::mul<PrdType>(a_elem_r, x_elem);
          sum = sum + prod;
          k++;
          aik += incaik2;
          xi += incx;
          for (; k < n_i; k++, aik += incaik2, xi += incx) {
            a_elem = a_i[aik];
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          y_elem = y_i[yi];
          tmp2 = impl::mul<TmpType>(y_elem, beta_i);
          tmp1 = impl::mul<TmpType>(sum, alpha_i);
          tmp1 = tmp1 + tmp2;
          y_i[yi] = impl::to<T>(tmp1);
        }
      }
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::hemv */

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
constexpr void hemv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      IdxType n,
                      T alpha,
                      const A *a,
                      IdxType lda,
                      const X *x,
                      IdxType incx,
                      T beta,
                      T *y,
                      IdxType incy,
                      blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routines computes the matrix product:
 *
 *     y  <-  alpha * A * x  +  beta * y
 *
 * where A is a Hermitian matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Storage format of input hermitian matrix A.
 *
 * uplo   (input) blas_uplo_type
 *        Determines which half of matrix A (upper or lower triangle)
 *          is accessed.
 *
 * n      (input) IdxType
 *        Dimension of A and size of vectors x, y.
 *
 * alpha  (input) T
 *
 * a      (input) const A*
 *        Matrix A.
 *
 * lda    (input) IdxType
 *        Leading dimension of matrix A.
 *
 * x      (input) const X*
 *        Vector x.
 *
 * incx   (input) IdxType
 *        Stride for vector x.
 *
 * beta   (input) T
 *
 * y      (input/output) T*
 *        Vector y.
 *
 * incy   (input) IdxType
 *        Stride for vector y.
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
//static const char routine_name[] = "XBLAS::hemv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::hemv<T, A, X, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::hemv<T, A, X, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::hemv<T, A, X, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::hemv<T, A, X, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  }
} /* end XBLAS::hemv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_HEMV_HPP
