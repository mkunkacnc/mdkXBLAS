#ifndef XBLAS_SYMV_HPP
#define XBLAS_SYMV_HPP

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
constexpr void symv(blas_order_type order,
                    blas_uplo_type uplo,
                    N n,
                    T alpha,
                    const A *a,
                    N lda,
                    const X *x,
                    N incx,
                    T beta,
                    T *y,
                    N incy)
/*
 * Purpose
 * =======
 *
 * This routines computes the matrix product:
 *
 *     y  <-  alpha * A * x  +  beta * y
 *
 * where A is a Symmetric matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Storage format of input symmetric matrix A.
 *
 * uplo   (input) blas_uplo_type
 *        Determines which half of matrix A (upper or lower triangle)
 *          is accessed.
 *
 * n      (input) N
 *        Dimension of A and size of vectors x, y.
 *
 * alpha  (input) T
 *
 * a      (input) const A*
 *        Matrix A.
 *
 * lda    (input) N
 *        Leading dimension of matrix A.
 *
 * x      (input) const X*
 *        Vector x.
 *
 * incx   (input) N
 *        Stride for vector x.
 *
 * beta   (input) T
 *
 * y      (input/output) T*
 *        Vector y.
 *
 * incy   (input) N
 *        Stride for vector y.
 *
 */
{
  /* Routine name */
  static const char routine_name[] = "XBLAS::symv";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  /* Integer Index Variables */
  IdxType i, k;

  IdxType xi, yi;
  IdxType aik, astarti, x_starti, y_starti;

  IdxType incai;
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
    incai = lda;
    incaik = 1;
    incaik2 = lda;
  } else {
    incai = 1;
    incaik = lda;
    incaik2 = 1;
  }

  /* Adjustment to increments (if any) */





  if (incx < 0) {
    x_starti = (-n + 1) * incx;
  } else {
    x_starti = 0;
  }
  if (incy < 0) {
    y_starti = (-n + 1) * incy;
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
  } else if (alpha_i == T(1)) {

    /* Case alpha == 1. */

    if (beta_i == T(0)) {
      /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
      for (i = 0, yi = y_starti, astarti = 0;
           i < n_i; i++, yi += incy, astarti += incai) {
        sum = impl::zero_v<PrdType>;

        for (k = 0, aik = astarti, xi = x_starti; k < i;
             k++, aik += incaik, xi += incx) {
          a_elem = a_i[aik];
          x_elem = x_i[xi];
          prod = impl::mul<PrdType>(a_elem, x_elem);
          sum = sum + prod;
        }
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
           i < n_i; i++, yi += incy, astarti += incai) {
        sum = impl::zero_v<PrdType>;

        for (k = 0, aik = astarti, xi = x_starti;
             k < i; k++, aik += incaik, xi += incx) {
          a_elem = a_i[aik];
          x_elem = x_i[xi];
          prod = impl::mul<PrdType>(a_elem, x_elem);
          sum = sum + prod;
        }
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
         i < n_i; i++, yi += incy, astarti += incai) {
      sum = impl::zero_v<PrdType>;

      for (k = 0, aik = astarti, xi = x_starti;
           k < i; k++, aik += incaik, xi += incx) {
        a_elem = a_i[aik];
        x_elem = x_i[xi];
        prod = impl::mul<PrdType>(a_elem, x_elem);
        sum = sum + prod;
      }
      for (; k < n_i; k++, aik += incaik2, xi += incx) {
        a_elem = a_i[aik];
        x_elem = x_i[xi];
        prod = impl::mul<PrdType>(a_elem, x_elem);
        sum = sum + prod;
      }
      y_elem = y_i[yi];
      tmp2 = impl::mul<TmpType>(y_elem, beta_i);
      tmp1 = impl::mul<TmpType>(sum, alpha_i);
      tmp1 = tmp2 + tmp1;
      y_i[yi] = impl::to<T>(tmp1);
    }
  }



  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::symv */

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
constexpr void symv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      N n,
                      T alpha,
                      const A *a,
                      N lda,
                      const X *x,
                      N incx,
                      T beta,
                      T *y,
                      N incy,
                      blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routines computes the matrix product:
 *
 *     y  <-  alpha * A * x  +  beta * y
 *
 * where A is a Symmetric matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Storage format of input symmetric matrix A.
 *
 * uplo   (input) blas_uplo_type
 *        Determines which half of matrix A (upper or lower triangle)
 *          is accessed.
 *
 * n      (input) N
 *        Dimension of A and size of vectors x, y.
 *
 * alpha  (input) T
 *
 * a      (input) const A*
 *        Matrix A.
 *
 * lda    (input) N
 *        Leading dimension of matrix A.
 *
 * x      (input) const X*
 *        Vector x.
 *
 * incx   (input) N
 *        Stride for vector x.
 *
 * beta   (input) T
 *
 * y      (input/output) T*
 *        Vector y.
 *
 * incy   (input) N
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
//static const char routine_name[] = "XBLAS::symv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::symv<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::symv<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::symv<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::symv<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  }
} /* end XBLAS::symv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_SYMV_HPP
