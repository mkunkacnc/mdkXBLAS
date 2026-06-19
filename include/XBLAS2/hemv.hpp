#ifndef XBLAS_HEMV_HPP
#define XBLAS_HEMV_HPP

#include "blas_enum.h"
#include "impl.hpp"

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

template<int do_conj1,
         int do_conj2,
         int need_alpha,
         int need_beta,
         typename TmpType,
         typename PrdType,
         typename T,
         typename A,
         typename X,
         typename N,
         typename IdxType>
constexpr void hemv_impl(N n,
                         T alpha,
                         const A *a,
                         const X *x,
                         N incx,
                         T beta,
                         T *y,
                         N incy,
                         IdxType x_starti,
                         IdxType y_starti,
                         IdxType incaik,
                         IdxType incaik2)
{
  IdxType yi = y_starti;
  IdxType astarti = 0;
  for (IdxType i = 0; i < n; i++) {
    PrdType sum = impl::zero_v<PrdType>;
    IdxType aik = astarti;
    IdxType xi = x_starti;
    IdxType k = 0;

    for (; k < i; k++) {
      A a_elem = impl::Conj_h<do_conj1>::func(a[aik]);
      PrdType prod = impl::mul<PrdType>(a_elem, x[xi]);
      sum += prod;
      aik += incaik;
      xi += incx;
    }

    auto a_elem_r = impl::Real_h<do_conj1 || do_conj2>::func(a[aik]);
    PrdType prod = impl::mul<PrdType>(a_elem_r, x[xi]);
    sum += prod;
    k++;
    aik += incaik2;
    xi += incx;

    for (; k < n; k++) {
      A a_elem = impl::Conj_h<do_conj2>::func(a[aik]);
      PrdType prod = impl::mul<PrdType>(a_elem, x[xi]);
      sum += prod;
      aik += incaik2;
      xi += incx;
    }

    if constexpr (need_alpha == 1) {
      if constexpr (need_beta == 0) {
        y[yi] = impl::to<T>(sum);
      } else {
        TmpType tmp1 = sum;
        TmpType tmp2 = impl::mul<TmpType>(y[yi], beta);
        tmp1 += tmp2;
        y[yi] = impl::to<T>(tmp1);
      }
    } else {
      TmpType tmp1 = impl::mul<TmpType>(sum, alpha);
      TmpType tmp2 = impl::mul<TmpType>(y[yi], beta);
      tmp1 += tmp2;
      y[yi] = impl::to<T>(tmp1);
    }

    yi += incy;
    astarti += incaik2;
  }
} /* end XBLAS::impl::hemv_impl */

//-----------------
} // namespace impl
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
constexpr void hemv(blas_order_type order,
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
  static const char *routine_name = "XBLAS::hemv";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  /* Check for error conditions. */
  if (order != blas_colmajor && order != blas_rowmajor) {
    BLAS_error(routine_name, -1, order, nullptr);
  }
  if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, nullptr);
  }
  if (n < 0) {
    BLAS_error(routine_name, -3, n, nullptr);
  }
  if (lda < n) {
    BLAS_error(routine_name, -6, lda, nullptr);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -8, incx, nullptr);
  }
  if (incy == 0) {
    BLAS_error(routine_name, -11, incy, nullptr);
  }

  /* Test for no-op */
  if (n == 0) {
    return;
  }
  if (alpha == T(0) && beta == T(1)) {
    return;
  }

  /* Set Index Parameters */
  IdxType incaik, incaik2;
  if ((order == blas_colmajor && uplo == blas_upper) ||
      (order == blas_rowmajor && uplo == blas_lower)) {
    incaik = 1;
    incaik2 = lda;
  } else {
    incaik = lda;
    incaik2 = 1;
  }

  IdxType x_starti;
  if (incx < 0) {
    x_starti = (-n + 1) * incx;
  } else {
    x_starti = 0;
  }

  IdxType y_starti;
  if (incy < 0) {
    y_starti = (-n + 1) * incy;
  } else {
    y_starti = 0;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  /* alpha = 0.  In this case, just return beta * y */
  if (alpha == T(0)) {
    IdxType yi = y_starti;
    for (IdxType i = 0; i < n; i++) {
      TmpType tmp1 = impl::mul<TmpType>(y[yi], beta);
      y[yi] = impl::to<T>(tmp1);
      yi += incy;
    }
  } else {
    /*  determine whether we conjugate in first loop or second loop */
    if (uplo == blas_lower) {
      /*  conjugate second */

      /* Case alpha == 1. */
      if (alpha == T(1)) {
        if (beta == T(0)) {
          /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
          impl::hemv_impl<0, 1,  1,  0, TmpType, PrdType>(n, alpha, a, x, incx, beta, y, incy,
                                                          x_starti, y_starti, incaik, incaik2);
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  y  <--- A * x + beta * y */
          impl::hemv_impl<0, 1,  1, -1, TmpType, PrdType>(n, alpha, a, x, incx, beta, y, incy,
                                                          x_starti, y_starti, incaik, incaik2);
        }
      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        impl::hemv_impl<0, 1, -1, -1, TmpType, PrdType>(n, alpha, a, x, incx, beta, y, incy,
                                                        x_starti, y_starti, incaik, incaik2);
      }
    } else {
      /*  conjugate first loop */

      /* Case alpha == 1. */
      if (alpha == T(1)) {
        if (beta == T(0)) {
          /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
          impl::hemv_impl<1, 0,  1,  0, TmpType, PrdType>(n, alpha, a, x, incx, beta, y, incy,
                                                          x_starti, y_starti, incaik, incaik2);
        } else {
          /* Case alpha = 1, but beta != 0.
               We compute  y  <--- A * x + beta * y */
          impl::hemv_impl<1, 0,  1, -1, TmpType, PrdType>(n, alpha, a, x, incx, beta, y, incy,
                                                          x_starti, y_starti, incaik, incaik2);
        }
      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        impl::hemv_impl<1, 0, -1, -1, TmpType, PrdType>(n, alpha, a, x, incx, beta, y, incy,
                                                        x_starti, y_starti, incaik, incaik2);
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
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void hemv_x(blas_order_type order,
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
  static const char *routine_name = "XBLAS::hemv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::hemv<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::hemv<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::hemv<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::hemv<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  default:
    BLAS_error(routine_name, -12, prec, nullptr);
    break;
  }
} /* end XBLAS::hemv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_HEMV_HPP
