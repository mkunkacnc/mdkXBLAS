#ifndef XBLAS_HBMV_HPP
#define XBLAS_HBMV_HPP

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
constexpr void hbmv_impl(N n,
                         N k,
                         T alpha,
                         const A *a,
                         const X *x,
                         N incx,
                         T beta,
                         T *y,
                         N incy,
                         IdxType x_starti,
                         IdxType y_starti,
                         IdxType astarti,
                         IdxType incaij,
                         IdxType incaij2)
{
  /*  determine the loop iteration counts */
  /* maj_first is number of elements done in first loop
     (this will increase by one over each column up to a limit) */
  IdxType maxj_first = 0;

  /* maxj_second is number of elements done in
     second loop the first time */
  IdxType maxj_second = std::min(k + 1, n);

  IdxType yi = y_starti;
  for (IdxType i = 0; i < n; i++) {
    PrdType sum = impl::zero_v<PrdType>;
    IdxType aij = astarti;
    IdxType xi = x_starti;

    for (IdxType j = 0; j < maxj_first; j++) {
      A a_elem = impl::Conj_h<do_conj1>::func(a[aij]);
      PrdType prod = impl::mul<PrdType>(a_elem, x[xi]);
      sum += prod;
      aij += incaij;
      xi += incx;
    }

    auto a_elem_r = impl::Real_h<do_conj1 || do_conj2>::func(a[aij]);
    PrdType prod = impl::mul<PrdType>(a_elem_r, x[xi]);
    sum += prod;
    aij += incaij2;
    xi += incx;

    for (IdxType j = 1; j < maxj_second; j++) {
      A a_elem = impl::Conj_h<do_conj2>::func(a[aij]);
      PrdType prod = impl::mul<PrdType>(a_elem, x[xi]);
      sum += prod;
      aij += incaij2;
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
    if (i + 1 >= (n - k)) {
      maxj_second--;
    }
    if (i >= k) {
      astarti += (incaij + incaij2);
      x_starti += incx;
    } else {
      maxj_first++;
      astarti += incaij2;
    }
  }
} /* end XBLAS::impl::hbmv_impl */

//-----------------
} // namespace impl
//-----------------

template<typename T,
         typename A,
         typename X,
         typename N,
         typename TmpType = T,
         typename IdxType = impl::internal_index_type_t<N>>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void hbmv(blas_order_type order,
                    blas_uplo_type uplo,
                    N n,
                    N k,
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
 * where A is a hermitian band matrix.
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
 * k      (input) N
 *        Number of subdiagonals ( = number of superdiagonals)
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
 *
 *  Notes on storing a hermitian band matrix:
 *
 *      Integers in the below arrays represent values of
 *              type complex float.
 *
 *    if we have a hermitian matrix:
 *
 *      1d  2   3   0   0
 *      2#  4d  5   6   0
 *      3#  5#  7d  8   9
 *      0   6#  8#  10d 11
 *      0   0   9#  11# 12d
 *
 *     This matrix has n == 5, and k == 2. It can be stored in the
 *      following ways:
 *
 *      Notes for the examples:
 *      Each column below represents a contiguous vector.
 *      Columns are strided by lda.
 *      An asterisk (*) represents a position in the
 *       matrix that is not used.
 *      A pound sign (#) represents the conjugated form is stored
 *      A d following an integer indicates that the imaginary
 *       part of the number is assumed to be zero.
 *      Note that the minimum lda (size of column) is 3 (k+1).
 *       lda may be arbitrarily large; an lda > 3 would mean
 *       there would be unused data at the bottom of the below
 *       columns.
 *
 *    blas_colmajor and blas_upper:
 *      *   *   3   6   9
 *      *   2   5   8   11
 *      1d  4d  7d  10d 12d
 *
 *
 *    blas_colmajor and blas_lower
 *      1d   4d   7d   10d  12d
 *      2#   5#   8#   11#  *
 *      3#   6#   9#   *    *
 *
 *
 *    blas_rowmajor and blas_upper
 *      Columns here also represent contiguous arrays.
 *      1d  4d  7d  10d  12d
 *      2   5   8   11   *
 *      3   6   9   *    *
 *
 *
 *    blas_rowmajor and blas_lower
 *      Columns here also represent contiguous arrays.
 *      *   *   3#  6#   9#
 *      *   2#  5#  8#   11#
 *      1d  4d  7d  10d  12d
 *
 */
{
  /* Routine name */
  static const char *routine_name = "XBLAS::hbmv";

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
  if (k < 0 || k > n) {
    BLAS_error(routine_name, -4, k, nullptr);
  }
  if ((lda < k + 1) || (lda < 1)) {
    BLAS_error(routine_name, -7, lda, nullptr);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -9, incx, nullptr);
  }
  if (incy == 0) {
    BLAS_error(routine_name, -12, incy, nullptr);
  }

  /* Test for no-op */
  if (n == 0) {
    return;
  }
  if (alpha == T(0) && beta == T(1)) {
    return;
  }

  IdxType astarti, incaij, incaij2;
  if (((uplo == blas_upper) && (order == blas_colmajor)) ||
      ((uplo == blas_lower) && (order == blas_rowmajor))) {
    incaij = 1;                 /* increment in first loop */
    incaij2 = lda - 1;          /* increment in second loop */
    astarti = k;                /* does not start on zero element */
  } else {
    incaij = lda - 1;
    incaij2 = 1;
    astarti = 0;                /* start on first element of array */
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
      /*  conjugate second loop */

      /* Case alpha == 1. */
      if (alpha == T(1)) {
        if (beta == T(0)) {
          /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
          impl::hbmv_impl<0, 1,  1,  0, TmpType, PrdType>(n, k, alpha, a, x, incx, beta, y, incy,
                                                          x_starti, y_starti, astarti, incaij, incaij2);
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  y  <--- A * x + beta * y */
          impl::hbmv_impl<0, 1,  1, -1, TmpType, PrdType>(n, k, alpha, a, x, incx, beta, y, incy,
                                                          x_starti, y_starti, astarti, incaij, incaij2);
        }
      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        impl::hbmv_impl<0, 1, -1, -1, TmpType, PrdType>(n, k, alpha, a, x, incx, beta, y, incy,
                                                        x_starti, y_starti, astarti, incaij, incaij2);
      }
    } else {
      /*  conjugate first loop */

      /* Case alpha == 1. */
      if (alpha == T(1)) {
        if (beta == T(0)) {
          /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
          impl::hbmv_impl<1, 0,  1,  0, TmpType, PrdType>(n, k, alpha, a, x, incx, beta, y, incy,
                                                          x_starti, y_starti, astarti, incaij, incaij2);
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  y  <--- A * x + beta * y */
          impl::hbmv_impl<1, 0,  1, -1, TmpType, PrdType>(n, k, alpha, a, x, incx, beta, y, incy,
                                                          x_starti, y_starti, astarti, incaij, incaij2);
        }
      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        impl::hbmv_impl<1, 0, -1, -1, TmpType, PrdType>(n, k, alpha, a, x, incx, beta, y, incy,
                                                        x_starti, y_starti, astarti, incaij, incaij2);
      }
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::hbmv */

//-----------------

template<typename T,
         typename A,
         typename X,
         typename N,
         typename IdxType = impl::internal_index_type_t<N>>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void hbmv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      N n,
                      N k,
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
 * where A is a hermitian band matrix.
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
 * k      (input) N
 *        Number of subdiagonals ( = number of superdiagonals)
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
 *
 *  Notes on storing a hermitian band matrix:
 *
 *      Integers in the below arrays represent values of
 *              type complex float.
 *
 *    if we have a hermitian matrix:
 *
 *      1d  2   3   0   0
 *      2#  4d  5   6   0
 *      3#  5#  7d  8   9
 *      0   6#  8#  10d 11
 *      0   0   9#  11# 12d
 *
 *     This matrix has n == 5, and k == 2. It can be stored in the
 *      following ways:
 *
 *      Notes for the examples:
 *      Each column below represents a contiguous vector.
 *      Columns are strided by lda.
 *      An asterisk (*) represents a position in the
 *       matrix that is not used.
 *      A pound sign (#) represents the conjugated form is stored
 *      A d following an integer indicates that the imaginary
 *       part of the number is assumed to be zero.
 *      Note that the minimum lda (size of column) is 3 (k+1).
 *       lda may be arbitrarily large; an lda > 3 would mean
 *       there would be unused data at the bottom of the below
 *       columns.
 *
 *    blas_colmajor and blas_upper:
 *      *   *   3   6   9
 *      *   2   5   8   11
 *      1d  4d  7d  10d 12d
 *
 *
 *    blas_colmajor and blas_lower
 *      1d   4d   7d   10d  12d
 *      2#   5#   8#   11#  *
 *      3#   6#   9#   *    *
 *
 *
 *    blas_rowmajor and blas_upper
 *      Columns here also represent contiguous arrays.
 *      1d  4d  7d  10d  12d
 *      2   5   8   11   *
 *      3   6   9   *    *
 *
 *
 *    blas_rowmajor and blas_lower
 *      Columns here also represent contiguous arrays.
 *      *   *   3#  6#   9#
 *      *   2#  5#  8#   11#
 *      1d  4d  7d  10d  12d
 *
 */
{
  static const char *routine_name = "XBLAS::hbmv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::hbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::hbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::hbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::hbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    break;
  default:
    BLAS_error(routine_name, -13, prec, nullptr);
    break;
  }
} /* end XBLAS::hbmv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_HBMV_HPP
