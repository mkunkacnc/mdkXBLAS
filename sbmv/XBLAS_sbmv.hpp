#ifndef XBLAS_SBMV_HPP
#define XBLAS_SBMV_HPP

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
constexpr void sbmv(blas_order_type order,
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
 * where A is a symmetric band matrix.
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
 *  Notes on storing a symmetric band matrix:
 *
 *      Integers in the below arrays represent values of
 *              type double.
 *
 *    if we have a symmetric matrix:
 *
 *      1  2  3  0  0
 *      2  4  5  6  0
 *      3  5  7  8  9
 *      0  6  8  10 11
 *      0  0  9  11 12
 *
 *     This matrix has n == 5, and k == 2. It can be stored in the
 *      following ways:
 *
 *      Notes for the examples:
 *      Each column below represents a contiguous vector.
 *      Columns are strided by lda.
 *      An asterisk (*) represents a position in the
 *       matrix that is not used.
 *      Note that the minimum lda (size of column) is 3 (k+1).
 *       lda may be arbitrarily large; an lda > 3 would mean
 *       there would be unused data at the bottom of the below
 *       columns.
 *
 *    blas_colmajor and blas_upper:
 *      *  *  3  6  9
 *      *  2  5  8  11
 *      1  4  7  10 12
 *
 *
 *    blas_colmajor and blas_lower
 *      1  4  7  10  12
 *      2  5  8  11  *
 *      3  6  9  *   *
 *
 *
 *    blas_rowmajor and blas_upper
 *      Columns here also represent contiguous arrays.
 *      1  4  7  10  12
 *      2  5  8  11  *
 *      3  6  9  *   *
 *
 *
 *    blas_rowmajor and blas_lower
 *      Columns here also represent contiguous arrays.
 *      *  *  3  6   9
 *      *  2  5  8   11
 *      1  4  7  10  12
 *
 */
{
  static const char *routine_name = "XBLAS::sbmv";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  /* Integer Index Variables */
  IdxType i, j;
  IdxType xi, yi;
  IdxType aij, astarti, x_starti, y_starti;
  IdxType incaij, incaij2;
  IdxType n_i;
  IdxType maxj_first, maxj_second;

  /* Input Matrices */
  const A *a_i = a;
  const X *x_i = x;

  /* Output Vector */
  T *y_i = y;

  /* Input Scalars */

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
  if (alpha == T(0) && beta == T(1)) {
    return;
  }

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

  /* Set Index Parameters */
  n_i = n;

  if (((uplo == blas_upper) && (order == blas_colmajor)) ||
      ((uplo == blas_lower) && (order == blas_rowmajor))) {
    incaij = 1;                        /* increment in first loop */
    incaij2 = lda - 1;                /* increment in second loop */
    astarti = k;                /* does not start on zero element */
  } else {
    incaij = lda - 1;
    incaij2 = 1;
    astarti = 0;                /* start on first element of array */
  }
  /* Adjustment to increments (if any) */
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
  if (alpha == T(0)) {
    for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
      y_elem = y_i[yi];
      tmp1 = impl::mul<TmpType>(y_elem, beta);
      y_i[yi] = impl::to<T>(tmp1);
    }
  } else {
    /*  determine the loop iteration counts */
    /* number of elements done in first loop
       (this will increase by one over each column up to a limit) */
    maxj_first = 0;
    /* number of elements done in second loop the first time */
    maxj_second = std::min(k + 1, n_i);

    /* Case alpha == 1. */
    if (alpha == T(1)) {
      if (beta == T(0)) {
        /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
        for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
          sum = impl::zero_v<PrdType>;
          for (j = 0, aij = astarti, xi = x_starti;
               j < maxj_first; j++, aij += incaij, xi += incx) {
            a_elem = a_i[aij];
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          for (j = 0; j < maxj_second; j++, aij += incaij2, xi += incx) {
            a_elem = a_i[aij];
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          y_i[yi] = impl::to<T>(sum);
          if (i + 1 >= (n_i - k))
            maxj_second--;
          if (i >= k) {
            astarti += (incaij + incaij2);
            x_starti += incx;
          } else {
            maxj_first++;
            astarti += incaij2;
          }
        }
      } else {
        /* Case alpha = 1, but beta != 0.
           We compute  y  <--- A * x + beta * y */
        for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
          sum = impl::zero_v<PrdType>;
          for (j = 0, aij = astarti, xi = x_starti;
               j < maxj_first; j++, aij += incaij, xi += incx) {
            a_elem = a_i[aij];
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          for (j = 0; j < maxj_second; j++, aij += incaij2, xi += incx) {
            a_elem = a_i[aij];
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          y_elem = y_i[yi];
          tmp2 = impl::mul<TmpType>(y_elem, beta);
          tmp1 = sum;
          tmp1 = tmp2 + tmp1;
          y_i[yi] = impl::to<T>(tmp1);
          if (i + 1 >= (n_i - k))
            maxj_second--;
          if (i >= k) {
            astarti += (incaij + incaij2);
            x_starti += incx;
          } else {
            maxj_first++;
            astarti += incaij2;
          }
        }
      }
    } else {
      if (beta == T(0)) {
        /* Case alpha != 1, but beta == 0.
           We compute  y  <--- A * x * a */
        for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
          sum = impl::zero_v<PrdType>;

          for (j = 0, aij = astarti, xi = x_starti;
               j < maxj_first; j++, aij += incaij, xi += incx) {
            a_elem = a_i[aij];
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          for (j = 0; j < maxj_second; j++, aij += incaij2, xi += incx) {
            a_elem = a_i[aij];
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          y_elem = y_i[yi];
          tmp1 = impl::mul<TmpType>(sum, alpha);
          y_i[yi] = impl::to<T>(tmp1);
          if (i + 1 >= (n_i - k))
            maxj_second--;
          if (i >= k) {
            astarti += (incaij + incaij2);
            x_starti += incx;
          } else {
            maxj_first++;
            astarti += incaij2;
          }
        }
      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
          sum = impl::zero_v<PrdType>;

          for (j = 0, aij = astarti, xi = x_starti;
               j < maxj_first; j++, aij += incaij, xi += incx) {
            a_elem = a_i[aij];
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          for (j = 0; j < maxj_second; j++, aij += incaij2, xi += incx) {
            a_elem = a_i[aij];
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          y_elem = y_i[yi];
          tmp2 = impl::mul<TmpType>(y_elem, beta);
          tmp1 = impl::mul<TmpType>(sum, alpha);
          tmp1 = tmp2 + tmp1;
          y_i[yi] = impl::to<T>(tmp1);
          if (i + 1 >= (n_i - k))
            maxj_second--;
          if (i >= k) {
            astarti += (incaij + incaij2);
            x_starti += incx;
          } else {
            maxj_first++;
            astarti += incaij2;
          }
        }
      }
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::sbmv */

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
constexpr void sbmv_x(blas_order_type order,
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
 * where A is a symmetric band matrix.
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
 *  Notes on storing a symmetric band matrix:
 *
 *      Integers in the below arrays represent values of
 *              type double.
 *
 *    if we have a symettric matrix:
 *
 *      1  2  3  0  0
 *      2  4  5  6  0
 *      3  5  7  8  9
 *      0  6  8  10 11
 *      0  0  9  11 12
 *
 *     This matrix has n == 5, and k == 2. It can be stored in the
 *      following ways:
 *
 *      Notes for the examples:
 *      Each column below represents a contiguous vector.
 *      Columns are strided by lda.
 *      An asterisk (*) represents a position in the
 *       matrix that is not used.
 *      Note that the minimum lda (size of column) is 3 (k+1).
 *       lda may be arbitrarily large; an lda > 3 would mean
 *       there would be unused data at the bottom of the below
 *       columns.
 *
 *    blas_colmajor and blas_upper:
 *      *  *  3  6  9
 *      *  2  5  8  11
 *      1  4  7  10 12
 *
 *
 *    blas_colmajor and blas_lower
 *      1  4  7  10  12
 *      2  5  8  11  *
 *      3  6  9  *   *
 *
 *
 *    blas_rowmajor and blas_upper
 *      Columns here also represent contiguous arrays.
 *      1  4  7  10  12
 *      2  5  8  11  *
 *      3  6  9  *   *
 *
 *
 *    blas_rowmajor and blas_lower
 *      Columns here also represent contiguous arrays.
 *      *  *  3  6   9
 *      *  2  5  8   11
 *      1  4  7  10  12
 *
 */
{
  static const char *routine_name = "XBLAS::sbmv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::sbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::sbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::sbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::sbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    break;
  default:
    BLAS_error(routine_name, -13, prec, nullptr);
    break;
  }
} /* end XBLAS::sbmv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_SBMV_HPP
