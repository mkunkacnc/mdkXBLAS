#ifndef XBLAS_HBMV_HPP
#define XBLAS_HBMV_HPP

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
constexpr void hbmv(blas_order_type order,
                    blas_uplo_type uplo,
                    IdxType n,
                    IdxType k,
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
 * n      (input) IdxType
 *        Dimension of A and size of vectors x, y.
 *
 * k      (input) IdxType
 *        Number of subdiagonals ( = number of superdiagonals)
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
 *      Each column below represents a contigous vector.
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
 *      Columns here also represent contigous arrays.
 *      1d  4d  7d  10d  12d
 *      2   5   8   11   *
 *      3   6   9   *    *
 *
 *
 *    blas_rowmajor and blas_lower
 *      Columns here also represent contigous arrays.
 *      *   *   3#  6#   9#
 *      *   2#  5#  8#   11#
 *      1d  4d  7d  10d  12d
 *
 */
{
  /* Routine name */
  static const char routine_name[] = "XBLAS::hbmv";

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
    BLAS_error(routine_name, -1, order, 0);
  }
  if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, 0);
  }
  if (n < 0) {
    BLAS_error(routine_name, -3, n, 0);
  }
  if (k < 0 || k > n) {
    BLAS_error(routine_name, -4, k, 0);
  }
  if ((lda < k + 1) || (lda < 1)) {
    BLAS_error(routine_name, -7, lda, 0);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -9, incx, 0);
  }
  if (incy == 0) {
    BLAS_error(routine_name, -12, incy, 0);
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
    /*  determine the loop iteration counts */
    /* maj_first is number of elements done in first loop
       (this will increase by one over each column up to a limit) */
    maxj_first = 0;

    /* maxj_second is number of elements done in
       second loop the first time */
    maxj_second = std::min(k + 1, n_i);

    /*  determine whether we conjugate in first loop or second loop */
    if (uplo == blas_lower) {
      /*  conjugate second loop */

      /* Case alpha == 1. */
      if (alpha_i == T(1)) {
        if (beta_i == T(0)) {
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
            auto a_elem_r = std::real(a_i[aij]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem_r, x_elem);
            sum = sum + prod;
            aij += incaij2;
            xi += incx;
            for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
              a_elem = impl::Conj::func(a_i[aij]);
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            y_i[yi] = impl::to<T>(sum);
            if (i + 1 >= (n_i - k)) {
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
            auto a_elem_r = std::real(a_i[aij]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem_r, x_elem);
            sum = sum + prod;
            aij += incaij2;
            xi += incx;
            for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
              a_elem = impl::Conj::func(a_i[aij]);
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            y_elem = y_i[yi];
            tmp2 = impl::mul<TmpType>(y_elem, beta_i);
            tmp1 = sum;
            tmp1 = tmp2 + tmp1;
            y_i[yi] = impl::to<T>(tmp1);
            if (i + 1 >= (n_i - k)) {
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
          auto a_elem_r = std::real(a_i[aij]);
          x_elem = x_i[xi];
          prod = impl::mul<PrdType>(a_elem_r, x_elem);
          sum = sum + prod;
          aij += incaij2;
          xi += incx;
          for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
            a_elem = impl::Conj::func(a_i[aij]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          y_elem = y_i[yi];
          tmp2 = impl::mul<TmpType>(y_elem, beta_i);
          tmp1 = impl::mul<TmpType>(sum, alpha_i);
          tmp1 = tmp2 + tmp1;
          y_i[yi] = impl::to<T>(tmp1);
          if (i + 1 >= (n_i - k)) {
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
      }
    } else {
      /*  conjugate first loop */

      /* Case alpha == 1. */
      if (alpha_i == T(1)) {
        if (beta_i == T(0)) {
          /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
          for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
            sum = impl::zero_v<PrdType>;
            for (j = 0, aij = astarti, xi = x_starti;
                 j < maxj_first; j++, aij += incaij, xi += incx) {
              a_elem = impl::Conj::func(a_i[aij]);
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            auto a_elem_r = std::real(a_i[aij]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem_r, x_elem);
            sum = sum + prod;
            aij += incaij2;
            xi += incx;
            for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
              a_elem = a_i[aij];
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            y_i[yi] = impl::to<T>(sum);
            if (i + 1 >= (n_i - k)) {
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
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  y  <--- A * x + beta * y */
          for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
            sum = impl::zero_v<PrdType>;
            for (j = 0, aij = astarti, xi = x_starti;
                 j < maxj_first; j++, aij += incaij, xi += incx) {
              a_elem = impl::Conj::func(a_i[aij]);
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            auto a_elem_r = std::real(a_i[aij]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem_r, x_elem);
            sum = sum + prod;
            aij += incaij2;
            xi += incx;
            for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
              a_elem = a_i[aij];
              x_elem = x_i[xi];
              prod = impl::mul<PrdType>(a_elem, x_elem);
              sum = sum + prod;
            }
            y_elem = y_i[yi];
            tmp2 = impl::mul<TmpType>(y_elem, beta_i);
            tmp1 = sum;
            tmp1 = tmp2 + tmp1;
            y_i[yi] = impl::to<T>(tmp1);
            if (i + 1 >= (n_i - k)) {
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
        }
      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
          sum = impl::zero_v<PrdType>;
          for (j = 0, aij = astarti, xi = x_starti;
               j < maxj_first; j++, aij += incaij, xi += incx) {
            a_elem = impl::Conj::func(a_i[aij]);
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          auto a_elem_r = std::real(a_i[aij]);
          x_elem = x_i[xi];
          prod = impl::mul<PrdType>(a_elem_r, x_elem);
          sum = sum + prod;
          aij += incaij2;
          xi += incx;
          for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
            a_elem = a_i[aij];
            x_elem = x_i[xi];
            prod = impl::mul<PrdType>(a_elem, x_elem);
            sum = sum + prod;
          }
          y_elem = y_i[yi];
          tmp2 = impl::mul<TmpType>(y_elem, beta_i);
          tmp1 = impl::mul<TmpType>(sum, alpha_i);
          tmp1 = tmp2 + tmp1;
          y_i[yi] = impl::to<T>(tmp1);
          if (i + 1 >= (n_i - k)) {
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
      }
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::hbmv */

//-----------------

inline
constexpr void hbmv_x(enum blas_order_type order,
                      enum blas_uplo_type uplo,
                      int n,
                      int k,
                      const void *alpha,
                      const void *a,
                      int lda,
                      const void *x,
                      int incx,
                      const void *beta,
                      void *y,
                      int incy,
                      enum blas_prec_type prec)
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
 * order  (input) enum blas_order_type
 *        Storage format of input hermitian matrix A.
 *
 * uplo   (input) enum blas_uplo_type
 *        Determines which half of matrix A (upper or lower triangle)
 *          is accessed.
 *
 * n      (input) int
 *        Dimension of A and size of vectors x, y.
 *
 * k      (input) int
 *        Number of subdiagonals ( = number of superdiagonals)
 *
 * alpha  (input) const void*
 *
 * a      (input) const void*
 *        Matrix A.
 *
 * lda    (input) int
 *        Leading dimension of matrix A.
 *
 * x      (input) const void*
 *        Vector x.
 *
 * incx   (input) int
 *        Stride for vector x.
 *
 * beta   (input) const void*
 *
 * y      (input/output) void*
 *        Vector y.
 *
 * incy   (input) int
 *        Stride for vector y.
 *
 * prec   (input) enum blas_prec_type
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
 *      Each column below represents a contigous vector.
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
 *      Columns here also represent contigous arrays.
 *      1d  4d  7d  10d  12d
 *      2   5   8   11   *
 *      3   6   9   *    *
 *
 *
 *    blas_rowmajor and blas_lower
 *      Columns here also represent contigous arrays.
 *      *   *   3#  6#   9#
 *      *   2#  5#  8#   11#
 *      1d  4d  7d  10d  12d
 *
 */
{
  /* Routine name */
  static const char routine_name[] = "BLAS_zhbmv_c_c";
  switch (prec) {

  case blas_prec_single:
  case blas_prec_indigenous:
  case blas_prec_double:{

      /* Integer Index Variables */
      int i, j;
      int xi, yi;
      int aij, astarti, x_starti, y_starti;
      int incaij, incaij2;
      int n_i;
      int maxj_first, maxj_second;

      /* Input Matrices */
      const float *a_i = (float *) a;
      const float *x_i = (float *) x;

      /* Output Vector */
      double *y_i = (double *) y;

      /* Input Scalars */
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;

      /* Temporary Floating-Point Variables */
      float a_elem[2];
      float x_elem[2];
      double y_elem[2];
      double prod[2];
      double sum[2];
      double tmp1[2];
      double tmp2[2];



      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
          && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
        return;
      }

      /* Check for error conditions. */
      if (order != blas_colmajor && order != blas_rowmajor) {
        BLAS_error(routine_name, -1, order, 0);
      }
      if (uplo != blas_upper && uplo != blas_lower) {
        BLAS_error(routine_name, -2, uplo, 0);
      }
      if (n < 0) {
        BLAS_error(routine_name, -3, n, 0);
      }
      if (k < 0 || k > n) {
        BLAS_error(routine_name, -4, k, 0);
      }
      if ((lda < k + 1) || (lda < 1)) {
        BLAS_error(routine_name, -7, lda, 0);
      }
      if (incx == 0) {
        BLAS_error(routine_name, -9, incx, 0);
      }
      if (incy == 0) {
        BLAS_error(routine_name, -12, incy, 0);
      }

      /* Set Index Parameters */
      n_i = n;

      if (((uplo == blas_upper) && (order == blas_colmajor)) ||
          ((uplo == blas_lower) && (order == blas_rowmajor))) {
        incaij = 1;                /* increment in first loop */
        incaij2 = lda - 1;        /* increment in second loop */
        astarti = k;                /* does not start on zero element */
      } else {
        incaij = lda - 1;
        incaij2 = 1;
        astarti = 0;                /* start on first element of array */
      }
      /* Adjustment to increments (if any) */
      incx *= 2;
      incy *= 2;
      astarti *= 2;
      incaij *= 2;
      incaij2 *= 2;
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

      /* alpha = 0.  In this case, just return beta * y */
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
        for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
          y_elem[0] = y_i[yi];
          y_elem[1] = y_i[yi + 1];
          {
            tmp1[0] =
              (double) y_elem[0] * beta_i[0] - (double) y_elem[1] * beta_i[1];
            tmp1[1] =
              (double) y_elem[0] * beta_i[1] + (double) y_elem[1] * beta_i[0];
          }
          y_i[yi] = tmp1[0];
          y_i[yi + 1] = tmp1[1];
        }
      } else {
        /*  determine the loop interation counts */
        /* maj_first is number of elements done in first loop
           (this will increase by one over each column up to a limit) */
        maxj_first = 0;

        /* maxj_second is number of elements done in
           second loop the first time */
        maxj_second = MIN(k + 1, n_i);

        /*  determine whether we conjugate in first loop or second loop */
        if (uplo == blas_lower) {
          /*  conjugate second loop */

          /* Case alpha == 1. */
          if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {

            if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
              /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
              for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
                sum[0] = sum[1] = 0.0;
                for (j = 0, aij = astarti, xi = x_starti;
                     j < maxj_first; j++, aij += incaij, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];

                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    prod[0] =
                      (double) a_elem[0] * x_elem[0] -
                      (double) a_elem[1] * x_elem[1];
                    prod[1] =
                      (double) a_elem[0] * x_elem[1] +
                      (double) a_elem[1] * x_elem[0];
                  }
                  sum[0] = sum[0] + prod[0];
                  sum[1] = sum[1] + prod[1];
                }
                a_elem[0] = a_i[aij];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  prod[0] = (double) x_elem[0] * a_elem[0];
                  prod[1] = (double) x_elem[1] * a_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];
                aij += incaij2;
                xi += incx;
                for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];
                  a_elem[1] = -a_elem[1];
                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    prod[0] =
                      (double) a_elem[0] * x_elem[0] -
                      (double) a_elem[1] * x_elem[1];
                    prod[1] =
                      (double) a_elem[0] * x_elem[1] +
                      (double) a_elem[1] * x_elem[0];
                  }
                  sum[0] = sum[0] + prod[0];
                  sum[1] = sum[1] + prod[1];
                }
                y_i[yi] = sum[0];
                y_i[yi + 1] = sum[1];
                if (i + 1 >= (n_i - k)) {
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
            } else {
              /* Case alpha = 1, but beta != 0.
                 We compute  y  <--- A * x + beta * y */
              for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
                sum[0] = sum[1] = 0.0;

                for (j = 0, aij = astarti, xi = x_starti;
                     j < maxj_first; j++, aij += incaij, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];

                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    prod[0] =
                      (double) a_elem[0] * x_elem[0] -
                      (double) a_elem[1] * x_elem[1];
                    prod[1] =
                      (double) a_elem[0] * x_elem[1] +
                      (double) a_elem[1] * x_elem[0];
                  }
                  sum[0] = sum[0] + prod[0];
                  sum[1] = sum[1] + prod[1];
                }
                a_elem[0] = a_i[aij];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  prod[0] = (double) x_elem[0] * a_elem[0];
                  prod[1] = (double) x_elem[1] * a_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];
                aij += incaij2;
                xi += incx;
                for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];
                  a_elem[1] = -a_elem[1];
                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    prod[0] =
                      (double) a_elem[0] * x_elem[0] -
                      (double) a_elem[1] * x_elem[1];
                    prod[1] =
                      (double) a_elem[0] * x_elem[1] +
                      (double) a_elem[1] * x_elem[0];
                  }
                  sum[0] = sum[0] + prod[0];
                  sum[1] = sum[1] + prod[1];
                }
                y_elem[0] = y_i[yi];
                y_elem[1] = y_i[yi + 1];
                {
                  tmp2[0] =
                    (double) y_elem[0] * beta_i[0] -
                    (double) y_elem[1] * beta_i[1];
                  tmp2[1] =
                    (double) y_elem[0] * beta_i[1] +
                    (double) y_elem[1] * beta_i[0];
                }
                tmp1[0] = sum[0];
                tmp1[1] = sum[1];
                tmp1[0] = tmp2[0] + tmp1[0];
                tmp1[1] = tmp2[1] + tmp1[1];
                y_i[yi] = tmp1[0];
                y_i[yi + 1] = tmp1[1];
                if (i + 1 >= (n_i - k)) {
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
            }
          } else {
            /* The most general form,   y <--- alpha * A * x + beta * y */
            for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
              sum[0] = sum[1] = 0.0;

              for (j = 0, aij = astarti, xi = x_starti;
                   j < maxj_first; j++, aij += incaij, xi += incx) {
                a_elem[0] = a_i[aij];
                a_elem[1] = a_i[aij + 1];

                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  prod[0] =
                    (double) a_elem[0] * x_elem[0] -
                    (double) a_elem[1] * x_elem[1];
                  prod[1] =
                    (double) a_elem[0] * x_elem[1] +
                    (double) a_elem[1] * x_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];
              }
              a_elem[0] = a_i[aij];
              x_elem[0] = x_i[xi];
              x_elem[1] = x_i[xi + 1];
              {
                prod[0] = (double) x_elem[0] * a_elem[0];
                prod[1] = (double) x_elem[1] * a_elem[0];
              }
              sum[0] = sum[0] + prod[0];
              sum[1] = sum[1] + prod[1];
              aij += incaij2;
              xi += incx;
              for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                a_elem[0] = a_i[aij];
                a_elem[1] = a_i[aij + 1];
                a_elem[1] = -a_elem[1];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  prod[0] =
                    (double) a_elem[0] * x_elem[0] -
                    (double) a_elem[1] * x_elem[1];
                  prod[1] =
                    (double) a_elem[0] * x_elem[1] +
                    (double) a_elem[1] * x_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];
              }
              y_elem[0] = y_i[yi];
              y_elem[1] = y_i[yi + 1];
              {
                tmp2[0] =
                  (double) y_elem[0] * beta_i[0] -
                  (double) y_elem[1] * beta_i[1];
                tmp2[1] =
                  (double) y_elem[0] * beta_i[1] +
                  (double) y_elem[1] * beta_i[0];
              }
              {
                tmp1[0] =
                  (double) sum[0] * alpha_i[0] - (double) sum[1] * alpha_i[1];
                tmp1[1] =
                  (double) sum[0] * alpha_i[1] + (double) sum[1] * alpha_i[0];
              }
              tmp1[0] = tmp2[0] + tmp1[0];
              tmp1[1] = tmp2[1] + tmp1[1];
              y_i[yi] = tmp1[0];
              y_i[yi + 1] = tmp1[1];
              if (i + 1 >= (n_i - k)) {
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
          }
        } else {
          /*  conjugate first loop */

          /* Case alpha == 1. */
          if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {

            if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
              /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
              for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
                sum[0] = sum[1] = 0.0;
                for (j = 0, aij = astarti, xi = x_starti;
                     j < maxj_first; j++, aij += incaij, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];
                  a_elem[1] = -a_elem[1];
                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    prod[0] =
                      (double) a_elem[0] * x_elem[0] -
                      (double) a_elem[1] * x_elem[1];
                    prod[1] =
                      (double) a_elem[0] * x_elem[1] +
                      (double) a_elem[1] * x_elem[0];
                  }
                  sum[0] = sum[0] + prod[0];
                  sum[1] = sum[1] + prod[1];
                }
                a_elem[0] = a_i[aij];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  prod[0] = (double) x_elem[0] * a_elem[0];
                  prod[1] = (double) x_elem[1] * a_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];
                aij += incaij2;
                xi += incx;
                for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];

                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    prod[0] =
                      (double) a_elem[0] * x_elem[0] -
                      (double) a_elem[1] * x_elem[1];
                    prod[1] =
                      (double) a_elem[0] * x_elem[1] +
                      (double) a_elem[1] * x_elem[0];
                  }
                  sum[0] = sum[0] + prod[0];
                  sum[1] = sum[1] + prod[1];
                }
                y_i[yi] = sum[0];
                y_i[yi + 1] = sum[1];
                if (i + 1 >= (n_i - k)) {
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
            } else {
              /* Case alpha = 1, but beta != 0.
                 We compute  y  <--- A * x + beta * y */
              for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
                sum[0] = sum[1] = 0.0;

                for (j = 0, aij = astarti, xi = x_starti;
                     j < maxj_first; j++, aij += incaij, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];
                  a_elem[1] = -a_elem[1];
                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    prod[0] =
                      (double) a_elem[0] * x_elem[0] -
                      (double) a_elem[1] * x_elem[1];
                    prod[1] =
                      (double) a_elem[0] * x_elem[1] +
                      (double) a_elem[1] * x_elem[0];
                  }
                  sum[0] = sum[0] + prod[0];
                  sum[1] = sum[1] + prod[1];
                }
                a_elem[0] = a_i[aij];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  prod[0] = (double) x_elem[0] * a_elem[0];
                  prod[1] = (double) x_elem[1] * a_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];
                aij += incaij2;
                xi += incx;
                for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];

                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    prod[0] =
                      (double) a_elem[0] * x_elem[0] -
                      (double) a_elem[1] * x_elem[1];
                    prod[1] =
                      (double) a_elem[0] * x_elem[1] +
                      (double) a_elem[1] * x_elem[0];
                  }
                  sum[0] = sum[0] + prod[0];
                  sum[1] = sum[1] + prod[1];
                }
                y_elem[0] = y_i[yi];
                y_elem[1] = y_i[yi + 1];
                {
                  tmp2[0] =
                    (double) y_elem[0] * beta_i[0] -
                    (double) y_elem[1] * beta_i[1];
                  tmp2[1] =
                    (double) y_elem[0] * beta_i[1] +
                    (double) y_elem[1] * beta_i[0];
                }
                tmp1[0] = sum[0];
                tmp1[1] = sum[1];
                tmp1[0] = tmp2[0] + tmp1[0];
                tmp1[1] = tmp2[1] + tmp1[1];
                y_i[yi] = tmp1[0];
                y_i[yi + 1] = tmp1[1];
                if (i + 1 >= (n_i - k)) {
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
            }
          } else {
            /* The most general form,   y <--- alpha * A * x + beta * y */
            for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
              sum[0] = sum[1] = 0.0;

              for (j = 0, aij = astarti, xi = x_starti;
                   j < maxj_first; j++, aij += incaij, xi += incx) {
                a_elem[0] = a_i[aij];
                a_elem[1] = a_i[aij + 1];
                a_elem[1] = -a_elem[1];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  prod[0] =
                    (double) a_elem[0] * x_elem[0] -
                    (double) a_elem[1] * x_elem[1];
                  prod[1] =
                    (double) a_elem[0] * x_elem[1] +
                    (double) a_elem[1] * x_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];
              }
              a_elem[0] = a_i[aij];
              x_elem[0] = x_i[xi];
              x_elem[1] = x_i[xi + 1];
              {
                prod[0] = (double) x_elem[0] * a_elem[0];
                prod[1] = (double) x_elem[1] * a_elem[0];
              }
              sum[0] = sum[0] + prod[0];
              sum[1] = sum[1] + prod[1];
              aij += incaij2;
              xi += incx;
              for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                a_elem[0] = a_i[aij];
                a_elem[1] = a_i[aij + 1];

                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  prod[0] =
                    (double) a_elem[0] * x_elem[0] -
                    (double) a_elem[1] * x_elem[1];
                  prod[1] =
                    (double) a_elem[0] * x_elem[1] +
                    (double) a_elem[1] * x_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];
              }
              y_elem[0] = y_i[yi];
              y_elem[1] = y_i[yi + 1];
              {
                tmp2[0] =
                  (double) y_elem[0] * beta_i[0] -
                  (double) y_elem[1] * beta_i[1];
                tmp2[1] =
                  (double) y_elem[0] * beta_i[1] +
                  (double) y_elem[1] * beta_i[0];
              }
              {
                tmp1[0] =
                  (double) sum[0] * alpha_i[0] - (double) sum[1] * alpha_i[1];
                tmp1[1] =
                  (double) sum[0] * alpha_i[1] + (double) sum[1] * alpha_i[0];
              }
              tmp1[0] = tmp2[0] + tmp1[0];
              tmp1[1] = tmp2[1] + tmp1[1];
              y_i[yi] = tmp1[0];
              y_i[yi + 1] = tmp1[1];
              if (i + 1 >= (n_i - k)) {
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
          }
        }
      }


      break;
    }

  case blas_prec_extra:{

      /* Integer Index Variables */
      int i, j;
      int xi, yi;
      int aij, astarti, x_starti, y_starti;
      int incaij, incaij2;
      int n_i;
      int maxj_first, maxj_second;

      /* Input Matrices */
      const float *a_i = (float *) a;
      const float *x_i = (float *) x;

      /* Output Vector */
      double *y_i = (double *) y;

      /* Input Scalars */
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;

      /* Temporary Floating-Point Variables */
      float a_elem[2];
      float x_elem[2];
      double y_elem[2];
      double head_prod[2], tail_prod[2];
      double head_sum[2], tail_sum[2];
      double head_tmp1[2], tail_tmp1[2];
      double head_tmp2[2], tail_tmp2[2];
      FPU_FIX_DECL;


      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
          && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
        return;
      }

      /* Check for error conditions. */
      if (order != blas_colmajor && order != blas_rowmajor) {
        BLAS_error(routine_name, -1, order, 0);
      }
      if (uplo != blas_upper && uplo != blas_lower) {
        BLAS_error(routine_name, -2, uplo, 0);
      }
      if (n < 0) {
        BLAS_error(routine_name, -3, n, 0);
      }
      if (k < 0 || k > n) {
        BLAS_error(routine_name, -4, k, 0);
      }
      if ((lda < k + 1) || (lda < 1)) {
        BLAS_error(routine_name, -7, lda, 0);
      }
      if (incx == 0) {
        BLAS_error(routine_name, -9, incx, 0);
      }
      if (incy == 0) {
        BLAS_error(routine_name, -12, incy, 0);
      }

      /* Set Index Parameters */
      n_i = n;

      if (((uplo == blas_upper) && (order == blas_colmajor)) ||
          ((uplo == blas_lower) && (order == blas_rowmajor))) {
        incaij = 1;                /* increment in first loop */
        incaij2 = lda - 1;        /* increment in second loop */
        astarti = k;                /* does not start on zero element */
      } else {
        incaij = lda - 1;
        incaij2 = 1;
        astarti = 0;                /* start on first element of array */
      }
      /* Adjustment to increments (if any) */
      incx *= 2;
      incy *= 2;
      astarti *= 2;
      incaij *= 2;
      incaij2 *= 2;
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

      FPU_FIX_START;

      /* alpha = 0.  In this case, just return beta * y */
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
        for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
          y_elem[0] = y_i[yi];
          y_elem[1] = y_i[yi + 1];
          {
            /* Compute complex-extra = complex-double * complex-double. */
            double head_t1, tail_t1;
            double head_t2, tail_t2;
            /* Real part */
            compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[0], beta_i[0]);
            compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[1], beta_i[1]);
            head_t2 = -head_t2;
            tail_t2 = -tail_t2;
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
            head_tmp1[0] = head_t1;
            tail_tmp1[0] = tail_t1;
            /* Imaginary part */
            compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[1], beta_i[0]);
            compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[0], beta_i[1]);
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
            head_tmp1[1] = head_t1;
            tail_tmp1[1] = tail_t1;
          }
          y_i[yi] = head_tmp1[0];
          y_i[yi + 1] = head_tmp1[1];
        }
      } else {
        /*  determine the loop interation counts */
        /* maj_first is number of elements done in first loop
           (this will increase by one over each column up to a limit) */
        maxj_first = 0;

        /* maxj_second is number of elements done in
           second loop the first time */
        maxj_second = MIN(k + 1, n_i);

        /*  determine whether we conjugate in first loop or second loop */
        if (uplo == blas_lower) {
          /*  conjugate second loop */

          /* Case alpha == 1. */
          if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {

            if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
              /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
              for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
                head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;
                for (j = 0, aij = astarti, xi = x_starti;
                     j < maxj_first; j++, aij += incaij, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];

                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    double head_e1, tail_e1;
                    double d1;
                    double d2;
                    /* Real part */
                    d1 = (double) a_elem[0] * x_elem[0];
                    d2 = (double) -a_elem[1] * x_elem[1];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[0] = head_e1;
                    tail_prod[0] = tail_e1;
                    /* imaginary part */
                    d1 = (double) a_elem[0] * x_elem[1];
                    d2 = (double) a_elem[1] * x_elem[0];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[1] = head_e1;
                    tail_prod[1] = tail_e1;
                  }
                  {
                    double head_t, tail_t;
                    double head_a, tail_a;
                    double head_b, tail_b;
                    /* Real part */
                    head_a = head_sum[0];
                    tail_a = tail_sum[0];
                    head_b = head_prod[0];
                    tail_b = tail_prod[0];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[0] = head_t;
                    tail_sum[0] = tail_t;
                    /* Imaginary part */
                    head_a = head_sum[1];
                    tail_a = tail_sum[1];
                    head_b = head_prod[1];
                    tail_b = tail_prod[1];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[1] = head_t;
                    tail_sum[1] = tail_t;
                  }
                }
                a_elem[0] = a_i[aij];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  head_prod[0] = (double) x_elem[0] * a_elem[0];
                  tail_prod[0] = 0.0;
                  head_prod[1] = (double) x_elem[1] * a_elem[0];
                  tail_prod[1] = 0.0;
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }
                aij += incaij2;
                xi += incx;
                for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];
                  a_elem[1] = -a_elem[1];
                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    double head_e1, tail_e1;
                    double d1;
                    double d2;
                    /* Real part */
                    d1 = (double) a_elem[0] * x_elem[0];
                    d2 = (double) -a_elem[1] * x_elem[1];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[0] = head_e1;
                    tail_prod[0] = tail_e1;
                    /* imaginary part */
                    d1 = (double) a_elem[0] * x_elem[1];
                    d2 = (double) a_elem[1] * x_elem[0];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[1] = head_e1;
                    tail_prod[1] = tail_e1;
                  }
                  {
                    double head_t, tail_t;
                    double head_a, tail_a;
                    double head_b, tail_b;
                    /* Real part */
                    head_a = head_sum[0];
                    tail_a = tail_sum[0];
                    head_b = head_prod[0];
                    tail_b = tail_prod[0];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[0] = head_t;
                    tail_sum[0] = tail_t;
                    /* Imaginary part */
                    head_a = head_sum[1];
                    tail_a = tail_sum[1];
                    head_b = head_prod[1];
                    tail_b = tail_prod[1];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[1] = head_t;
                    tail_sum[1] = tail_t;
                  }
                }
                y_i[yi] = head_sum[0];
                y_i[yi + 1] = head_sum[1];
                if (i + 1 >= (n_i - k)) {
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
            } else {
              /* Case alpha = 1, but beta != 0.
                 We compute  y  <--- A * x + beta * y */
              for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
                head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

                for (j = 0, aij = astarti, xi = x_starti;
                     j < maxj_first; j++, aij += incaij, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];

                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    double head_e1, tail_e1;
                    double d1;
                    double d2;
                    /* Real part */
                    d1 = (double) a_elem[0] * x_elem[0];
                    d2 = (double) -a_elem[1] * x_elem[1];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[0] = head_e1;
                    tail_prod[0] = tail_e1;
                    /* imaginary part */
                    d1 = (double) a_elem[0] * x_elem[1];
                    d2 = (double) a_elem[1] * x_elem[0];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[1] = head_e1;
                    tail_prod[1] = tail_e1;
                  }
                  {
                    double head_t, tail_t;
                    double head_a, tail_a;
                    double head_b, tail_b;
                    /* Real part */
                    head_a = head_sum[0];
                    tail_a = tail_sum[0];
                    head_b = head_prod[0];
                    tail_b = tail_prod[0];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[0] = head_t;
                    tail_sum[0] = tail_t;
                    /* Imaginary part */
                    head_a = head_sum[1];
                    tail_a = tail_sum[1];
                    head_b = head_prod[1];
                    tail_b = tail_prod[1];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[1] = head_t;
                    tail_sum[1] = tail_t;
                  }
                }
                a_elem[0] = a_i[aij];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  head_prod[0] = (double) x_elem[0] * a_elem[0];
                  tail_prod[0] = 0.0;
                  head_prod[1] = (double) x_elem[1] * a_elem[0];
                  tail_prod[1] = 0.0;
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }
                aij += incaij2;
                xi += incx;
                for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];
                  a_elem[1] = -a_elem[1];
                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    double head_e1, tail_e1;
                    double d1;
                    double d2;
                    /* Real part */
                    d1 = (double) a_elem[0] * x_elem[0];
                    d2 = (double) -a_elem[1] * x_elem[1];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[0] = head_e1;
                    tail_prod[0] = tail_e1;
                    /* imaginary part */
                    d1 = (double) a_elem[0] * x_elem[1];
                    d2 = (double) a_elem[1] * x_elem[0];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[1] = head_e1;
                    tail_prod[1] = tail_e1;
                  }
                  {
                    double head_t, tail_t;
                    double head_a, tail_a;
                    double head_b, tail_b;
                    /* Real part */
                    head_a = head_sum[0];
                    tail_a = tail_sum[0];
                    head_b = head_prod[0];
                    tail_b = tail_prod[0];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[0] = head_t;
                    tail_sum[0] = tail_t;
                    /* Imaginary part */
                    head_a = head_sum[1];
                    tail_a = tail_sum[1];
                    head_b = head_prod[1];
                    tail_b = tail_prod[1];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[1] = head_t;
                    tail_sum[1] = tail_t;
                  }
                }
                y_elem[0] = y_i[yi];
                y_elem[1] = y_i[yi + 1];
                {
                  /* Compute complex-extra = complex-double * complex-double. */
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  /* Real part */
                  compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[0], beta_i[0]);
                  compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[1], beta_i[1]);
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp2[0] = head_t1;
                  tail_tmp2[0] = tail_t1;
                  /* Imaginary part */
                  compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[1], beta_i[0]);
                  compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[0], beta_i[1]);
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp2[1] = head_t1;
                  tail_tmp2[1] = tail_t1;
                }
                head_tmp1[0] = head_sum[0];
                tail_tmp1[0] = tail_sum[0];
                head_tmp1[1] = head_sum[1];
                tail_tmp1[1] = tail_sum[1];
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_tmp2[0];
                  tail_a = tail_tmp2[0];
                  head_b = head_tmp1[0];
                  tail_b = tail_tmp1[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_tmp1[0] = head_t;
                  tail_tmp1[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_tmp2[1];
                  tail_a = tail_tmp2[1];
                  head_b = head_tmp1[1];
                  tail_b = tail_tmp1[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_tmp1[1] = head_t;
                  tail_tmp1[1] = tail_t;
                }
                y_i[yi] = head_tmp1[0];
                y_i[yi + 1] = head_tmp1[1];
                if (i + 1 >= (n_i - k)) {
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
            }
          } else {
            /* The most general form,   y <--- alpha * A * x + beta * y */
            for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
              head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

              for (j = 0, aij = astarti, xi = x_starti;
                   j < maxj_first; j++, aij += incaij, xi += incx) {
                a_elem[0] = a_i[aij];
                a_elem[1] = a_i[aij + 1];

                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  double head_e1, tail_e1;
                  double d1;
                  double d2;
                  /* Real part */
                  d1 = (double) a_elem[0] * x_elem[0];
                  d2 = (double) -a_elem[1] * x_elem[1];
                  compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                  head_prod[0] = head_e1;
                  tail_prod[0] = tail_e1;
                  /* imaginary part */
                  d1 = (double) a_elem[0] * x_elem[1];
                  d2 = (double) a_elem[1] * x_elem[0];
                  compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                  head_prod[1] = head_e1;
                  tail_prod[1] = tail_e1;
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }
              }
              a_elem[0] = a_i[aij];
              x_elem[0] = x_i[xi];
              x_elem[1] = x_i[xi + 1];
              {
                head_prod[0] = (double) x_elem[0] * a_elem[0];
                tail_prod[0] = 0.0;
                head_prod[1] = (double) x_elem[1] * a_elem[0];
                tail_prod[1] = 0.0;
              }
              {
                double head_t, tail_t;
                double head_a, tail_a;
                double head_b, tail_b;
                /* Real part */
                head_a = head_sum[0];
                tail_a = tail_sum[0];
                head_b = head_prod[0];
                tail_b = tail_prod[0];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_sum[0] = head_t;
                tail_sum[0] = tail_t;
                /* Imaginary part */
                head_a = head_sum[1];
                tail_a = tail_sum[1];
                head_b = head_prod[1];
                tail_b = tail_prod[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_sum[1] = head_t;
                tail_sum[1] = tail_t;
              }
              aij += incaij2;
              xi += incx;
              for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                a_elem[0] = a_i[aij];
                a_elem[1] = a_i[aij + 1];
                a_elem[1] = -a_elem[1];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  double head_e1, tail_e1;
                  double d1;
                  double d2;
                  /* Real part */
                  d1 = (double) a_elem[0] * x_elem[0];
                  d2 = (double) -a_elem[1] * x_elem[1];
                  compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                  head_prod[0] = head_e1;
                  tail_prod[0] = tail_e1;
                  /* imaginary part */
                  d1 = (double) a_elem[0] * x_elem[1];
                  d2 = (double) a_elem[1] * x_elem[0];
                  compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                  head_prod[1] = head_e1;
                  tail_prod[1] = tail_e1;
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }
              }
              y_elem[0] = y_i[yi];
              y_elem[1] = y_i[yi + 1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[0], beta_i[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[1], beta_i[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp2[0] = head_t1;
                tail_tmp2[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[1], beta_i[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[0], beta_i[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp2[1] = head_t1;
                tail_tmp2[1] = tail_t1;
              }
              {
                /* Compute complex-extra = complex-extra * complex-double. */
                double head_a0, tail_a0;
                double head_a1, tail_a1;
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                head_a0 = head_sum[0];
                tail_a0 = tail_sum[0];
                head_a1 = head_sum[1];
                tail_a1 = tail_sum[1];
                /* real part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp1[0] = head_t1;
                tail_tmp1[0] = tail_t1;
                /* imaginary part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp1[1] = head_t1;
                tail_tmp1[1] = tail_t1;
              }

              {
                double head_t, tail_t;
                double head_a, tail_a;
                double head_b, tail_b;
                /* Real part */
                head_a = head_tmp2[0];
                tail_a = tail_tmp2[0];
                head_b = head_tmp1[0];
                tail_b = tail_tmp1[0];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_tmp1[0] = head_t;
                tail_tmp1[0] = tail_t;
                /* Imaginary part */
                head_a = head_tmp2[1];
                tail_a = tail_tmp2[1];
                head_b = head_tmp1[1];
                tail_b = tail_tmp1[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_tmp1[1] = head_t;
                tail_tmp1[1] = tail_t;
              }
              y_i[yi] = head_tmp1[0];
              y_i[yi + 1] = head_tmp1[1];
              if (i + 1 >= (n_i - k)) {
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
          }
        } else {
          /*  conjugate first loop */

          /* Case alpha == 1. */
          if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {

            if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
              /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
              for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
                head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;
                for (j = 0, aij = astarti, xi = x_starti;
                     j < maxj_first; j++, aij += incaij, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];
                  a_elem[1] = -a_elem[1];
                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    double head_e1, tail_e1;
                    double d1;
                    double d2;
                    /* Real part */
                    d1 = (double) a_elem[0] * x_elem[0];
                    d2 = (double) -a_elem[1] * x_elem[1];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[0] = head_e1;
                    tail_prod[0] = tail_e1;
                    /* imaginary part */
                    d1 = (double) a_elem[0] * x_elem[1];
                    d2 = (double) a_elem[1] * x_elem[0];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[1] = head_e1;
                    tail_prod[1] = tail_e1;
                  }
                  {
                    double head_t, tail_t;
                    double head_a, tail_a;
                    double head_b, tail_b;
                    /* Real part */
                    head_a = head_sum[0];
                    tail_a = tail_sum[0];
                    head_b = head_prod[0];
                    tail_b = tail_prod[0];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[0] = head_t;
                    tail_sum[0] = tail_t;
                    /* Imaginary part */
                    head_a = head_sum[1];
                    tail_a = tail_sum[1];
                    head_b = head_prod[1];
                    tail_b = tail_prod[1];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[1] = head_t;
                    tail_sum[1] = tail_t;
                  }
                }
                a_elem[0] = a_i[aij];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  head_prod[0] = (double) x_elem[0] * a_elem[0];
                  tail_prod[0] = 0.0;
                  head_prod[1] = (double) x_elem[1] * a_elem[0];
                  tail_prod[1] = 0.0;
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }
                aij += incaij2;
                xi += incx;
                for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];

                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    double head_e1, tail_e1;
                    double d1;
                    double d2;
                    /* Real part */
                    d1 = (double) a_elem[0] * x_elem[0];
                    d2 = (double) -a_elem[1] * x_elem[1];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[0] = head_e1;
                    tail_prod[0] = tail_e1;
                    /* imaginary part */
                    d1 = (double) a_elem[0] * x_elem[1];
                    d2 = (double) a_elem[1] * x_elem[0];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[1] = head_e1;
                    tail_prod[1] = tail_e1;
                  }
                  {
                    double head_t, tail_t;
                    double head_a, tail_a;
                    double head_b, tail_b;
                    /* Real part */
                    head_a = head_sum[0];
                    tail_a = tail_sum[0];
                    head_b = head_prod[0];
                    tail_b = tail_prod[0];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[0] = head_t;
                    tail_sum[0] = tail_t;
                    /* Imaginary part */
                    head_a = head_sum[1];
                    tail_a = tail_sum[1];
                    head_b = head_prod[1];
                    tail_b = tail_prod[1];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[1] = head_t;
                    tail_sum[1] = tail_t;
                  }
                }
                y_i[yi] = head_sum[0];
                y_i[yi + 1] = head_sum[1];
                if (i + 1 >= (n_i - k)) {
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
            } else {
              /* Case alpha = 1, but beta != 0.
                 We compute  y  <--- A * x + beta * y */
              for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
                head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

                for (j = 0, aij = astarti, xi = x_starti;
                     j < maxj_first; j++, aij += incaij, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];
                  a_elem[1] = -a_elem[1];
                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    double head_e1, tail_e1;
                    double d1;
                    double d2;
                    /* Real part */
                    d1 = (double) a_elem[0] * x_elem[0];
                    d2 = (double) -a_elem[1] * x_elem[1];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[0] = head_e1;
                    tail_prod[0] = tail_e1;
                    /* imaginary part */
                    d1 = (double) a_elem[0] * x_elem[1];
                    d2 = (double) a_elem[1] * x_elem[0];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[1] = head_e1;
                    tail_prod[1] = tail_e1;
                  }
                  {
                    double head_t, tail_t;
                    double head_a, tail_a;
                    double head_b, tail_b;
                    /* Real part */
                    head_a = head_sum[0];
                    tail_a = tail_sum[0];
                    head_b = head_prod[0];
                    tail_b = tail_prod[0];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[0] = head_t;
                    tail_sum[0] = tail_t;
                    /* Imaginary part */
                    head_a = head_sum[1];
                    tail_a = tail_sum[1];
                    head_b = head_prod[1];
                    tail_b = tail_prod[1];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[1] = head_t;
                    tail_sum[1] = tail_t;
                  }
                }
                a_elem[0] = a_i[aij];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  head_prod[0] = (double) x_elem[0] * a_elem[0];
                  tail_prod[0] = 0.0;
                  head_prod[1] = (double) x_elem[1] * a_elem[0];
                  tail_prod[1] = 0.0;
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }
                aij += incaij2;
                xi += incx;
                for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                  a_elem[0] = a_i[aij];
                  a_elem[1] = a_i[aij + 1];

                  x_elem[0] = x_i[xi];
                  x_elem[1] = x_i[xi + 1];
                  {
                    double head_e1, tail_e1;
                    double d1;
                    double d2;
                    /* Real part */
                    d1 = (double) a_elem[0] * x_elem[0];
                    d2 = (double) -a_elem[1] * x_elem[1];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[0] = head_e1;
                    tail_prod[0] = tail_e1;
                    /* imaginary part */
                    d1 = (double) a_elem[0] * x_elem[1];
                    d2 = (double) a_elem[1] * x_elem[0];
                    compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                    head_prod[1] = head_e1;
                    tail_prod[1] = tail_e1;
                  }
                  {
                    double head_t, tail_t;
                    double head_a, tail_a;
                    double head_b, tail_b;
                    /* Real part */
                    head_a = head_sum[0];
                    tail_a = tail_sum[0];
                    head_b = head_prod[0];
                    tail_b = tail_prod[0];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[0] = head_t;
                    tail_sum[0] = tail_t;
                    /* Imaginary part */
                    head_a = head_sum[1];
                    tail_a = tail_sum[1];
                    head_b = head_prod[1];
                    tail_b = tail_prod[1];
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                    head_sum[1] = head_t;
                    tail_sum[1] = tail_t;
                  }
                }
                y_elem[0] = y_i[yi];
                y_elem[1] = y_i[yi + 1];
                {
                  /* Compute complex-extra = complex-double * complex-double. */
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  /* Real part */
                  compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[0], beta_i[0]);
                  compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[1], beta_i[1]);
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp2[0] = head_t1;
                  tail_tmp2[0] = tail_t1;
                  /* Imaginary part */
                  compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[1], beta_i[0]);
                  compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[0], beta_i[1]);
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp2[1] = head_t1;
                  tail_tmp2[1] = tail_t1;
                }
                head_tmp1[0] = head_sum[0];
                tail_tmp1[0] = tail_sum[0];
                head_tmp1[1] = head_sum[1];
                tail_tmp1[1] = tail_sum[1];
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_tmp2[0];
                  tail_a = tail_tmp2[0];
                  head_b = head_tmp1[0];
                  tail_b = tail_tmp1[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_tmp1[0] = head_t;
                  tail_tmp1[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_tmp2[1];
                  tail_a = tail_tmp2[1];
                  head_b = head_tmp1[1];
                  tail_b = tail_tmp1[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_tmp1[1] = head_t;
                  tail_tmp1[1] = tail_t;
                }
                y_i[yi] = head_tmp1[0];
                y_i[yi + 1] = head_tmp1[1];
                if (i + 1 >= (n_i - k)) {
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
            }
          } else {
            /* The most general form,   y <--- alpha * A * x + beta * y */
            for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
              head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

              for (j = 0, aij = astarti, xi = x_starti;
                   j < maxj_first; j++, aij += incaij, xi += incx) {
                a_elem[0] = a_i[aij];
                a_elem[1] = a_i[aij + 1];
                a_elem[1] = -a_elem[1];
                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  double head_e1, tail_e1;
                  double d1;
                  double d2;
                  /* Real part */
                  d1 = (double) a_elem[0] * x_elem[0];
                  d2 = (double) -a_elem[1] * x_elem[1];
                  compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                  head_prod[0] = head_e1;
                  tail_prod[0] = tail_e1;
                  /* imaginary part */
                  d1 = (double) a_elem[0] * x_elem[1];
                  d2 = (double) a_elem[1] * x_elem[0];
                  compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                  head_prod[1] = head_e1;
                  tail_prod[1] = tail_e1;
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }
              }
              a_elem[0] = a_i[aij];
              x_elem[0] = x_i[xi];
              x_elem[1] = x_i[xi + 1];
              {
                head_prod[0] = (double) x_elem[0] * a_elem[0];
                tail_prod[0] = 0.0;
                head_prod[1] = (double) x_elem[1] * a_elem[0];
                tail_prod[1] = 0.0;
              }
              {
                double head_t, tail_t;
                double head_a, tail_a;
                double head_b, tail_b;
                /* Real part */
                head_a = head_sum[0];
                tail_a = tail_sum[0];
                head_b = head_prod[0];
                tail_b = tail_prod[0];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_sum[0] = head_t;
                tail_sum[0] = tail_t;
                /* Imaginary part */
                head_a = head_sum[1];
                tail_a = tail_sum[1];
                head_b = head_prod[1];
                tail_b = tail_prod[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_sum[1] = head_t;
                tail_sum[1] = tail_t;
              }
              aij += incaij2;
              xi += incx;
              for (j = 1; j < maxj_second; j++, aij += incaij2, xi += incx) {
                a_elem[0] = a_i[aij];
                a_elem[1] = a_i[aij + 1];

                x_elem[0] = x_i[xi];
                x_elem[1] = x_i[xi + 1];
                {
                  double head_e1, tail_e1;
                  double d1;
                  double d2;
                  /* Real part */
                  d1 = (double) a_elem[0] * x_elem[0];
                  d2 = (double) -a_elem[1] * x_elem[1];
                  compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                  head_prod[0] = head_e1;
                  tail_prod[0] = tail_e1;
                  /* imaginary part */
                  d1 = (double) a_elem[0] * x_elem[1];
                  d2 = (double) a_elem[1] * x_elem[0];
                  compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                  head_prod[1] = head_e1;
                  tail_prod[1] = tail_e1;
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }
              }
              y_elem[0] = y_i[yi];
              y_elem[1] = y_i[yi + 1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[0], beta_i[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[1], beta_i[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp2[0] = head_t1;
                tail_tmp2[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[1], beta_i[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[0], beta_i[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp2[1] = head_t1;
                tail_tmp2[1] = tail_t1;
              }
              {
                /* Compute complex-extra = complex-extra * complex-double. */
                double head_a0, tail_a0;
                double head_a1, tail_a1;
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                head_a0 = head_sum[0];
                tail_a0 = tail_sum[0];
                head_a1 = head_sum[1];
                tail_a1 = tail_sum[1];
                /* real part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp1[0] = head_t1;
                tail_tmp1[0] = tail_t1;
                /* imaginary part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp1[1] = head_t1;
                tail_tmp1[1] = tail_t1;
              }

              {
                double head_t, tail_t;
                double head_a, tail_a;
                double head_b, tail_b;
                /* Real part */
                head_a = head_tmp2[0];
                tail_a = tail_tmp2[0];
                head_b = head_tmp1[0];
                tail_b = tail_tmp1[0];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_tmp1[0] = head_t;
                tail_tmp1[0] = tail_t;
                /* Imaginary part */
                head_a = head_tmp2[1];
                tail_a = tail_tmp2[1];
                head_b = head_tmp1[1];
                tail_b = tail_tmp1[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_tmp1[1] = head_t;
                tail_tmp1[1] = tail_t;
              }
              y_i[yi] = head_tmp1[0];
              y_i[yi + 1] = head_tmp1[1];
              if (i + 1 >= (n_i - k)) {
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
          }
        }
      }
      FPU_FIX_STOP;

      break;
    }

  default:
    BLAS_error(routine_name, -13, prec, 0);
    break;
  }
} /* end XBLAS::hbmv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_HBMV_HPP
