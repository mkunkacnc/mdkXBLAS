#ifndef XBLAS_GE_SUM_MV_HPP
#define XBLAS_GE_SUM_MV_HPP

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
constexpr void ge_sum_mv(blas_order_type order,
                         IdxType m,
                         IdxType n,
                         T alpha,
                         const A *a,
                         IdxType lda,
                         const X *x,
                         IdxType incx,
                         T beta,
                         const A *b,
                         IdxType ldb,
                         T *y,
                         IdxType incy)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * A * x + beta * B * y,
 *     where A, B are general matrices.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Order of A; row or column major
 *
 * m      (input) IdxType
 *        Row Dimension of A, B, length of output vector y
 *
 * n      (input) IdxType
 *        Column Dimension of A, B and the length of vector x
 *
 * alpha  (input) T
 *
 * A      (input) const A*
 *
 * lda    (input) IdxType
 *        Leading dimension of A
 *
 * x      (input) const X*
 *
 * incx   (input) IdxType
 *        The stride for vector x.
 *
 * beta   (input) T
 *
 * b      (input) const A*
 *
 * ldb    (input) IdxType
 *        Leading dimension of B
 *
 * y      (input/output) T*
 *
 * incy   (input) IdxType
 *        The stride for vector y.
 *
 */
{
  static const char routine_name[] = "XBLAS::ge_sum_mv";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  /* m is number of rows */
  /* n is number of columns */

  if (m == 0 || n == 0)
    return;

  /* all error calls */
  IdxType lda_min, incai, incbi, incaij, incbij;
  if (order == blas_rowmajor) {
    lda_min = n;
    incai = lda; /* row stride */
    incbi = ldb;
    incaij = 1; /* column stride */
    incbij = 1;
  } else if (order == blas_colmajor) {
    lda_min = m;
    incai = 1; /*row stride */
    incbi = 1;
    incaij = lda; /* column stride */
    incbij = ldb;
  } else {
    /* error, order not blas_colmajor not blas_rowmajor */
    BLAS_error(routine_name, -1, order, 0);
    return;
  }

  if (m < 0)
    BLAS_error(routine_name, -2, m, 0);
  else if (n < 0)
    BLAS_error(routine_name, -3, n, 0);
  if (lda < lda_min)
    BLAS_error(routine_name, -6, lda, 0);
  else if (ldb < lda_min)
    BLAS_error(routine_name, -11, ldb, 0);
  else if (incx == 0)
    BLAS_error(routine_name, -8, incx, 0);
  else if (incy == 0)
    BLAS_error(routine_name, -13, incy, 0);

  IdxType incxi = incx;
  IdxType incyi = incy;

  IdxType x_starti;
  if (incxi > 0)
    x_starti = 0;
  else
    x_starti = (1 - n) * incxi;

  IdxType y_starti;
  if (incyi > 0)
    y_starti = 0;
  else
    y_starti = (1 - m) * incyi;

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if (alpha == T(0)) {
    if (beta == T(0)) {
      /* alpha, beta are 0 */
      for (IdxType i = 0, yi = y_starti; i < m; ++i, yi += incyi) {
        y[yi] = T(0);
      }
    } else if (beta == T(1)) {
      /* alpha is 0, beta is 1 */
      IdxType bi = 0;
      for (IdxType i = 0, yi = y_starti; i < m; ++i, yi += incyi) {
        PrdType sumB = impl::zero_v<PrdType>;
        IdxType bij = bi;
        for (IdxType j = 0, xi = x_starti; j < n; ++j, xi += incxi) {
          sumB += impl::mul<PrdType>(b[bij], x[xi]);
          bij += incbij;
        }
        /* now put the result into y */
        y[yi] = impl::to<T>(sumB);
        bi += incbi;
      }
    } else {
      /* alpha is 0, beta not 1 nor 0 */
      IdxType bi = 0;
      for (IdxType i = 0, yi = y_starti; i < m; ++i, yi += incyi) {
        PrdType sumB = impl::zero_v<PrdType>;
        IdxType bij = bi;
        for (IdxType j = 0, xi = x_starti; j < n; ++j, xi += incxi) {
          sumB += impl::mul<PrdType>(b[bij], x[xi]);
          bij += incbij;
        }
        /* now put the result into y */
        TmpType tmp1 = impl::mul<TmpType>(sumB, beta);
        y[yi] = impl::to<T>(tmp1);
        bi += incbi;
      }
    }
  } else if (alpha == T(1)) {
    if (beta == T(0)) {
      /* alpha is 1, beta is 0 */
      IdxType ai = 0;
      for (IdxType i = 0, yi = y_starti; i < m; ++i, yi += incyi) {
        PrdType sumA = impl::zero_v<PrdType>;
        IdxType aij = ai;
        for (IdxType j = 0, xi = x_starti; j < n; ++j, xi += incxi) {
          sumA += impl::mul<PrdType>(a[aij], x[xi]);
          aij += incaij;
        }
        /* now put the result into y */
        y[yi] = impl::to<T>(sumA);
        ai += incai;

      }
    } else if (beta == T(1)) {
      /* alpha is 1, beta is 1 */
      IdxType ai = 0;
      IdxType bi = 0;
      for (IdxType i = 0, yi = y_starti; i < m; ++i, yi += incyi) {
        PrdType sumA = impl::zero_v<PrdType>;
        IdxType aij = ai;
        PrdType sumB = impl::zero_v<PrdType>;
        IdxType bij = bi;
        for (IdxType j = 0, xi = x_starti; j < n; ++j, xi += incxi) {
          sumA += impl::mul<PrdType>(a[aij], x[xi]);
          aij += incaij;
          sumB += impl::mul<PrdType>(b[bij], x[xi]);
          bij += incbij;
        }
        /* now put the result into y */
        TmpType tmp1 = sumA;
        TmpType tmp2 = sumB;
        y[yi] = impl::add<T>(tmp1, tmp2);
        ai += incai;
        bi += incbi;
      }
    } else {
      /* alpha is 1, beta is other */
      IdxType ai = 0;
      IdxType bi = 0;
      for (IdxType i = 0, yi = y_starti; i < m; ++i, yi += incyi) {
        PrdType sumA = impl::zero_v<PrdType>;
        IdxType aij = ai;
        PrdType sumB = impl::zero_v<PrdType>;
        IdxType bij = bi;
        for (IdxType j = 0, xi = x_starti; j < n; ++j, xi += incxi) {
          sumA += impl::mul<PrdType>(a[aij], x[xi]);
          aij += incaij;
          sumB += impl::mul<PrdType>(b[bij], x[xi]);
          bij += incbij;
        }
        /* now put the result into y */
        TmpType tmp1 = sumA;
        TmpType tmp2 = impl::mul<TmpType>(sumB, beta);
        y[yi] = impl::add<T>(tmp1, tmp2);
        ai += incai;
        bi += incbi;
      }
    }
  } else {
    if (beta == T(0)) {
      /* alpha is other, beta is 0 */
      IdxType ai = 0;
      for (IdxType i = 0, yi = y_starti; i < m; ++i, yi += incyi) {
        PrdType sumA = impl::zero_v<PrdType>;
        IdxType aij = ai;
        for (IdxType j = 0, xi = x_starti; j < n; ++j, xi += incxi) {
          sumA += impl::mul<PrdType>(a[aij], x[xi]);
          aij += incaij;
        }
        /* now put the result into y */
        TmpType tmp1 = impl::mul<TmpType>(sumA, alpha);
        y[yi] = impl::to<T>(tmp1);
        ai += incai;
      }
    } else if (beta == T(1)) {
      /* alpha is other, beta is 1 */
      IdxType ai = 0;
      IdxType bi = 0;
      for (IdxType i = 0, yi = y_starti; i < m; ++i, yi += incyi) {
        PrdType sumA = impl::zero_v<PrdType>;
        IdxType aij = ai;
        PrdType sumB = impl::zero_v<PrdType>;
        IdxType bij = bi;
        for (IdxType j = 0, xi = x_starti; j < n; ++j, xi += incxi) {
          sumA += impl::mul<PrdType>(a[aij], x[xi]);
          aij += incaij;
          sumB += impl::mul<PrdType>(b[bij], x[xi]);
          bij += incbij;
        }
        /* now put the result into y */
        TmpType tmp1 = impl::mul<TmpType>(sumA, alpha);
        TmpType tmp2 = sumB;
        y[yi] = impl::add<T>(tmp1, tmp2);
        ai += incai;
        bi += incbi;
      }
    } else {
      /* most general form, alpha, beta are other */
      IdxType ai = 0;
      IdxType bi = 0;
      for (IdxType i = 0, yi = y_starti; i < m; ++i, yi += incyi) {
        PrdType sumA = impl::zero_v<PrdType>;
        IdxType aij = ai;
        PrdType sumB = impl::zero_v<PrdType>;
        IdxType bij = bi;
        for (IdxType j = 0, xi = x_starti; j < n; ++j, xi += incxi) {
          sumA += impl::mul<PrdType>(a[aij], x[xi]);
          aij += incaij;
          sumB += impl::mul<PrdType>(b[bij], x[xi]);
          bij += incbij;
        }
        /* now put the result into y */
        TmpType tmp1 = impl::mul<TmpType>(sumA, alpha);
        TmpType tmp2 = impl::mul<TmpType>(sumB, beta);
        y[yi] = impl::add<T>(tmp1, tmp2);
        ai += incai;
        bi += incbi;
      }
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::ge_sum_mv */

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
constexpr void ge_sum_mv_x(blas_order_type order,
                           IdxType m,
                           IdxType n,
                           T alpha,
                           const A *a,
                           IdxType lda,
                           const X *x,
                           IdxType incx,
                           T beta,
                           const A *b,
                           IdxType ldb,
                           T *y,
                           IdxType incy,
                           blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * A * x + beta * B * y,
 *     where A, B are general matrices.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Order of A; row or column major
 *
 * m      (input) int
 *        Row Dimension of A, B, length of output vector y
 *
 * n      (input) int
 *        Column Dimension of A, B and the length of vector x
 *
 * alpha  (input) double
 *
 * A      (input) const double*
 *
 * lda    (input) int
 *        Leading dimension of A
 *
 * x      (input) const float*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) double
 *
 * b      (input) const double*
 *
 * ldb    (input) int
 *        Leading dimension of B
 *
 * y      (input/output) double*
 *
 * incy   (input) int
 *        The stride for vector y.
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
  /* Routine name */
  static const char routine_name[] = "BLAS_dge_sum_mv_d_s";
  switch (prec) {
  case blas_prec_single:
  case blas_prec_indigenous:
  case blas_prec_double:
    {
      int i, j;
      int xi, yi;
      int x_starti, y_starti, incxi, incyi;
      int lda_min;
      int ai;
      int incai;
      int aij;
      int incaij;
      int bi;
      int incbi;
      int bij;
      int incbij;

      const double *a_i = a;
      const double *b_i = b;
      const float *x_i = x;
      double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      double a_elem;
      double b_elem;
      float x_elem;
      double prod;
      double sumA;
      double sumB;
      double tmp1;
      double tmp2;



      /* m is number of rows */
      /* n is number of columns */

      if (m == 0 || n == 0)
        return;


      /* all error calls */
      if (order == blas_rowmajor) {
        lda_min = n;
        incai = lda;                /* row stride */
        incbi = ldb;
        incbij = incaij = 1;        /* column stride */
      } else if (order == blas_colmajor) {
        lda_min = m;
        incai = incbi = 1;        /*row stride */
        incaij = lda;                /* column stride */
        incbij = ldb;
      } else {
        /* error, order not blas_colmajor not blas_rowmajor */
        BLAS_error(routine_name, -1, order, 0);
        return;
      }

      if (m < 0)
        BLAS_error(routine_name, -2, m, 0);
      else if (n < 0)
        BLAS_error(routine_name, -3, n, 0);
      if (lda < lda_min)
        BLAS_error(routine_name, -6, lda, 0);
      else if (ldb < lda_min)
        BLAS_error(routine_name, -11, ldb, 0);
      else if (incx == 0)
        BLAS_error(routine_name, -8, incx, 0);
      else if (incy == 0)
        BLAS_error(routine_name, -13, incy, 0);

      incxi = incx;
      incyi = incy;







      if (incxi > 0)
        x_starti = 0;
      else
        x_starti = (1 - n) * incxi;

      if (incyi > 0)
        y_starti = 0;
      else
        y_starti = (1 - m) * incyi;



      if (alpha_i == 0.0) {
        if (beta_i == 0.0) {
          /* alpha, beta are 0.0 */
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            y_i[yi] = 0.0;
          }
        } else if (beta_i == 1.0) {
          /* alpha is 0.0, beta is 1.0 */


          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {

            sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];

              b_elem = b_i[bij];
              prod = b_elem * x_elem;
              sumB = sumB + prod;
              bij += incbij;
            }
            /* now put the result into y_i */
            y_i[yi] = sumB;

            bi += incbi;
          }
        } else {
          /* alpha is 0.0, beta not 1.0 nor 0.0 */


          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {

            sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];

              b_elem = b_i[bij];
              prod = b_elem * x_elem;
              sumB = sumB + prod;
              bij += incbij;
            }
            /* now put the result into y_i */
            tmp1 = sumB * beta_i;
            y_i[yi] = tmp1;

            bi += incbi;
          }
        }
      } else if (alpha_i == 1.0) {
        if (beta_i == 0.0) {
          /* alpha is 1.0, beta is 0.0 */

          ai = 0;

          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            sumA = 0.0;
            aij = ai;

            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              prod = a_elem * x_elem;
              sumA = sumA + prod;
              aij += incaij;

            }
            /* now put the result into y_i */
            y_i[yi] = sumA;
            ai += incai;

          }
        } else if (beta_i == 1.0) {
          /* alpha is 1.0, beta is 1.0 */

          ai = 0;
          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            sumA = 0.0;
            aij = ai;
            sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              prod = a_elem * x_elem;
              sumA = sumA + prod;
              aij += incaij;
              b_elem = b_i[bij];
              prod = b_elem * x_elem;
              sumB = sumB + prod;
              bij += incbij;
            }
            /* now put the result into y_i */
            tmp1 = sumA;
            tmp2 = sumB;
            tmp1 = tmp1 + tmp2;
            y_i[yi] = tmp1;
            ai += incai;
            bi += incbi;
          }
        } else {
          /* alpha is 1.0, beta is other */

          ai = 0;
          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            sumA = 0.0;
            aij = ai;
            sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              prod = a_elem * x_elem;
              sumA = sumA + prod;
              aij += incaij;
              b_elem = b_i[bij];
              prod = b_elem * x_elem;
              sumB = sumB + prod;
              bij += incbij;
            }
            /* now put the result into y_i */
            tmp1 = sumA;
            tmp2 = sumB * beta_i;
            tmp1 = tmp1 + tmp2;
            y_i[yi] = tmp1;
            ai += incai;
            bi += incbi;
          }
        }
      } else {
        if (beta_i == 0.0) {
          /* alpha is other, beta is 0.0 */

          ai = 0;

          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            sumA = 0.0;
            aij = ai;

            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              prod = a_elem * x_elem;
              sumA = sumA + prod;
              aij += incaij;

            }
            /* now put the result into y_i */
            tmp1 = sumA * alpha_i;
            y_i[yi] = tmp1;
            ai += incai;

          }
        } else if (beta_i == 1.0) {
          /* alpha is other, beta is 1.0 */

          ai = 0;
          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            sumA = 0.0;
            aij = ai;
            sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              prod = a_elem * x_elem;
              sumA = sumA + prod;
              aij += incaij;
              b_elem = b_i[bij];
              prod = b_elem * x_elem;
              sumB = sumB + prod;
              bij += incbij;
            }
            /* now put the result into y_i */
            tmp1 = sumA * alpha_i;
            tmp2 = sumB;
            tmp1 = tmp1 + tmp2;
            y_i[yi] = tmp1;
            ai += incai;
            bi += incbi;
          }
        } else {
          /* most general form, alpha, beta are other */

          ai = 0;
          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            sumA = 0.0;
            aij = ai;
            sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              prod = a_elem * x_elem;
              sumA = sumA + prod;
              aij += incaij;
              b_elem = b_i[bij];
              prod = b_elem * x_elem;
              sumB = sumB + prod;
              bij += incbij;
            }
            /* now put the result into y_i */
            tmp1 = sumA * alpha_i;
            tmp2 = sumB * beta_i;
            tmp1 = tmp1 + tmp2;
            y_i[yi] = tmp1;
            ai += incai;
            bi += incbi;
          }
        }
      }

    }
    break;

  case blas_prec_extra:
    {
      int i, j;
      int xi, yi;
      int x_starti, y_starti, incxi, incyi;
      int lda_min;
      int ai;
      int incai;
      int aij;
      int incaij;
      int bi;
      int incbi;
      int bij;
      int incbij;

      const double *a_i = a;
      const double *b_i = b;
      const float *x_i = x;
      double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      double a_elem;
      double b_elem;
      float x_elem;
      double head_prod, tail_prod;
      double head_sumA, tail_sumA;
      double head_sumB, tail_sumB;
      double head_tmp1, tail_tmp1;
      double head_tmp2, tail_tmp2;

      FPU_FIX_DECL;

      /* m is number of rows */
      /* n is number of columns */

      if (m == 0 || n == 0)
        return;


      /* all error calls */
      if (order == blas_rowmajor) {
        lda_min = n;
        incai = lda;                /* row stride */
        incbi = ldb;
        incbij = incaij = 1;        /* column stride */
      } else if (order == blas_colmajor) {
        lda_min = m;
        incai = incbi = 1;        /*row stride */
        incaij = lda;                /* column stride */
        incbij = ldb;
      } else {
        /* error, order not blas_colmajor not blas_rowmajor */
        BLAS_error(routine_name, -1, order, 0);
        return;
      }

      if (m < 0)
        BLAS_error(routine_name, -2, m, 0);
      else if (n < 0)
        BLAS_error(routine_name, -3, n, 0);
      if (lda < lda_min)
        BLAS_error(routine_name, -6, lda, 0);
      else if (ldb < lda_min)
        BLAS_error(routine_name, -11, ldb, 0);
      else if (incx == 0)
        BLAS_error(routine_name, -8, incx, 0);
      else if (incy == 0)
        BLAS_error(routine_name, -13, incy, 0);

      incxi = incx;
      incyi = incy;







      if (incxi > 0)
        x_starti = 0;
      else
        x_starti = (1 - n) * incxi;

      if (incyi > 0)
        y_starti = 0;
      else
        y_starti = (1 - m) * incyi;

      FPU_FIX_START;

      if (alpha_i == 0.0) {
        if (beta_i == 0.0) {
          /* alpha, beta are 0.0 */
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            y_i[yi] = 0.0;
          }
        } else if (beta_i == 1.0) {
          /* alpha is 0.0, beta is 1.0 */


          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {

            head_sumB = tail_sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];

              b_elem = b_i[bij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            y_i[yi] = head_sumB;

            bi += incbi;
          }
        } else {
          /* alpha is 0.0, beta not 1.0 nor 0.0 */


          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {

            head_sumB = tail_sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];

              b_elem = b_i[bij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_sumB, tail_sumB, beta_i);
            y_i[yi] = head_tmp1;

            bi += incbi;
          }
        }
      } else if (alpha_i == 1.0) {
        if (beta_i == 0.0) {
          /* alpha is 1.0, beta is 0.0 */

          ai = 0;

          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            head_sumA = tail_sumA = 0.0;
            aij = ai;

            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;

            }
            /* now put the result into y_i */
            y_i[yi] = head_sumA;
            ai += incai;

          }
        } else if (beta_i == 1.0) {
          /* alpha is 1.0, beta is 1.0 */

          ai = 0;
          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            head_sumA = tail_sumA = 0.0;
            aij = ai;
            head_sumB = tail_sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;
              b_elem = b_i[bij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            head_tmp1 = head_sumA;
            tail_tmp1 = tail_sumA;
            head_tmp2 = head_sumB;
            tail_tmp2 = tail_sumB;
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
            y_i[yi] = head_tmp1;
            ai += incai;
            bi += incbi;
          }
        } else {
          /* alpha is 1.0, beta is other */

          ai = 0;
          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            head_sumA = tail_sumA = 0.0;
            aij = ai;
            head_sumB = tail_sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;
              b_elem = b_i[bij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            head_tmp1 = head_sumA;
            tail_tmp1 = tail_sumA;
            compute_doubledouble_eq_doubledouble_mul_double(&head_tmp2, &tail_tmp2, head_sumB, tail_sumB, beta_i);
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
            y_i[yi] = head_tmp1;
            ai += incai;
            bi += incbi;
          }
        }
      } else {
        if (beta_i == 0.0) {
          /* alpha is other, beta is 0.0 */

          ai = 0;

          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            head_sumA = tail_sumA = 0.0;
            aij = ai;

            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;

            }
            /* now put the result into y_i */
            compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_sumA, tail_sumA, alpha_i);
            y_i[yi] = head_tmp1;
            ai += incai;

          }
        } else if (beta_i == 1.0) {
          /* alpha is other, beta is 1.0 */

          ai = 0;
          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            head_sumA = tail_sumA = 0.0;
            aij = ai;
            head_sumB = tail_sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;
              b_elem = b_i[bij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_sumA, tail_sumA, alpha_i);
            head_tmp2 = head_sumB;
            tail_tmp2 = tail_sumB;
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
            y_i[yi] = head_tmp1;
            ai += incai;
            bi += incbi;
          }
        } else {
          /* most general form, alpha, beta are other */

          ai = 0;
          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            head_sumA = tail_sumA = 0.0;
            aij = ai;
            head_sumB = tail_sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;
              b_elem = b_i[bij];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_sumA, tail_sumA, alpha_i);
            compute_doubledouble_eq_doubledouble_mul_double(&head_tmp2, &tail_tmp2, head_sumB, tail_sumB, beta_i);
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
            y_i[yi] = head_tmp1;
            ai += incai;
            bi += incbi;
          }
        }
      }
      FPU_FIX_STOP;
    }
    break;

  default:
    {
      BLAS_error(routine_name, -14, prec, 0);
    }
    break;
  }

} /* end XBLAS::ge_sum_mv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_GE_SUM_MV_HPP
