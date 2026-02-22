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
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void symv(blas_order_type order,
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
    BLAS_error(routine_name, -3, n, NULL);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -8, incx, NULL);
  }
  if (incy == 0) {
    BLAS_error(routine_name, -11, incy, NULL);
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
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void symv_x(blas_order_type order,
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
 * where A is a Symmetric matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Storage format of input symmetric matrix A.
 *
 * uplo   (input) enum blas_uplo_type
 *        Determines which half of matrix A (upper or lower triangle)
 *          is accessed.
 *
 * n      (input) int
 *        Dimension of A and size of vectors x, y.
 *
 * alpha  (input) double
 *
 * a      (input) const double*
 *        Matrix A.
 *
 * lda    (input) int
 *        Leading dimension of matrix A.
 *
 * x      (input) const float*
 *        Vector x.
 *
 * incx   (input) int
 *        Stride for vector x.
 *
 * beta   (input) double
 *
 * y      (input/output) double*
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
 */
{
  /* Routine name */
  static const char routine_name[] = "BLAS_dsymv_d_s_x";
  switch (prec) {

  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      /* Integer Index Variables */
      int i, k;

      int xi, yi;
      int aik, astarti, x_starti, y_starti;

      int incai;
      int incaik, incaik2;

      int n_i;

      /* Input Matrices */
      const double *a_i = a;
      const float *x_i = x;

      /* Output Vector */
      double *y_i = y;

      /* Input Scalars */
      double alpha_i = alpha;
      double beta_i = beta;

      /* Temporary Floating-Point Variables */
      double a_elem;
      float x_elem;
      double y_elem;
      double prod;
      double sum;
      double tmp1;
      double tmp2;



      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i == 0.0 && beta_i == 1.0) {
        return;
      }

      /* Check for error conditions. */
      if (lda < n) {
        BLAS_error(routine_name, -3, n, NULL);
      }
      if (incx == 0) {
        BLAS_error(routine_name, -8, incx, NULL);
      }
      if (incy == 0) {
        BLAS_error(routine_name, -11, incy, NULL);
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



      /* alpha = 0.  In this case, just return beta * y */
      if (alpha_i == 0.0) {
        for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
          y_elem = y_i[yi];
          tmp1 = y_elem * beta_i;
          y_i[yi] = tmp1;
        }
      } else if (alpha_i == 1.0) {

        /* Case alpha == 1. */

        if (beta_i == 0.0) {
          /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incai) {
            sum = 0.0;

            for (k = 0, aik = astarti, xi = x_starti; k < i;
                 k++, aik += incaik, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              prod = a_elem * x_elem;
              sum = sum + prod;
            }
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              prod = a_elem * x_elem;
              sum = sum + prod;
            }
            y_i[yi] = sum;
          }
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  y  <--- A * x + beta * y */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incai) {
            sum = 0.0;

            for (k = 0, aik = astarti, xi = x_starti;
                 k < i; k++, aik += incaik, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              prod = a_elem * x_elem;
              sum = sum + prod;
            }
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              prod = a_elem * x_elem;
              sum = sum + prod;
            }
            y_elem = y_i[yi];
            tmp2 = y_elem * beta_i;
            tmp1 = sum;
            tmp1 = tmp2 + tmp1;
            y_i[yi] = tmp1;
          }
        }

      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        for (i = 0, yi = y_starti, astarti = 0;
             i < n_i; i++, yi += incy, astarti += incai) {
          sum = 0.0;

          for (k = 0, aik = astarti, xi = x_starti;
               k < i; k++, aik += incaik, xi += incx) {
            a_elem = a_i[aik];
            x_elem = x_i[xi];
            prod = a_elem * x_elem;
            sum = sum + prod;
          }
          for (; k < n_i; k++, aik += incaik2, xi += incx) {
            a_elem = a_i[aik];
            x_elem = x_i[xi];
            prod = a_elem * x_elem;
            sum = sum + prod;
          }
          y_elem = y_i[yi];
          tmp2 = y_elem * beta_i;
          tmp1 = sum * alpha_i;
          tmp1 = tmp2 + tmp1;
          y_i[yi] = tmp1;
        }
      }



      break;
    }

  case blas_prec_extra:{

      /* Integer Index Variables */
      int i, k;

      int xi, yi;
      int aik, astarti, x_starti, y_starti;

      int incai;
      int incaik, incaik2;

      int n_i;

      /* Input Matrices */
      const double *a_i = a;
      const float *x_i = x;

      /* Output Vector */
      double *y_i = y;

      /* Input Scalars */
      double alpha_i = alpha;
      double beta_i = beta;

      /* Temporary Floating-Point Variables */
      double a_elem;
      float x_elem;
      double y_elem;
      double head_prod, tail_prod;
      double head_sum, tail_sum;
      double head_tmp1, tail_tmp1;
      double head_tmp2, tail_tmp2;

      FPU_FIX_DECL;

      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i == 0.0 && beta_i == 1.0) {
        return;
      }

      /* Check for error conditions. */
      if (lda < n) {
        BLAS_error(routine_name, -3, n, NULL);
      }
      if (incx == 0) {
        BLAS_error(routine_name, -8, incx, NULL);
      }
      if (incy == 0) {
        BLAS_error(routine_name, -11, incy, NULL);
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

      FPU_FIX_START;

      /* alpha = 0.  In this case, just return beta * y */
      if (alpha_i == 0.0) {
        for (i = 0, yi = y_starti; i < n_i; i++, yi += incy) {
          y_elem = y_i[yi];
          compute_doubledouble_eq_double_mul_double(&head_tmp1, &tail_tmp1, y_elem, beta_i);
          y_i[yi] = head_tmp1;
        }
      } else if (alpha_i == 1.0) {

        /* Case alpha == 1. */

        if (beta_i == 0.0) {
          /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incai) {
            head_sum = tail_sum = 0.0;

            for (k = 0, aik = astarti, xi = x_starti; k < i;
                 k++, aik += incaik, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
            }
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
            }
            y_i[yi] = head_sum;
          }
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  y  <--- A * x + beta * y */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incai) {
            head_sum = tail_sum = 0.0;

            for (k = 0, aik = astarti, xi = x_starti;
                 k < i; k++, aik += incaik, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
            }
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              {
                double dt = (double) x_elem;
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
            }
            y_elem = y_i[yi];
            compute_doubledouble_eq_double_mul_double(&head_tmp2, &tail_tmp2, y_elem, beta_i);
            head_tmp1 = head_sum;
            tail_tmp1 = tail_sum;
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp2, tail_tmp2, head_tmp1, tail_tmp1);
            y_i[yi] = head_tmp1;
          }
        }

      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        for (i = 0, yi = y_starti, astarti = 0;
             i < n_i; i++, yi += incy, astarti += incai) {
          head_sum = tail_sum = 0.0;

          for (k = 0, aik = astarti, xi = x_starti;
               k < i; k++, aik += incaik, xi += incx) {
            a_elem = a_i[aik];
            x_elem = x_i[xi];
            {
              double dt = (double) x_elem;
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
            }
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
          }
          for (; k < n_i; k++, aik += incaik2, xi += incx) {
            a_elem = a_i[aik];
            x_elem = x_i[xi];
            {
              double dt = (double) x_elem;
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, dt);
            }
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
          }
          y_elem = y_i[yi];
          compute_doubledouble_eq_double_mul_double(&head_tmp2, &tail_tmp2, y_elem, beta_i);
          compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_sum, tail_sum, alpha_i);
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp2, tail_tmp2, head_tmp1, tail_tmp1);
          y_i[yi] = head_tmp1;
        }
      }

      FPU_FIX_STOP;

      break;
    }
  }
} /* end XBLAS::symv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_SYMV_HPP
