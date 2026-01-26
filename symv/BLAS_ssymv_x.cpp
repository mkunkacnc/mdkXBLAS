#include "blas_extended.h"
#include "blas_extended_private.h"
#include "symv/XBLAS_symv.hpp"

void BLAS_ssymv_x(enum blas_order_type order,
                  enum blas_uplo_type uplo,
                  int n,
                  float alpha,
                  const float *a,
                  int lda,
                  const float *x,
                  int incx,
                  float beta,
                  float *y,
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
 * where A is a Symmetric matrix.
 *
 * Arguments
 * =========
 *
 * order   (input) enum blas_order_type
 *         Storage format of input symmetric matrix A.
 *
 * uplo    (input) enum blas_uplo_type
 *         Determines which half of matrix A (upper or lower triangle)
 *         is accessed.
 *
 * n       (input) int
 *         Dimension of A and size of vectors x, y.
 *
 * alpha   (input) float
 *
 * a       (input) float*
 *         Matrix A.
 *
 * lda     (input) int
 *         Leading dimension of matrix A.
 *
 * x       (input) float*
 *         Vector x.
 *
 * incx    (input) int
 *         Stride for vector x.
 *
 * beta    (input) float
 *
 * y       (input/output) float*
 *         Vector y.
 *
 * incy    (input) int
 *         Stride for vector y.
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
  static const char routine_name[] = "BLAS_ssymv_x";
  switch (prec) {

  case blas_prec_single:{

      /* Integer Index Variables */
      int i, k;

      int xi, yi;
      int aik, astarti, x_starti, y_starti;

      int incai;
      int incaik, incaik2;

      int n_i;

      /* Input Matrices */
      const float *a_i = a;
      const float *x_i = x;

      /* Output Vector */
      float *y_i = y;

      /* Input Scalars */
      float alpha_i = alpha;
      float beta_i = beta;

      /* Temporary Floating-Point Variables */
      float a_elem;
      float x_elem;
      float y_elem;
      float prod;
      float sum;
      float tmp1;
      float tmp2;



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
      const float *a_i = a;
      const float *x_i = x;

      /* Output Vector */
      float *y_i = y;

      /* Input Scalars */
      float alpha_i = alpha;
      float beta_i = beta;

      /* Temporary Floating-Point Variables */
      float a_elem;
      float x_elem;
      float y_elem;
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
          tmp1 = (double) y_elem *beta_i;
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
              prod = (double) a_elem *x_elem;
              sum = sum + prod;
            }
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              prod = (double) a_elem *x_elem;
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
              prod = (double) a_elem *x_elem;
              sum = sum + prod;
            }
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              prod = (double) a_elem *x_elem;
              sum = sum + prod;
            }
            y_elem = y_i[yi];
            tmp2 = (double) y_elem *beta_i;
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
            prod = (double) a_elem *x_elem;
            sum = sum + prod;
          }
          for (; k < n_i; k++, aik += incaik2, xi += incx) {
            a_elem = a_i[aik];
            x_elem = x_i[xi];
            prod = (double) a_elem *x_elem;
            sum = sum + prod;
          }
          y_elem = y_i[yi];
          tmp2 = (double) y_elem *beta_i;
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
      const float *a_i = a;
      const float *x_i = x;

      /* Output Vector */
      float *y_i = y;

      /* Input Scalars */
      float alpha_i = alpha;
      float beta_i = beta;

      /* Temporary Floating-Point Variables */
      float a_elem;
      float x_elem;
      float y_elem;
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
          head_tmp1 = (double) y_elem *beta_i;
          tail_tmp1 = 0.0;
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
              head_prod = (double) a_elem *x_elem;
              tail_prod = 0.0;
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
            }
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              head_prod = (double) a_elem *x_elem;
              tail_prod = 0.0;
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
              head_prod = (double) a_elem *x_elem;
              tail_prod = 0.0;
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
            }
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem = a_i[aik];
              x_elem = x_i[xi];
              head_prod = (double) a_elem *x_elem;
              tail_prod = 0.0;
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
            }
            y_elem = y_i[yi];
            head_tmp2 = (double) y_elem *beta_i;
            tail_tmp2 = 0.0;
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
            head_prod = (double) a_elem *x_elem;
            tail_prod = 0.0;
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
          }
          for (; k < n_i; k++, aik += incaik2, xi += incx) {
            a_elem = a_i[aik];
            x_elem = x_i[xi];
            head_prod = (double) a_elem *x_elem;
            tail_prod = 0.0;
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
          }
          y_elem = y_i[yi];
          head_tmp2 = (double) y_elem *beta_i;
          tail_tmp2 = 0.0;
          {
            double dt = (double) alpha_i;
            compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_sum, tail_sum, dt);
          }
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp2, tail_tmp2, head_tmp1, tail_tmp1);
          y_i[yi] = head_tmp1;
        }
      }

      FPU_FIX_STOP;

      break;
    }
  }
}                                /* end BLAS_ssymv_x */
