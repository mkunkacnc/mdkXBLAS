#include "blas_extended.h"
#include "blas_extended_private.h"
#include "symv/XBLAS_symv.hpp"

void BLAS_zsymv_z_d_x(enum blas_order_type order,
                      enum blas_uplo_type uplo,
                      int n,
                      const void *alpha,
                      const void *a,
                      int lda,
                      const double *x,
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
 * alpha   (input) const void*
 *
 * a       (input) void*
 *         Matrix A.
 *
 * lda     (input) int
 *         Leading dimension of matrix A.
 *
 * x       (input) double*
 *         Vector x.
 *
 * incx    (input) int
 *         Stride for vector x.
 *
 * beta    (input) const void*
 *
 * y       (input/output) void*
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
//static const char routine_name[] = "BLAS_zsymv_z_d_x";
#if 0
  XBLAS::symv_x(order,
                uplo,
                n,
                *static_cast<const std::complex<> *>(alpha),
                static_cast<const std::complex<> *>(a),
                lda,
                x,
                incx,
                *static_cast<const std::complex<> *>(beta),
                static_cast<std::complex<> *>(y),
                incy,
                prec);
} /* end BLAS_zsymv_z_d_x */
#else
  /* Routine name */
  static const char routine_name[] = "BLAS_zsymv_z_d_x";
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
      const double *a_i = (double *) a;
      const double *x_i = x;

      /* Output Vector */
      double *y_i = (double *) y;

      /* Input Scalars */
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;

      /* Temporary Floating-Point Variables */
      double a_elem[2];
      double x_elem;
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

      incy *= 2;
      incai *= 2;
      incaik *= 2;
      incaik2 *= 2;
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
      } else if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {

        /* Case alpha == 1. */

        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incai) {
            sum[0] = sum[1] = 0.0;

            for (k = 0, aik = astarti, xi = x_starti; k < i;
                 k++, aik += incaik, xi += incx) {
              a_elem[0] = a_i[aik];
              a_elem[1] = a_i[aik + 1];
              x_elem = x_i[xi];
              {
                prod[0] = a_elem[0] * x_elem;
                prod[1] = a_elem[1] * x_elem;
              }
              sum[0] = sum[0] + prod[0];
              sum[1] = sum[1] + prod[1];
            }
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem[0] = a_i[aik];
              a_elem[1] = a_i[aik + 1];
              x_elem = x_i[xi];
              {
                prod[0] = a_elem[0] * x_elem;
                prod[1] = a_elem[1] * x_elem;
              }
              sum[0] = sum[0] + prod[0];
              sum[1] = sum[1] + prod[1];
            }
            y_i[yi] = sum[0];
            y_i[yi + 1] = sum[1];
          }
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  y  <--- A * x + beta * y */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incai) {
            sum[0] = sum[1] = 0.0;

            for (k = 0, aik = astarti, xi = x_starti;
                 k < i; k++, aik += incaik, xi += incx) {
              a_elem[0] = a_i[aik];
              a_elem[1] = a_i[aik + 1];
              x_elem = x_i[xi];
              {
                prod[0] = a_elem[0] * x_elem;
                prod[1] = a_elem[1] * x_elem;
              }
              sum[0] = sum[0] + prod[0];
              sum[1] = sum[1] + prod[1];
            }
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem[0] = a_i[aik];
              a_elem[1] = a_i[aik + 1];
              x_elem = x_i[xi];
              {
                prod[0] = a_elem[0] * x_elem;
                prod[1] = a_elem[1] * x_elem;
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
          }
        }

      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        for (i = 0, yi = y_starti, astarti = 0;
             i < n_i; i++, yi += incy, astarti += incai) {
          sum[0] = sum[1] = 0.0;

          for (k = 0, aik = astarti, xi = x_starti;
               k < i; k++, aik += incaik, xi += incx) {
            a_elem[0] = a_i[aik];
            a_elem[1] = a_i[aik + 1];
            x_elem = x_i[xi];
            {
              prod[0] = a_elem[0] * x_elem;
              prod[1] = a_elem[1] * x_elem;
            }
            sum[0] = sum[0] + prod[0];
            sum[1] = sum[1] + prod[1];
          }
          for (; k < n_i; k++, aik += incaik2, xi += incx) {
            a_elem[0] = a_i[aik];
            a_elem[1] = a_i[aik + 1];
            x_elem = x_i[xi];
            {
              prod[0] = a_elem[0] * x_elem;
              prod[1] = a_elem[1] * x_elem;
            }
            sum[0] = sum[0] + prod[0];
            sum[1] = sum[1] + prod[1];
          }
          y_elem[0] = y_i[yi];
          y_elem[1] = y_i[yi + 1];
          {
            tmp2[0] =
              (double) y_elem[0] * beta_i[0] - (double) y_elem[1] * beta_i[1];
            tmp2[1] =
              (double) y_elem[0] * beta_i[1] + (double) y_elem[1] * beta_i[0];
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
      const double *a_i = (double *) a;
      const double *x_i = x;

      /* Output Vector */
      double *y_i = (double *) y;

      /* Input Scalars */
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;

      /* Temporary Floating-Point Variables */
      double a_elem[2];
      double x_elem;
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

      incy *= 2;
      incai *= 2;
      incaik *= 2;
      incaik2 *= 2;
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
      } else if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {

        /* Case alpha == 1. */

        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          /* Case alpha = 1, beta = 0.  We compute  y <--- A * x */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incai) {
            head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

            for (k = 0, aik = astarti, xi = x_starti; k < i;
                 k++, aik += incaik, xi += incx) {
              a_elem[0] = a_i[aik];
              a_elem[1] = a_i[aik + 1];
              x_elem = x_i[xi];
              {
                /* Compute complex-extra = complex-double * real. */
                double head_t, tail_t;
                compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[0]);
                head_prod[0] = head_t;
                tail_prod[0] = tail_t;
                compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[1]);
                head_prod[1] = head_t;
                tail_prod[1] = tail_t;
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
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem[0] = a_i[aik];
              a_elem[1] = a_i[aik + 1];
              x_elem = x_i[xi];
              {
                /* Compute complex-extra = complex-double * real. */
                double head_t, tail_t;
                compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[0]);
                head_prod[0] = head_t;
                tail_prod[0] = tail_t;
                compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[1]);
                head_prod[1] = head_t;
                tail_prod[1] = tail_t;
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
          }
        } else {
          /* Case alpha = 1, but beta != 0.
             We compute  y  <--- A * x + beta * y */
          for (i = 0, yi = y_starti, astarti = 0;
               i < n_i; i++, yi += incy, astarti += incai) {
            head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

            for (k = 0, aik = astarti, xi = x_starti;
                 k < i; k++, aik += incaik, xi += incx) {
              a_elem[0] = a_i[aik];
              a_elem[1] = a_i[aik + 1];
              x_elem = x_i[xi];
              {
                /* Compute complex-extra = complex-double * real. */
                double head_t, tail_t;
                compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[0]);
                head_prod[0] = head_t;
                tail_prod[0] = tail_t;
                compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[1]);
                head_prod[1] = head_t;
                tail_prod[1] = tail_t;
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
            for (; k < n_i; k++, aik += incaik2, xi += incx) {
              a_elem[0] = a_i[aik];
              a_elem[1] = a_i[aik + 1];
              x_elem = x_i[xi];
              {
                /* Compute complex-extra = complex-double * real. */
                double head_t, tail_t;
                compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[0]);
                head_prod[0] = head_t;
                tail_prod[0] = tail_t;
                compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[1]);
                head_prod[1] = head_t;
                tail_prod[1] = tail_t;
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
          }
        }

      } else {
        /* The most general form,   y <--- alpha * A * x + beta * y */
        for (i = 0, yi = y_starti, astarti = 0;
             i < n_i; i++, yi += incy, astarti += incai) {
          head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

          for (k = 0, aik = astarti, xi = x_starti;
               k < i; k++, aik += incaik, xi += incx) {
            a_elem[0] = a_i[aik];
            a_elem[1] = a_i[aik + 1];
            x_elem = x_i[xi];
            {
              /* Compute complex-extra = complex-double * real. */
              double head_t, tail_t;
              compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[0]);
              head_prod[0] = head_t;
              tail_prod[0] = tail_t;
              compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[1]);
              head_prod[1] = head_t;
              tail_prod[1] = tail_t;
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
          for (; k < n_i; k++, aik += incaik2, xi += incx) {
            a_elem[0] = a_i[aik];
            a_elem[1] = a_i[aik + 1];
            x_elem = x_i[xi];
            {
              /* Compute complex-extra = complex-double * real. */
              double head_t, tail_t;
              compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[0]);
              head_prod[0] = head_t;
              tail_prod[0] = tail_t;
              compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_elem, a_elem[1]);
              head_prod[1] = head_t;
              tail_prod[1] = tail_t;
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
        }
      }

      FPU_FIX_STOP;

      break;
    }
  }
}                                /* end BLAS_zsymv_z_d_x */
#endif
