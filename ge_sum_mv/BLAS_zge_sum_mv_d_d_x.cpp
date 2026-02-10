#include "blas_extended.h"
#include "blas_extended_private.h"
#include "ge_sum_mv/XBLAS_ge_sum_mv.hpp"

void BLAS_zge_sum_mv_d_d_x(enum blas_order_type order,
                           int m,
                           int n,
                           const void *alpha,
                           const double *a,
                           int lda,
                           const double *x,
                           int incx,
                           const void *beta,
                           const double *b,
                           int ldb,
                           void *y,
                           int incy,
                           enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * A * x + beta * B * y,
 *     where A, B are general matricies.
 *
 * Arguments
 * =========
 *
 * order        (input) blas_order_type
 *              Order of A; row or column major
 *
 * m            (input) int
 *              Row Dimension of A, B, length of output vector y
 *
 * n            (input) int
 *              Column Dimension of A, B and the length of vector x
 *
 * alpha        (input) const void*
 *
 * A            (input) const double*
 *
 * lda          (input) int
 *              Leading dimension of A
 *
 * x            (input) const double*
 *
 * incx         (input) int
 *              The stride for vector x.
 *
 * beta         (input) const void*
 *
 * b            (input) const double*
 *
 * ldb          (input) int
 *              Leading dimension of B
 *
 * y            (input/output) void*
 *
 * incy         (input) int
 *              The stride for vector y.
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
//static const char routine_name[] = "BLAS_zge_sum_mv_d_d_x";
#if 0
  XBLAS::ge_sum_mv_x(order,
                     m,
                     n,
                     *static_cast<const std::complex<double> *>(alpha),
                     a,
                     lda,
                     x,
                     incx,
                     *static_cast<const std::complex<double> *>(beta),
                     b,
                     ldb,
                     static_cast<std::complex<double> *>(y),
                     incy,
                     prec);
} /* end BLAS_zge_sum_mv_d_d_x */
#else
  /* Routine name */
  static const char routine_name[] = "BLAS_zge_sum_mv_d_d";
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
      const double *x_i = x;
      double *y_i = (double *) y;
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;
      double a_elem;
      double b_elem;
      double x_elem;
      double prod;
      double sumA;
      double sumB;
      double tmp1[2];
      double tmp2[2];



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

      incyi *= 2;





      if (incxi > 0)
        x_starti = 0;
      else
        x_starti = (1 - n) * incxi;

      if (incyi > 0)
        y_starti = 0;
      else
        y_starti = (1 - m) * incyi;



      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          /* alpha, beta are 0.0 */
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            y_i[yi] = 0.0;
            y_i[yi + 1] = 0.0;
          }
        } else if ((beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
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
            tmp1[0] = sumB;
            tmp1[1] = 0.0;
            y_i[yi] = tmp1[0];
            y_i[yi + 1] = tmp1[1];

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
            {
              tmp1[0] = beta_i[0] * sumB;
              tmp1[1] = beta_i[1] * sumB;
            }
            y_i[yi] = tmp1[0];
            y_i[yi + 1] = tmp1[1];

            bi += incbi;
          }
        }
      } else if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
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
            tmp1[0] = sumA;
            tmp1[1] = 0.0;
            y_i[yi] = tmp1[0];
            y_i[yi + 1] = tmp1[1];
            ai += incai;

          }
        } else if ((beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
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
            tmp1[0] = sumA;
            tmp1[1] = 0.0;
            tmp2[0] = sumB;
            tmp2[1] = 0.0;
            tmp1[0] = tmp1[0] + tmp2[0];
            tmp1[1] = tmp1[1] + tmp2[1];
            y_i[yi] = tmp1[0];
            y_i[yi + 1] = tmp1[1];
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
            tmp1[0] = sumA;
            tmp1[1] = 0.0;
            {
              tmp2[0] = beta_i[0] * sumB;
              tmp2[1] = beta_i[1] * sumB;
            }
            tmp1[0] = tmp1[0] + tmp2[0];
            tmp1[1] = tmp1[1] + tmp2[1];
            y_i[yi] = tmp1[0];
            y_i[yi + 1] = tmp1[1];
            ai += incai;
            bi += incbi;
          }
        }
      } else {
        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
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
            {
              tmp1[0] = alpha_i[0] * sumA;
              tmp1[1] = alpha_i[1] * sumA;
            }
            y_i[yi] = tmp1[0];
            y_i[yi + 1] = tmp1[1];
            ai += incai;

          }
        } else if ((beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
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
            {
              tmp1[0] = alpha_i[0] * sumA;
              tmp1[1] = alpha_i[1] * sumA;
            }
            tmp2[0] = sumB;
            tmp2[1] = 0.0;
            tmp1[0] = tmp1[0] + tmp2[0];
            tmp1[1] = tmp1[1] + tmp2[1];
            y_i[yi] = tmp1[0];
            y_i[yi + 1] = tmp1[1];
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
            {
              tmp1[0] = alpha_i[0] * sumA;
              tmp1[1] = alpha_i[1] * sumA;
            }
            {
              tmp2[0] = beta_i[0] * sumB;
              tmp2[1] = beta_i[1] * sumB;
            }
            tmp1[0] = tmp1[0] + tmp2[0];
            tmp1[1] = tmp1[1] + tmp2[1];
            y_i[yi] = tmp1[0];
            y_i[yi + 1] = tmp1[1];
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
      const double *x_i = x;
      double *y_i = (double *) y;
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;
      double a_elem;
      double b_elem;
      double x_elem;
      double head_prod, tail_prod;
      double head_sumA, tail_sumA;
      double head_sumB, tail_sumB;
      double head_tmp1[2], tail_tmp1[2];
      double head_tmp2[2], tail_tmp2[2];

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

      incyi *= 2;





      if (incxi > 0)
        x_starti = 0;
      else
        x_starti = (1 - n) * incxi;

      if (incyi > 0)
        y_starti = 0;
      else
        y_starti = (1 - m) * incyi;

      FPU_FIX_START;

      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          /* alpha, beta are 0.0 */
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            y_i[yi] = 0.0;
            y_i[yi + 1] = 0.0;
          }
        } else if ((beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
          /* alpha is 0.0, beta is 1.0 */


          bi = 0;
          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {

            head_sumB = tail_sumB = 0.0;
            bij = bi;
            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];

              b_elem = b_i[bij];
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            head_tmp1[0] = head_sumB;
            tail_tmp1[0] = tail_sumB;
            head_tmp1[1] = tail_tmp1[1] = 0.0;
            y_i[yi] = head_tmp1[0];
            y_i[yi + 1] = head_tmp1[1];

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
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            {
              /* Compute complex-extra = complex-double * real. */
              double head_t, tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumB, tail_sumB, beta_i[0]);
              head_tmp1[0] = head_t;
              tail_tmp1[0] = tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumB, tail_sumB, beta_i[1]);
              head_tmp1[1] = head_t;
              tail_tmp1[1] = tail_t;
            }
            y_i[yi] = head_tmp1[0];
            y_i[yi + 1] = head_tmp1[1];

            bi += incbi;
          }
        }
      } else if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          /* alpha is 1.0, beta is 0.0 */

          ai = 0;

          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            head_sumA = tail_sumA = 0.0;
            aij = ai;

            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;

            }
            /* now put the result into y_i */
            head_tmp1[0] = head_sumA;
            tail_tmp1[0] = tail_sumA;
            head_tmp1[1] = tail_tmp1[1] = 0.0;
            y_i[yi] = head_tmp1[0];
            y_i[yi + 1] = head_tmp1[1];
            ai += incai;

          }
        } else if ((beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
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
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;
              b_elem = b_i[bij];
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            head_tmp1[0] = head_sumA;
            tail_tmp1[0] = tail_sumA;
            head_tmp1[1] = tail_tmp1[1] = 0.0;
            head_tmp2[0] = head_sumB;
            tail_tmp2[0] = tail_sumB;
            head_tmp2[1] = tail_tmp2[1] = 0.0;
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_tmp1[0];
              tail_a = tail_tmp1[0];
              head_b = head_tmp2[0];
              tail_b = tail_tmp2[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_tmp1[0] = head_t;
              tail_tmp1[0] = tail_t;
              /* Imaginary part */
              head_a = head_tmp1[1];
              tail_a = tail_tmp1[1];
              head_b = head_tmp2[1];
              tail_b = tail_tmp2[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_tmp1[1] = head_t;
              tail_tmp1[1] = tail_t;
            }
            y_i[yi] = head_tmp1[0];
            y_i[yi + 1] = head_tmp1[1];
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
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;
              b_elem = b_i[bij];
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            head_tmp1[0] = head_sumA;
            tail_tmp1[0] = tail_sumA;
            head_tmp1[1] = tail_tmp1[1] = 0.0;
            {
              /* Compute complex-extra = complex-double * real. */
              double head_t, tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumB, tail_sumB, beta_i[0]);
              head_tmp2[0] = head_t;
              tail_tmp2[0] = tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumB, tail_sumB, beta_i[1]);
              head_tmp2[1] = head_t;
              tail_tmp2[1] = tail_t;
            }
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_tmp1[0];
              tail_a = tail_tmp1[0];
              head_b = head_tmp2[0];
              tail_b = tail_tmp2[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_tmp1[0] = head_t;
              tail_tmp1[0] = tail_t;
              /* Imaginary part */
              head_a = head_tmp1[1];
              tail_a = tail_tmp1[1];
              head_b = head_tmp2[1];
              tail_b = tail_tmp2[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_tmp1[1] = head_t;
              tail_tmp1[1] = tail_t;
            }
            y_i[yi] = head_tmp1[0];
            y_i[yi + 1] = head_tmp1[1];
            ai += incai;
            bi += incbi;
          }
        }
      } else {
        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          /* alpha is other, beta is 0.0 */

          ai = 0;

          for (i = 0, yi = y_starti; i < m; i++, yi += incyi) {
            head_sumA = tail_sumA = 0.0;
            aij = ai;

            for (j = 0, xi = x_starti; j < n; j++, xi += incxi) {
              x_elem = x_i[xi];
              a_elem = a_i[aij];
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;

            }
            /* now put the result into y_i */
            {
              /* Compute complex-extra = complex-double * real. */
              double head_t, tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumA, tail_sumA, alpha_i[0]);
              head_tmp1[0] = head_t;
              tail_tmp1[0] = tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumA, tail_sumA, alpha_i[1]);
              head_tmp1[1] = head_t;
              tail_tmp1[1] = tail_t;
            }
            y_i[yi] = head_tmp1[0];
            y_i[yi + 1] = head_tmp1[1];
            ai += incai;

          }
        } else if ((beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
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
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;
              b_elem = b_i[bij];
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            {
              /* Compute complex-extra = complex-double * real. */
              double head_t, tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumA, tail_sumA, alpha_i[0]);
              head_tmp1[0] = head_t;
              tail_tmp1[0] = tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumA, tail_sumA, alpha_i[1]);
              head_tmp1[1] = head_t;
              tail_tmp1[1] = tail_t;
            }
            head_tmp2[0] = head_sumB;
            tail_tmp2[0] = tail_sumB;
            head_tmp2[1] = tail_tmp2[1] = 0.0;
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_tmp1[0];
              tail_a = tail_tmp1[0];
              head_b = head_tmp2[0];
              tail_b = tail_tmp2[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_tmp1[0] = head_t;
              tail_tmp1[0] = tail_t;
              /* Imaginary part */
              head_a = head_tmp1[1];
              tail_a = tail_tmp1[1];
              head_b = head_tmp2[1];
              tail_b = tail_tmp2[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_tmp1[1] = head_t;
              tail_tmp1[1] = tail_t;
            }
            y_i[yi] = head_tmp1[0];
            y_i[yi + 1] = head_tmp1[1];
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
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumA, &tail_sumA, head_sumA, tail_sumA, head_prod, tail_prod);
              aij += incaij;
              b_elem = b_i[bij];
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, b_elem, x_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sumB, &tail_sumB, head_sumB, tail_sumB, head_prod, tail_prod);
              bij += incbij;
            }
            /* now put the result into y_i */
            {
              /* Compute complex-extra = complex-double * real. */
              double head_t, tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumA, tail_sumA, alpha_i[0]);
              head_tmp1[0] = head_t;
              tail_tmp1[0] = tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumA, tail_sumA, alpha_i[1]);
              head_tmp1[1] = head_t;
              tail_tmp1[1] = tail_t;
            }
            {
              /* Compute complex-extra = complex-double * real. */
              double head_t, tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumB, tail_sumB, beta_i[0]);
              head_tmp2[0] = head_t;
              tail_tmp2[0] = tail_t;
              compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sumB, tail_sumB, beta_i[1]);
              head_tmp2[1] = head_t;
              tail_tmp2[1] = tail_t;
            }
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_tmp1[0];
              tail_a = tail_tmp1[0];
              head_b = head_tmp2[0];
              tail_b = tail_tmp2[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_tmp1[0] = head_t;
              tail_tmp1[0] = tail_t;
              /* Imaginary part */
              head_a = head_tmp1[1];
              tail_a = tail_tmp1[1];
              head_b = head_tmp2[1];
              tail_b = tail_tmp2[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_tmp1[1] = head_t;
              tail_tmp1[1] = tail_t;
            }
            y_i[yi] = head_tmp1[0];
            y_i[yi + 1] = head_tmp1[1];
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

}                                /* end BLAS_zge_sum_mv_d_d */
#endif
