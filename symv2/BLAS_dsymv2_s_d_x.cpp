#include <blas_extended.h>
#include <blas_extended_private.h>
#include <blas_fpu.h>
void BLAS_dsymv2_s_d_x(enum blas_order_type order,
                       enum blas_uplo_type uplo,
                       int n,
                       double alpha,
                       const float *a,
                       int lda,
                       const double *x_head,
                       const double *x_tail,
                       int incx,
                       double beta,
                       double *y,
                       int incy,
                       enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routines computes the matrix product:
 *
 *     y  <-  alpha * A * (x_head + x_tail) + beta * y
 *
 * where A is a symmetric matrix.
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
 * alpha   (input) double
 *
 * a       (input) const float*
 *         Matrix A.
 *
 * lda     (input) int
 *         Leading dimension of matrix A.
 *
 * x_head  (input) const double*
 *         Vector x_head
 *
 * x_tail  (input) const double*
 *         Vector x_tail
 *
 * incx    (input) int
 *         Stride for vector x.
 *
 * beta    (input) double
 *
 * y       (input/output) double*
 *         Vector y.
 *
 * incy    (input) int
 *         Stride for vector y.
 *
 * prec    (input) enum blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 */
{
//static const char routine_name[] = "BLAS_dsymv2_s_d_x";
#if 0
  XBLAS::symv2_x(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy, prec);
} /* end BLAS_dsymv2_s_d_x */
#else
  /* Routine name */
  const char routine_name[] = "BLAS_dsymv2_s_d_x";
  switch (prec) {

  case blas_prec_single:{

      int i, j;
      int xi, yi, xi0, yi0;
      int aij, ai;
      int incai;
      int incaij, incaij2;

      const float *a_i = a;
      const double *x_head_i = x_head;
      const double *x_tail_i = x_tail;
      double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      float a_elem;
      double x_elem;
      double y_elem;
      double prod1;
      double prod2;
      double sum1;
      double sum2;
      double tmp1;
      double tmp2;
      double tmp3;



      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i == 0.0 && beta_i == 1.0) {
        return;
      }

      /* Check for error conditions. */
      if (n < 0) {
        BLAS_error(routine_name, -3, n, NULL);
      }
      if (lda < n) {
        BLAS_error(routine_name, -6, n, NULL);
      }
      if (incx == 0) {
        BLAS_error(routine_name, -9, incx, NULL);
      }
      if (incy == 0) {
        BLAS_error(routine_name, -12, incy, NULL);
      }

      if ((order == blas_colmajor && uplo == blas_upper) ||
          (order == blas_rowmajor && uplo == blas_lower)) {
        incai = lda;
        incaij = 1;
        incaij2 = lda;
      } else {
        incai = 1;
        incaij = lda;
        incaij2 = 1;
      }






      xi0 = (incx > 0) ? 0 : ((-n + 1) * incx);
      yi0 = (incy > 0) ? 0 : ((-n + 1) * incy);



      /* The most general form,   y <--- alpha * A * (x_head + x_tail) + beta * y   */
      for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
        sum1 = 0.0;
        sum2 = 0.0;

        for (j = 0, aij = ai, xi = xi0; j < i; j++, aij += incaij, xi += incx) {
          a_elem = a_i[aij];
          x_elem = x_head_i[xi];
          prod1 = a_elem * x_elem;
          sum1 = sum1 + prod1;
          x_elem = x_tail_i[xi];
          prod2 = a_elem * x_elem;
          sum2 = sum2 + prod2;
        }
        for (; j < n; j++, aij += incaij2, xi += incx) {
          a_elem = a_i[aij];
          x_elem = x_head_i[xi];
          prod1 = a_elem * x_elem;
          sum1 = sum1 + prod1;
          x_elem = x_tail_i[xi];
          prod2 = a_elem * x_elem;
          sum2 = sum2 + prod2;
        }
        sum1 = sum1 + sum2;
        tmp1 = sum1 * alpha_i;
        y_elem = y_i[yi];
        tmp2 = y_elem * beta_i;
        tmp3 = tmp1 + tmp2;
        y_i[yi] = tmp3;
      }



      break;
    }

  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, j;
      int xi, yi, xi0, yi0;
      int aij, ai;
      int incai;
      int incaij, incaij2;

      const float *a_i = a;
      const double *x_head_i = x_head;
      const double *x_tail_i = x_tail;
      double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      float a_elem;
      double x_elem;
      double y_elem;
      double prod1;
      double prod2;
      double sum1;
      double sum2;
      double tmp1;
      double tmp2;
      double tmp3;



      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i == 0.0 && beta_i == 1.0) {
        return;
      }

      /* Check for error conditions. */
      if (n < 0) {
        BLAS_error(routine_name, -3, n, NULL);
      }
      if (lda < n) {
        BLAS_error(routine_name, -6, n, NULL);
      }
      if (incx == 0) {
        BLAS_error(routine_name, -9, incx, NULL);
      }
      if (incy == 0) {
        BLAS_error(routine_name, -12, incy, NULL);
      }

      if ((order == blas_colmajor && uplo == blas_upper) ||
          (order == blas_rowmajor && uplo == blas_lower)) {
        incai = lda;
        incaij = 1;
        incaij2 = lda;
      } else {
        incai = 1;
        incaij = lda;
        incaij2 = 1;
      }






      xi0 = (incx > 0) ? 0 : ((-n + 1) * incx);
      yi0 = (incy > 0) ? 0 : ((-n + 1) * incy);



      /* The most general form,   y <--- alpha * A * (x_head + x_tail) + beta * y   */
      for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
        sum1 = 0.0;
        sum2 = 0.0;

        for (j = 0, aij = ai, xi = xi0; j < i; j++, aij += incaij, xi += incx) {
          a_elem = a_i[aij];
          x_elem = x_head_i[xi];
          prod1 = a_elem * x_elem;
          sum1 = sum1 + prod1;
          x_elem = x_tail_i[xi];
          prod2 = a_elem * x_elem;
          sum2 = sum2 + prod2;
        }
        for (; j < n; j++, aij += incaij2, xi += incx) {
          a_elem = a_i[aij];
          x_elem = x_head_i[xi];
          prod1 = a_elem * x_elem;
          sum1 = sum1 + prod1;
          x_elem = x_tail_i[xi];
          prod2 = a_elem * x_elem;
          sum2 = sum2 + prod2;
        }
        sum1 = sum1 + sum2;
        tmp1 = sum1 * alpha_i;
        y_elem = y_i[yi];
        tmp2 = y_elem * beta_i;
        tmp3 = tmp1 + tmp2;
        y_i[yi] = tmp3;
      }



      break;
    }

  case blas_prec_extra:{

      int i, j;
      int xi, yi, xi0, yi0;
      int aij, ai;
      int incai;
      int incaij, incaij2;

      const float *a_i = a;
      const double *x_head_i = x_head;
      const double *x_tail_i = x_tail;
      double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      float a_elem;
      double x_elem;
      double y_elem;
      double head_prod1, tail_prod1;
      double head_prod2, tail_prod2;
      double head_sum1, tail_sum1;
      double head_sum2, tail_sum2;
      double head_tmp1, tail_tmp1;
      double head_tmp2, tail_tmp2;
      double head_tmp3, tail_tmp3;

      FPU_FIX_DECL;

      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i == 0.0 && beta_i == 1.0) {
        return;
      }

      /* Check for error conditions. */
      if (n < 0) {
        BLAS_error(routine_name, -3, n, NULL);
      }
      if (lda < n) {
        BLAS_error(routine_name, -6, n, NULL);
      }
      if (incx == 0) {
        BLAS_error(routine_name, -9, incx, NULL);
      }
      if (incy == 0) {
        BLAS_error(routine_name, -12, incy, NULL);
      }

      if ((order == blas_colmajor && uplo == blas_upper) ||
          (order == blas_rowmajor && uplo == blas_lower)) {
        incai = lda;
        incaij = 1;
        incaij2 = lda;
      } else {
        incai = 1;
        incaij = lda;
        incaij2 = 1;
      }






      xi0 = (incx > 0) ? 0 : ((-n + 1) * incx);
      yi0 = (incy > 0) ? 0 : ((-n + 1) * incy);

      FPU_FIX_START;

      /* The most general form,   y <--- alpha * A * (x_head + x_tail) + beta * y   */
      for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
        head_sum1 = tail_sum1 = 0.0;
        head_sum2 = tail_sum2 = 0.0;

        for (j = 0, aij = ai, xi = xi0; j < i; j++, aij += incaij, xi += incx) {
          a_elem = a_i[aij];
          x_elem = x_head_i[xi];
          {
            double dt = (double) a_elem;
            compute_doubledouble_eq_double_mul_double(&head_prod1, &tail_prod1, dt, x_elem);
          }
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum1, &tail_sum1, head_sum1, tail_sum1, head_prod1, tail_prod1);
          x_elem = x_tail_i[xi];
          {
            double dt = (double) a_elem;
            compute_doubledouble_eq_double_mul_double(&head_prod2, &tail_prod2, dt, x_elem);
          }
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod2, tail_prod2);
        }
        for (; j < n; j++, aij += incaij2, xi += incx) {
          a_elem = a_i[aij];
          x_elem = x_head_i[xi];
          {
            double dt = (double) a_elem;
            compute_doubledouble_eq_double_mul_double(&head_prod1, &tail_prod1, dt, x_elem);
          }
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum1, &tail_sum1, head_sum1, tail_sum1, head_prod1, tail_prod1);
          x_elem = x_tail_i[xi];
          {
            double dt = (double) a_elem;
            compute_doubledouble_eq_double_mul_double(&head_prod2, &tail_prod2, dt, x_elem);
          }
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod2, tail_prod2);
        }
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum1, &tail_sum1, head_sum1, tail_sum1, head_sum2, tail_sum2);
        compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_sum1, tail_sum1, alpha_i);
        y_elem = y_i[yi];
        compute_doubledouble_eq_double_mul_double(&head_tmp2, &tail_tmp2, y_elem, beta_i);
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp3, &tail_tmp3, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
        y_i[yi] = head_tmp3;
      }

      FPU_FIX_STOP;

      break;
    }
  }
}                                /* end BLAS_dsymv2_s_d_x */
#endif
