#include <blas_extended.h>
#include <blas_extended_private.h>
#include <blas_fpu.h>
void BLAS_csymv2_s_s_x(enum blas_order_type order,
                       enum blas_uplo_type uplo,
                       int n,
                       const void *alpha,
                       const float *a,
                       int lda,
                       const float *x_head,
                       const float *x_tail,
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
 * alpha   (input) const void*
 *
 * a       (input) const float*
 *         Matrix A.
 *
 * lda     (input) int
 *         Leading dimension of matrix A.
 *
 * x_head  (input) const float*
 *         Vector x_head
 *
 * x_tail  (input) const float*
 *         Vector x_tail
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
 * prec    (input) enum blas_prec_type
 *         Specifies the internal precision to be used.
 *         = blas_prec_single: single precision.
 *         = blas_prec_double: double precision.
 *         = blas_prec_extra : anything at least 1.5 times as accurate
 *                             than double, and wider than 80-bits.
 *                             We use double-double in our implementation.
 *
 */
{
//static const char routine_name[] = "BLAS_csymv2_s_s_x";
#if 0
  XBLAS::symv2_x(order,
                 uplo,
                 n,
                 *static_cast<const std::complex<float> *>(alpha),
                 a,
                 lda,
                 x_head,
                 x_tail,
                 incx,
                 *static_cast<const std::complex<float> *>(beta),
                 static_cast<std::complex<float> *>(y),
                 incy,
                 prec);
} /* end BLAS_csymv2_s_s_x */
#else
  /* Routine name */
  const char routine_name[] = "BLAS_csymv2_s_s_x";
  switch (prec) {

  case blas_prec_single:{

      int i, j;
      int xi, yi, xi0, yi0;
      int aij, ai;
      int incai;
      int incaij, incaij2;

      const float *a_i = a;
      const float *x_head_i = x_head;
      const float *x_tail_i = x_tail;
      float *y_i = (float *) y;
      float *alpha_i = (float *) alpha;
      float *beta_i = (float *) beta;
      float a_elem;
      float x_elem;
      float y_elem[2];
      float prod1;
      float prod2;
      float sum1;
      float sum2;
      float tmp1[2];
      float tmp2[2];
      float tmp3[2];



      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
          && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
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


      incy *= 2;



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
        {
          tmp1[0] = alpha_i[0] * sum1;
          tmp1[1] = alpha_i[1] * sum1;
        }
        y_elem[0] = y_i[yi];
        y_elem[1] = y_i[yi + 1];
        {
          tmp2[0] = y_elem[0] * beta_i[0] - y_elem[1] * beta_i[1];
          tmp2[1] = y_elem[0] * beta_i[1] + y_elem[1] * beta_i[0];
        }

        tmp3[0] = tmp1[0] + tmp2[0];
        tmp3[1] = tmp1[1] + tmp2[1];
        y_i[yi] = tmp3[0];
        y_i[yi + 1] = tmp3[1];
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
      const float *x_head_i = x_head;
      const float *x_tail_i = x_tail;
      float *y_i = (float *) y;
      float *alpha_i = (float *) alpha;
      float *beta_i = (float *) beta;
      float a_elem;
      float x_elem;
      float y_elem[2];
      double prod1;
      double prod2;
      double sum1;
      double sum2;
      double tmp1[2];
      double tmp2[2];
      double tmp3[2];



      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
          && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
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


      incy *= 2;



      xi0 = (incx > 0) ? 0 : ((-n + 1) * incx);
      yi0 = (incy > 0) ? 0 : ((-n + 1) * incy);



      /* The most general form,   y <--- alpha * A * (x_head + x_tail) + beta * y   */
      for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
        sum1 = 0.0;
        sum2 = 0.0;

        for (j = 0, aij = ai, xi = xi0; j < i; j++, aij += incaij, xi += incx) {
          a_elem = a_i[aij];
          x_elem = x_head_i[xi];
          prod1 = (double) a_elem *x_elem;
          sum1 = sum1 + prod1;
          x_elem = x_tail_i[xi];
          prod2 = (double) a_elem *x_elem;
          sum2 = sum2 + prod2;
        }
        for (; j < n; j++, aij += incaij2, xi += incx) {
          a_elem = a_i[aij];
          x_elem = x_head_i[xi];
          prod1 = (double) a_elem *x_elem;
          sum1 = sum1 + prod1;
          x_elem = x_tail_i[xi];
          prod2 = (double) a_elem *x_elem;
          sum2 = sum2 + prod2;
        }
        sum1 = sum1 + sum2;
        {
          tmp1[0] = alpha_i[0] * sum1;
          tmp1[1] = alpha_i[1] * sum1;
        }
        y_elem[0] = y_i[yi];
        y_elem[1] = y_i[yi + 1];
        {
          tmp2[0] =
            (double) y_elem[0] * beta_i[0] - (double) y_elem[1] * beta_i[1];
          tmp2[1] =
            (double) y_elem[0] * beta_i[1] + (double) y_elem[1] * beta_i[0];
        }
        tmp3[0] = tmp1[0] + tmp2[0];
        tmp3[1] = tmp1[1] + tmp2[1];
        y_i[yi] = tmp3[0];
        y_i[yi + 1] = tmp3[1];
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
      const float *x_head_i = x_head;
      const float *x_tail_i = x_tail;
      float *y_i = (float *) y;
      float *alpha_i = (float *) alpha;
      float *beta_i = (float *) beta;
      float a_elem;
      float x_elem;
      float y_elem[2];
      double head_prod1, tail_prod1;
      double head_prod2, tail_prod2;
      double head_sum1, tail_sum1;
      double head_sum2, tail_sum2;
      double head_tmp1[2], tail_tmp1[2];
      double head_tmp2[2], tail_tmp2[2];
      double head_tmp3[2], tail_tmp3[2];

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


      incy *= 2;



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
          head_prod1 = (double) a_elem *x_elem;
          tail_prod1 = 0.0;
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum1, &tail_sum1, head_sum1, tail_sum1, head_prod1, tail_prod1);
          x_elem = x_tail_i[xi];
          head_prod2 = (double) a_elem *x_elem;
          tail_prod2 = 0.0;
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod2, tail_prod2);
        }
        for (; j < n; j++, aij += incaij2, xi += incx) {
          a_elem = a_i[aij];
          x_elem = x_head_i[xi];
          head_prod1 = (double) a_elem *x_elem;
          tail_prod1 = 0.0;
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum1, &tail_sum1, head_sum1, tail_sum1, head_prod1, tail_prod1);
          x_elem = x_tail_i[xi];
          head_prod2 = (double) a_elem *x_elem;
          tail_prod2 = 0.0;
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod2, tail_prod2);
        }
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum1, &tail_sum1, head_sum1, tail_sum1, head_sum2, tail_sum2);
        {
          double head_e1, tail_e1;
          double dt;
          dt = (double) alpha_i[0];
          compute_doubledouble_eq_doubledouble_mul_double(&head_e1, &tail_e1, head_sum1, tail_sum1, dt);
          head_tmp1[0] = head_e1;
          tail_tmp1[0] = tail_e1;
          dt = (double) alpha_i[1];
          compute_doubledouble_eq_doubledouble_mul_double(&head_e1, &tail_e1, head_sum1, tail_sum1, dt);
          head_tmp1[1] = head_e1;
          tail_tmp1[1] = tail_e1;
        }
        y_elem[0] = y_i[yi];
        y_elem[1] = y_i[yi + 1];
        {
          double head_e1, tail_e1;
          double d1;
          double d2;
          /* Real part */
          d1 = (double) y_elem[0] * beta_i[0];
          d2 = (double) -y_elem[1] * beta_i[1];
          compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
          head_tmp2[0] = head_e1;
          tail_tmp2[0] = tail_e1;
          /* imaginary part */
          d1 = (double) y_elem[0] * beta_i[1];
          d2 = (double) y_elem[1] * beta_i[0];
          compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
          head_tmp2[1] = head_e1;
          tail_tmp2[1] = tail_e1;
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
          head_tmp3[0] = head_t;
          tail_tmp3[0] = tail_t;
          /* Imaginary part */
          head_a = head_tmp1[1];
          tail_a = tail_tmp1[1];
          head_b = head_tmp2[1];
          tail_b = tail_tmp2[1];
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
          head_tmp3[1] = head_t;
          tail_tmp3[1] = tail_t;
        }
        y_i[yi] = head_tmp3[0];
        y_i[yi + 1] = head_tmp3[1];
      }

      FPU_FIX_STOP;

      break;
    }
  }
}                                /* end BLAS_csymv2_s_s_x */
#endif
