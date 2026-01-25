#include "blas_extended.h"
#include "blas_extended_private.h"
#include "dot/XBLAS_dot.hpp"

void BLAS_cdot_s_c_x(enum blas_conj_type conj, int n, const void *alpha,
                     const float *x, int incx, const void *beta,
                     const void *y, int incy,
                     void *r, enum blas_prec_type prec)

/*
 * Purpose
 * =======
 *
 * This routine computes the inner product:
 *
 *     r <- beta * r + alpha * SUM_{i=0, n-1} x[i] * y[i].
 *
 * Arguments
 * =========
 *
 * conj   (input) enum blas_conj_type
 *        When x and y are complex vectors, specifies whether vector
 *        components x[i] are used unconjugated or conjugated.
 *
 * n      (input) int
 *        The length of vectors x and y.
 *
 * alpha  (input) const void*
 *
 * x      (input) const float*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 * beta   (input) const void*
 *
 * y      (input) const void*
 *        Array of length n.
 *
 * incy   (input) int
 *        The stride used to access components y[i].
 *
 * r      (input/output) void*
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
  static const char routine_name[] = "BLAS_cdot_s_c_x";

  switch (prec) {
  case blas_prec_single:{

      int i, ix = 0, iy = 0;
      float *r_i = (float *) r;
      const float *x_i = x;
      const float *y_i = (float *) y;
      float *alpha_i = (float *) alpha;
      float *beta_i = (float *) beta;
      float x_ii;
      float y_ii[2];
      float r_v[2];
      float prod[2];
      float sum[2];
      float tmp1[2];
      float tmp2[2];


      /* Test the input parameters. */
      if (n < 0)
        BLAS_error(routine_name, -2, n, NULL);
      else if (incx == 0)
        BLAS_error(routine_name, -5, incx, NULL);
      else if (incy == 0)
        BLAS_error(routine_name, -8, incy, NULL);

      /* Immediate return. */
      if (((beta_i[0] == 1.0 && beta_i[1] == 0.0))
          && (n == 0 || (alpha_i[0] == 0.0 && alpha_i[1] == 0.0)))
        return;



      r_v[0] = r_i[0];
      r_v[1] = r_i[0 + 1];
      sum[0] = sum[1] = 0.0;

      incy *= 2;
      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii[0] = y_i[iy];
        y_ii[1] = y_i[iy + 1];

        {
          prod[0] = y_ii[0] * x_ii;
          prod[1] = y_ii[1] * x_ii;
        }                        /* prod = x[i]*y[i] */
        sum[0] = sum[0] + prod[0];
        sum[1] = sum[1] + prod[1];        /* sum = sum+prod */
        ix += incx;
        iy += incy;
      }                                /* endfor */


      {
        tmp1[0] = sum[0] * alpha_i[0] - sum[1] * alpha_i[1];
        tmp1[1] = sum[0] * alpha_i[1] + sum[1] * alpha_i[0];
      }
      /* tmp1 = sum*alpha */
      {
        tmp2[0] = r_v[0] * beta_i[0] - r_v[1] * beta_i[1];
        tmp2[1] = r_v[0] * beta_i[1] + r_v[1] * beta_i[0];
      }
      /* tmp2 = r*beta */
      tmp1[0] = tmp1[0] + tmp2[0];
      tmp1[1] = tmp1[1] + tmp2[1];        /* tmp1 = tmp1+tmp2 */
      ((float *) r)[0] = tmp1[0];
      ((float *) r)[1] = tmp1[1];        /* r = tmp1 */



      break;
    }
  case blas_prec_double:
  case blas_prec_indigenous:
    {
      int i, ix = 0, iy = 0;
      float *r_i = (float *) r;
      const float *x_i = x;
      const float *y_i = (float *) y;
      float *alpha_i = (float *) alpha;
      float *beta_i = (float *) beta;
      float x_ii;
      float y_ii[2];
      float r_v[2];
      double prod[2];
      double sum[2];
      double tmp1[2];
      double tmp2[2];


      /* Test the input parameters. */
      if (n < 0)
        BLAS_error(routine_name, -2, n, NULL);
      else if (incx == 0)
        BLAS_error(routine_name, -5, incx, NULL);
      else if (incy == 0)
        BLAS_error(routine_name, -8, incy, NULL);

      /* Immediate return. */
      if (((beta_i[0] == 1.0 && beta_i[1] == 0.0))
          && (n == 0 || (alpha_i[0] == 0.0 && alpha_i[1] == 0.0)))
        return;



      r_v[0] = r_i[0];
      r_v[1] = r_i[0 + 1];
      sum[0] = sum[1] = 0.0;

      incy *= 2;
      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii[0] = y_i[iy];
        y_ii[1] = y_i[iy + 1];

        {
          prod[0] = (double) y_ii[0] * x_ii;
          prod[1] = (double) y_ii[1] * x_ii;
        }                        /* prod = x[i]*y[i] */
        sum[0] = sum[0] + prod[0];
        sum[1] = sum[1] + prod[1];        /* sum = sum+prod */
        ix += incx;
        iy += incy;
      }                                /* endfor */


      {
        tmp1[0] = (double) sum[0] * alpha_i[0] - (double) sum[1] * alpha_i[1];
        tmp1[1] = (double) sum[0] * alpha_i[1] + (double) sum[1] * alpha_i[0];
      }                                /* tmp1 = sum*alpha */
      {
        tmp2[0] = (double) r_v[0] * beta_i[0] - (double) r_v[1] * beta_i[1];
        tmp2[1] = (double) r_v[0] * beta_i[1] + (double) r_v[1] * beta_i[0];
      }                                /* tmp2 = r*beta */
      tmp1[0] = tmp1[0] + tmp2[0];
      tmp1[1] = tmp1[1] + tmp2[1];        /* tmp1 = tmp1+tmp2 */
      ((float *) r)[0] = tmp1[0];
      ((float *) r)[1] = tmp1[1];        /* r = tmp1 */


    }
    break;
  case blas_prec_extra:
    {
      int i, ix = 0, iy = 0;
      float *r_i = (float *) r;
      const float *x_i = x;
      const float *y_i = (float *) y;
      float *alpha_i = (float *) alpha;
      float *beta_i = (float *) beta;
      float x_ii;
      float y_ii[2];
      float r_v[2];
      double head_prod[2], tail_prod[2];
      double head_sum[2], tail_sum[2];
      double head_tmp1[2], tail_tmp1[2];
      double head_tmp2[2], tail_tmp2[2];
      FPU_FIX_DECL;

      /* Test the input parameters. */
      if (n < 0)
        BLAS_error(routine_name, -2, n, NULL);
      else if (incx == 0)
        BLAS_error(routine_name, -5, incx, NULL);
      else if (incy == 0)
        BLAS_error(routine_name, -8, incy, NULL);

      /* Immediate return. */
      if (((beta_i[0] == 1.0 && beta_i[1] == 0.0))
          && (n == 0 || (alpha_i[0] == 0.0 && alpha_i[1] == 0.0)))
        return;

      FPU_FIX_START;

      r_v[0] = r_i[0];
      r_v[1] = r_i[0 + 1];
      head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

      incy *= 2;
      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii[0] = y_i[iy];
        y_ii[1] = y_i[iy + 1];

        {
          head_prod[0] = (double) y_ii[0] * x_ii;
          tail_prod[0] = 0.0;
          head_prod[1] = (double) y_ii[1] * x_ii;
          tail_prod[1] = 0.0;
        }                        /* prod = x[i]*y[i] */
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
        }                        /* sum = sum+prod */
        ix += incx;
        iy += incy;
      }                                /* endfor */


      {
        double cd[2];
        cd[0] = (double) alpha_i[0];
        cd[1] = (double) alpha_i[1];
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
          compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
          compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
          head_t2 = -head_t2;
          tail_t2 = -tail_t2;
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
          head_tmp1[0] = head_t1;
          tail_tmp1[0] = tail_t1;
          /* imaginary part */
          compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
          compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
          head_tmp1[1] = head_t1;
          tail_tmp1[1] = tail_t1;
        }

      }                                /* tmp1 = sum*alpha */
      {
        double head_e1, tail_e1;
        double d1;
        double d2;
        /* Real part */
        d1 = (double) r_v[0] * beta_i[0];
        d2 = (double) -r_v[1] * beta_i[1];
        compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
        head_tmp2[0] = head_e1;
        tail_tmp2[0] = tail_e1;
        /* imaginary part */
        d1 = (double) r_v[0] * beta_i[1];
        d2 = (double) r_v[1] * beta_i[0];
        compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
        head_tmp2[1] = head_e1;
        tail_tmp2[1] = tail_e1;
      }                                /* tmp2 = r*beta */
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
      }                                /* tmp1 = tmp1+tmp2 */
      ((float *) r)[0] = head_tmp1[0];
      ((float *) r)[1] = head_tmp1[1];        /* r = tmp1 */

      FPU_FIX_STOP;
    }
    break;
  }
}
