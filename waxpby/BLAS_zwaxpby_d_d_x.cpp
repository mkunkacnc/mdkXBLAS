#include "blas_extended.h"
#include "blas_extended_private.h"
#include "waxpby/XBLAS_waxpby.hpp"


void BLAS_zwaxpby_d_d_x(int n, const void *alpha, const double *x, int incx,
                        const void *beta, const double *y, int incy, void *w,
                        int incw, enum blas_prec_type prec)

/*
 * Purpose
 * =======
 *
 * This routine computes:
 *
 *     w <- alpha * x + beta * y
 *
 * Arguments
 * =========
 *
 * n     (input) int
 *       The length of vectors x, y, and w.
 *
 * alpha (input) const void*
 *
 * x     (input) const double*
 *       Array of length n.
 *
 * incx  (input) int
 *       The stride used to access components x[i].
 *
 * beta  (input) const void*
 *
 * y     (input) double*
 *       Array of length n.
 *
 * incy  (input) int
 *       The stride used to access components y[i].
 *
 * w     (output) void*
 *       Array of length n.
 *
 * incw  (input) int
 *       The stride used to write components w[i].
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
  static const char routine_name[] = "BLAS_zwaxpby_d_d_x";
  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, ix = 0, iy = 0, iw = 0;
      double *w_i = (double *) w;
      const double *x_i = x;
      const double *y_i = y;
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;
      double x_ii;
      double y_ii;
      double tmpx[2];
      double tmpy[2];



      /* Test the input parameters. */
      if (incx == 0)
        BLAS_error(routine_name, -4, incx, NULL);
      else if (incy == 0)
        BLAS_error(routine_name, -7, incy, NULL);
      else if (incw == 0)
        BLAS_error(routine_name, -9, incw, NULL);


      /* Immediate return */
      if (n <= 0) {
        return;
      }





      incw *= 2;
      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;
      if (incw < 0)
        iw = (-n + 1) * incw;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii = y_i[iy];
        {
          tmpx[0] = alpha_i[0] * x_ii;
          tmpx[1] = alpha_i[1] * x_ii;
        }                        /* tmpx  = alpha * x[ix] */
        {
          tmpy[0] = beta_i[0] * y_ii;
          tmpy[1] = beta_i[1] * y_ii;
        }                        /* tmpy = beta * y[iy] */
        tmpy[0] = tmpy[0] + tmpx[0];
        tmpy[1] = tmpy[1] + tmpx[1];
        w_i[iw] = tmpy[0];
        w_i[iw + 1] = tmpy[1];
        ix += incx;
        iy += incy;
        iw += incw;
      }                                /* endfor */



      break;
    }

  case blas_prec_extra:{

      int i, ix = 0, iy = 0, iw = 0;
      double *w_i = (double *) w;
      const double *x_i = x;
      const double *y_i = y;
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;
      double x_ii;
      double y_ii;
      double head_tmpx[2], tail_tmpx[2];
      double head_tmpy[2], tail_tmpy[2];

      FPU_FIX_DECL;

      /* Test the input parameters. */
      if (incx == 0)
        BLAS_error(routine_name, -4, incx, NULL);
      else if (incy == 0)
        BLAS_error(routine_name, -7, incy, NULL);
      else if (incw == 0)
        BLAS_error(routine_name, -9, incw, NULL);


      /* Immediate return */
      if (n <= 0) {
        return;
      }

      FPU_FIX_START;



      incw *= 2;
      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;
      if (incw < 0)
        iw = (-n + 1) * incw;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii = y_i[iy];
        {
          /* Compute complex-extra = complex-double * real. */
          double head_t, tail_t;
          compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_ii, alpha_i[0]);
          head_tmpx[0] = head_t;
          tail_tmpx[0] = tail_t;
          compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, x_ii, alpha_i[1]);
          head_tmpx[1] = head_t;
          tail_tmpx[1] = tail_t;
        }                        /* tmpx  = alpha * x[ix] */
        {
          /* Compute complex-extra = complex-double * real. */
          double head_t, tail_t;
          compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, y_ii, beta_i[0]);
          head_tmpy[0] = head_t;
          tail_tmpy[0] = tail_t;
          compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, y_ii, beta_i[1]);
          head_tmpy[1] = head_t;
          tail_tmpy[1] = tail_t;
        }                        /* tmpy = beta * y[iy] */
        {
          double head_t, tail_t;
          double head_a, tail_a;
          double head_b, tail_b;
          /* Real part */
          head_a = head_tmpy[0];
          tail_a = tail_tmpy[0];
          head_b = head_tmpx[0];
          tail_b = tail_tmpx[0];
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
          head_tmpy[0] = head_t;
          tail_tmpy[0] = tail_t;
          /* Imaginary part */
          head_a = head_tmpy[1];
          tail_a = tail_tmpy[1];
          head_b = head_tmpx[1];
          tail_b = tail_tmpx[1];
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
          head_tmpy[1] = head_t;
          tail_tmpy[1] = tail_t;
        }
        w_i[iw] = head_tmpy[0];
        w_i[iw + 1] = head_tmpy[1];
        ix += incx;
        iy += incy;
        iw += incw;
      }                                /* endfor */

      FPU_FIX_STOP;

      break;
    }
  }
}
