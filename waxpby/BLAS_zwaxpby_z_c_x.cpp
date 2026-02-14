#include "blas_extended.h"
#include "blas_extended_private.h"
#include "waxpby/XBLAS_waxpby.hpp"


void BLAS_zwaxpby_z_c_x(int n,
                        const void *alpha,
                        const void *x,
                        int incx,
                        const void *beta,
                        const void *y,
                        int incy,
                        void *w,
                        int incw,
                        enum blas_prec_type prec)
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
 * x     (input) const void*
 *       Array of length n.
 *
 * incx  (input) int
 *       The stride used to access components x[i].
 *
 * beta  (input) const void*
 *
 * y     (input) const void*
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
//static const char routine_name[] = "BLAS_zwaxpby_z_c_x";
#if 0
  XBLAS::waxpby_x(n,
                  *static_cast<const std::complex<double> *>(alpha),
                  static_cast<const std::complex<double> *>(x),
                  incx,
                  *static_cast<const std::complex<double> *>(beta),
                  static_cast<const std::complex<float> *>(y),
                  incy,
                  static_cast<std::complex<double> *>(w),
                  incw,
                  prec);
} /* end BLAS_zwaxpby_z_c_x */
#else
  static const char routine_name[] = "BLAS_zwaxpby_z_c_x";
  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, ix = 0, iy = 0, iw = 0;
      double *w_i = (double *) w;
      const double *x_i = (double *) x;
      const float *y_i = (float *) y;
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;
      double x_ii[2];
      float y_ii[2];
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



      incx *= 2;
      incy *= 2;
      incw *= 2;
      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;
      if (incw < 0)
        iw = (-n + 1) * incw;

      for (i = 0; i < n; ++i) {
        x_ii[0] = x_i[ix];
        x_ii[1] = x_i[ix + 1];
        y_ii[0] = y_i[iy];
        y_ii[1] = y_i[iy + 1];
        {
          tmpx[0] =
            (double) alpha_i[0] * x_ii[0] - (double) alpha_i[1] * x_ii[1];
          tmpx[1] =
            (double) alpha_i[0] * x_ii[1] + (double) alpha_i[1] * x_ii[0];
        }                        /* tmpx  = alpha * x[ix] */
        {
          tmpy[0] =
            (double) beta_i[0] * y_ii[0] - (double) beta_i[1] * y_ii[1];
          tmpy[1] =
            (double) beta_i[0] * y_ii[1] + (double) beta_i[1] * y_ii[0];
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
      const double *x_i = (double *) x;
      const float *y_i = (float *) y;
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;
      double x_ii[2];
      float y_ii[2];
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

      incx *= 2;
      incy *= 2;
      incw *= 2;
      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;
      if (incw < 0)
        iw = (-n + 1) * incw;

      for (i = 0; i < n; ++i) {
        x_ii[0] = x_i[ix];
        x_ii[1] = x_i[ix + 1];
        y_ii[0] = y_i[iy];
        y_ii[1] = y_i[iy + 1];
        {
          /* Compute complex-extra = complex-double * complex-double. */
          double head_t1, tail_t1;
          double head_t2, tail_t2;
          /* Real part */
          compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, alpha_i[0], x_ii[0]);
          compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, alpha_i[1], x_ii[1]);
          head_t2 = -head_t2;
          tail_t2 = -tail_t2;
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
          head_tmpx[0] = head_t1;
          tail_tmpx[0] = tail_t1;
          /* Imaginary part */
          compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, alpha_i[1], x_ii[0]);
          compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, alpha_i[0], x_ii[1]);
          compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
          head_tmpx[1] = head_t1;
          tail_tmpx[1] = tail_t1;
        }                        /* tmpx  = alpha * x[ix] */
        {
          double cd[2];
          cd[0] = (double) y_ii[0];
          cd[1] = (double) y_ii[1];
          {
            /* Compute complex-extra = complex-double * complex-double. */
            double head_t1, tail_t1;
            double head_t2, tail_t2;
            /* Real part */
            compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, beta_i[0], cd[0]);
            compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, beta_i[1], cd[1]);
            head_t2 = -head_t2;
            tail_t2 = -tail_t2;
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
            head_tmpy[0] = head_t1;
            tail_tmpy[0] = tail_t1;
            /* Imaginary part */
            compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, beta_i[1], cd[0]);
            compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, beta_i[0], cd[1]);
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
            head_tmpy[1] = head_t1;
            tail_tmpy[1] = tail_t1;
          }
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
#endif
