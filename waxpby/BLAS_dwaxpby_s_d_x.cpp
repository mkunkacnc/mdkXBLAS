#include "blas_extended.h"
#include "blas_extended_private.h"
#include "waxpby/XBLAS_waxpby.hpp"

void BLAS_dwaxpby_s_d_x(int n,
                        double alpha,
                        const float *x,
                        int incx,
                        double beta,
                        const double *y,
                        int incy,
                        double *w,
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
 * n      (input) int
 *        The length of vectors x, y, and w.
 *
 * alpha  (input) double
 *
 * x      (input) const float*
 *        Array of length n.
 *
 * incx   (input) int
 *        The stride used to access components x[i].
 *
 * beta   (input) double
 *
 * y      (input) const double*
 *        Array of length n.
 *
 * incy   (input) int
 *        The stride used to access components y[i].
 *
 * w      (output) double*
 *        Array of length n.
 *
 * incw   (input) int
 *        The stride used to write components w[i].
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
//static const char routine_name[] = "BLAS_dwaxpby_s_d_x";
#if 0
  XBLAS::waxpby_x(n, alpha, x, incx, beta, y, incy, w, incw, prec);
} /* end BLAS_dwaxpby_s_d_x */
#else
  static const char routine_name[] = "BLAS_dwaxpby_s_d_x";
  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, ix = 0, iy = 0, iw = 0;
      double *w_i = w;
      const float *x_i = x;
      const double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      float x_ii;
      double y_ii;
      double tmpx;
      double tmpy;



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






      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;
      if (incw < 0)
        iw = (-n + 1) * incw;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii = y_i[iy];
        tmpx = alpha_i * x_ii;        /* tmpx  = alpha * x[ix] */
        tmpy = beta_i * y_ii;        /* tmpy = beta * y[iy] */
        tmpy = tmpy + tmpx;
        w_i[iw] = tmpy;
        ix += incx;
        iy += incy;
        iw += incw;
      }                                /* endfor */



      break;
    }

  case blas_prec_extra:{

      int i, ix = 0, iy = 0, iw = 0;
      double *w_i = w;
      const float *x_i = x;
      const double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      float x_ii;
      double y_ii;
      double head_tmpx, tail_tmpx;
      double head_tmpy, tail_tmpy;

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
          double dt = (double) x_ii;
          compute_doubledouble_eq_double_mul_double(&head_tmpx, &tail_tmpx, alpha_i, dt);
        }                        /* tmpx  = alpha * x[ix] */
        compute_doubledouble_eq_double_mul_double(&head_tmpy, &tail_tmpy, beta_i, y_ii);
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmpy, &tail_tmpy, head_tmpy, tail_tmpy, head_tmpx, tail_tmpx);
        w_i[iw] = head_tmpy;
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
