#include "blas_extended.h"
#include "blas_extended_private.h"
#include "axpby/BLAS_axpby.hpp"

void BLAS_saxpby_x(int n, float alpha, const float *x, int incx,
                   float beta, float *y, int incy, enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine computes:
 *
 *      y <- alpha * x + beta * y.
 *
 * Arguments
 * =========
 *
 * n         (input) int
 *           The length of vectors x and y.
 *
 * alpha     (input) float
 *
 * x         (input) const float*
 *           Array of length n.
 *
 * incx      (input) int
 *           The stride used to access components x[i].
 *
 * beta      (input) float
 *
 * y         (input) float*
 *           Array of length n.
 *
 * incy      (input) int
 *           The stride used to access components y[i].
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
  static const char routine_name[] = "BLAS_saxpby_x";

  switch (prec) {
  case blas_prec_single:
    BLAS_axpby_cpp(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
  case blas_prec_indigenous:
    BLAS_axpby_cpp<float, float, double>(n, alpha, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    {
      int i, ix = 0, iy = 0;
      const float *x_i = x;
      float *y_i = y;
      float alpha_i = alpha;
      float beta_i = beta;
      float x_ii;
      float y_ii;
      double head_tmpx, tail_tmpx;
      double head_tmpy, tail_tmpy;
      FPU_FIX_DECL;

      /* Test the input parameters. */
      if (incx == 0)
        BLAS_error(routine_name, -4, incx, NULL);
      else if (incy == 0)
        BLAS_error(routine_name, -7, incy, NULL);

      /* Immediate return */
      if (n <= 0 || (alpha_i == 0.0 && beta_i == 1.0))
        return;

      FPU_FIX_START;

      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii = y_i[iy];
        compute_doubledouble_eq_float_mul_float(&head_tmpx, &tail_tmpx, alpha_i, x_ii);
        compute_doubledouble_eq_float_mul_float(&head_tmpy, &tail_tmpy, beta_i, y_ii);
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmpy, &tail_tmpy, head_tmpy, tail_tmpy, head_tmpx, tail_tmpx);
        y_i[iy] = head_tmpy;
        ix += incx;
        iy += incy;
      }                         /* endfor */

      FPU_FIX_STOP;
    }
    break;
  }
}                               /* end BLAS_saxpby_x */
