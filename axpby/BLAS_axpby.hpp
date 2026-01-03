#ifndef XBLAS_AXPBY_HPP
#define XBLAS_AXPBY_HPP

#include "blas_extended_private.h"

template<typename X>
void BLAS_daxpby_x_cpp(int n, double alpha, const X *x, int incx,
                       double beta, double *y, int incy, enum blas_prec_type prec)
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
 * alpha     (input) double
 *
 * x         (input) const X*
 *           Array of length n.
 *
 * incx      (input) int
 *           The stride used to access components x[i].
 *
 * beta      (input) double
 *
 * y         (input) double*
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
  static const char routine_name[] = "BLAS_daxpby_x_cpp";

  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:
    {
      int i, ix = 0, iy = 0;
      const X *x_i = x;
      double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      X x_ii;
      double y_ii;
      double tmpx;
      double tmpy;

      /* Test the input parameters. */
      if (incx == 0)
        BLAS_error(routine_name, -4, incx, NULL);
      else if (incy == 0)
        BLAS_error(routine_name, -7, incy, NULL);

      /* Immediate return */
      if (n <= 0 || (alpha_i == 0.0 && beta_i == 1.0))
        return;

      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii = y_i[iy];
        tmpx = alpha_i * x_ii;        /* tmpx  = alpha * x[ix] */
        tmpy = beta_i * y_ii;        /* tmpy = beta * y[iy] */
        tmpy = tmpy + tmpx;
        y_i[iy] = tmpy;
        ix += incx;
        iy += incy;
      }                                /* endfor */
    }
    break;
  case blas_prec_extra:
    {
      int i, ix = 0, iy = 0;
      const X *x_i = x;
      double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      X x_ii;
      double y_ii;
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
        compute_doubledouble_eq_double_mul_double(&head_tmpx, &tail_tmpx, alpha_i, x_ii);
        compute_doubledouble_eq_double_mul_double(&head_tmpy, &tail_tmpy, beta_i, y_ii);
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmpy, &tail_tmpy, head_tmpx, tail_tmpx, head_tmpy, tail_tmpy);
        y_i[iy] = head_tmpy;
        ix += incx;
        iy += incy;
      }                                /* endfor */

      FPU_FIX_STOP;
    }
    break;
  }
}                                /* end BLAS_daxpby_x_cpp */

#endif
