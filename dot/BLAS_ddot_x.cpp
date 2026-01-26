#include "blas_extended.h"
#include "blas_extended_private.h"
#include "dot/XBLAS_dot.hpp"

void BLAS_ddot_x(enum blas_conj_type conj,
                 int n,
                 double alpha,
                 const double *x,
                 int incx,
                 double beta,
                 const double *y,
                 int incy,
                 double *r,
                 enum blas_prec_type prec)
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
 * alpha  (input) double
 *
 * x      (input) const double*
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
 * r      (input/output) double*
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
  static const char routine_name[] = "BLAS_ddot_x";

  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:
    {
      int i, ix = 0, iy = 0;
      double *r_i = r;
      const double *x_i = x;
      const double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      double x_ii;
      double y_ii;
      double r_v;
      double prod;
      double sum;
      double tmp1;
      double tmp2;


      /* Test the input parameters. */
      if (n < 0)
        BLAS_error(routine_name, -2, n, NULL);
      else if (incx == 0)
        BLAS_error(routine_name, -5, incx, NULL);
      else if (incy == 0)
        BLAS_error(routine_name, -8, incy, NULL);

      /* Immediate return. */
      if ((beta_i == 1.0) && (n == 0 || (alpha_i == 0.0)))
        return;



      r_v = r_i[0];
      sum = 0.0;


      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii = y_i[iy];

        prod = x_ii * y_ii;        /* prod = x[i]*y[i] */
        sum = sum + prod;        /* sum = sum+prod */
        ix += incx;
        iy += incy;
      }                                /* endfor */


      tmp1 = sum * alpha_i;        /* tmp1 = sum*alpha */
      tmp2 = r_v * beta_i;        /* tmp2 = r*beta */
      tmp1 = tmp1 + tmp2;        /* tmp1 = tmp1+tmp2 */
      *r = tmp1;                /* r = tmp1 */


    }
    break;
  case blas_prec_extra:
    {
      int i, ix = 0, iy = 0;
      double *r_i = r;
      const double *x_i = x;
      const double *y_i = y;
      double alpha_i = alpha;
      double beta_i = beta;
      double x_ii;
      double y_ii;
      double r_v;
      double head_prod, tail_prod;
      double head_sum, tail_sum;
      double head_tmp1, tail_tmp1;
      double head_tmp2, tail_tmp2;
      FPU_FIX_DECL;

      /* Test the input parameters. */
      if (n < 0)
        BLAS_error(routine_name, -2, n, NULL);
      else if (incx == 0)
        BLAS_error(routine_name, -5, incx, NULL);
      else if (incy == 0)
        BLAS_error(routine_name, -8, incy, NULL);

      /* Immediate return. */
      if ((beta_i == 1.0) && (n == 0 || (alpha_i == 0.0)))
        return;

      FPU_FIX_START;

      r_v = r_i[0];
      head_sum = tail_sum = 0.0;


      if (incx < 0)
        ix = (-n + 1) * incx;
      if (incy < 0)
        iy = (-n + 1) * incy;

      for (i = 0; i < n; ++i) {
        x_ii = x_i[ix];
        y_ii = y_i[iy];

        compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, x_ii, y_ii);
        compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
        ix += incx;
        iy += incy;
      }                                /* endfor */


      compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_sum, tail_sum, alpha_i);                                /* tmp1 = sum*alpha */
      compute_doubledouble_eq_double_mul_double(&head_tmp2, &tail_tmp2, r_v, beta_i);
      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp1, &tail_tmp1, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
      *r = head_tmp1;                /* r = tmp1 */

      FPU_FIX_STOP;
    }
    break;
  }
}
