#include "blas_extended.h"
#include "blas_extended_private.h"
#include "sum/XBLAS_sum.hpp"

void BLAS_ssum_x(int n,
                 const float *x,
                 int incx,
                 float *sum,
                 enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine computes the summation:
 *
 *     sum <- SUM_{i=0, n-1} x[i].
 *
 * Arguments
 * =========
 *
 * n     (input) int
 *       The length of vector x.
 *
 * x     (input) const float*
 *       Array of length n.
 *
 * incx  (input) int
 *       The stride used to access components x[i].
 *
 * sum   (output) float*
 *
 * prec  (input) enum blas_prec_type
 *       Specifies the internal precision to be used.
 *       = blas_prec_single: single precision.
 *       = blas_prec_double: double precision.
 *       = blas_prec_extra : anything at least 1.5 times as accurate
 *                           than double, and wider than 80-bits.
 *                           We use double-double in our implementation.
 *
 */
{
//static const char routine_name[] = "BLAS_ssum_x";
#if 0
  XBLAS::sum_x(n, x, incx, sum, prec);
} /* end BLAS_ssum_x */
#else
  static const char routine_name[] = "BLAS_ssum_x";
  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, xi;
      float *sum_i = sum;
      const float *x_i = x;
      float x_elem;
      double tmp;


      /* Test the input parameters. */
      if (n < 0)
        BLAS_error(routine_name, -1, n, NULL);
      if (incx == 0)
        BLAS_error(routine_name, -3, incx, NULL);

      /* Immediate return. */
      if (n <= 0) {
        *sum_i = 0.0;
        return;
      }



      tmp = 0.0;


      if (incx < 0)
        xi = -(n - 1) * incx;
      else
        xi = 0;

      for (i = 0; i < n; i++, xi += incx) {
        x_elem = x_i[xi];
        tmp = tmp + x_elem;
      }
      *sum = tmp;



      break;
    }

  case blas_prec_extra:
    {
      int i, xi;
      float *sum_i = sum;
      const float *x_i = x;
      float x_elem;
      double head_tmp, tail_tmp;
      FPU_FIX_DECL;

      /* Test the input parameters. */
      if (n < 0)
        BLAS_error(routine_name, -1, n, NULL);
      if (incx == 0)
        BLAS_error(routine_name, -3, incx, NULL);

      /* Immediate return. */
      if (n <= 0) {
        *sum_i = 0.0;
        return;
      }

      FPU_FIX_START;

      head_tmp = tail_tmp = 0.0;


      if (incx < 0)
        xi = -(n - 1) * incx;
      else
        xi = 0;

      for (i = 0; i < n; i++, xi += incx) {
        x_elem = x_i[xi];
        {
          double dt = (double) x_elem;
          compute_doubledouble_eq_doubledouble_add_double(&head_tmp, &tail_tmp, head_tmp, tail_tmp, dt);
        }
      }
      *sum = head_tmp;

      FPU_FIX_STOP;
    }
    break;
  }
}
#endif
