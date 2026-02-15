#include "blas_extended.h"
#include "blas_extended_private.h"
#include "sum/XBLAS_sum.hpp"

void BLAS_dsum_x(int n,
                 const double *x,
                 int incx,
                 double *sum,
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
 * x     (input) const double*
 *       Array of length n.
 *
 * incx  (input) int
 *       The stride used to access components x[i].
 *
 * sum   (output) double*
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
//static const char routine_name[] = "BLAS_dsum_x";
#if 1
  XBLAS::sum_x(n, x, incx, sum, prec);
} /* end BLAS_dsum_x */
#else
  static const char routine_name[] = "BLAS_dsum_x";
  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, xi;
      double *sum_i = sum;
      const double *x_i = x;
      double x_elem;
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
      double *sum_i = sum;
      const double *x_i = x;
      double x_elem;
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
        compute_doubledouble_eq_doubledouble_add_double(&head_tmp, &tail_tmp, head_tmp, tail_tmp, x_elem);
      }
      *sum = head_tmp;

      FPU_FIX_STOP;
    }
    break;
  }
}
#endif
