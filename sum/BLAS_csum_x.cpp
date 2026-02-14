#include "blas_extended.h"
#include "blas_extended_private.h"
#include "sum/XBLAS_sum.hpp"

void BLAS_csum_x(int n,
                 const void *x,
                 int incx,
                 void *sum,
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
 * x     (input) const void*
 *       Array of length n.
 *
 * incx  (input) int
 *       The stride used to access components x[i].
 *
 * sum   (output) void*
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
//static const char routine_name[] = "BLAS_csum_x";
#if 0
  XBLAS::sum_x(n,
               static_cast<const std::complex<float> *>(x),
               incx,
               static_cast<std::complex<float> *>(sum),
               prec);
} /* end BLAS_csum_x */
#else
  static const char routine_name[] = "BLAS_csum_x";
  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, xi;
      float *sum_i = (float *) sum;
      const float *x_i = (float *) x;
      float x_elem[2];
      double tmp[2];


      /* Test the input parameters. */
      if (n < 0)
        BLAS_error(routine_name, -1, n, NULL);
      if (incx == 0)
        BLAS_error(routine_name, -3, incx, NULL);

      /* Immediate return. */
      if (n <= 0) {
        sum_i[0] = sum_i[1] = 0.0;
        return;
      }



      tmp[0] = tmp[1] = 0.0;

      incx *= 2;
      if (incx < 0)
        xi = -(n - 1) * incx;
      else
        xi = 0;

      for (i = 0; i < n; i++, xi += incx) {
        x_elem[0] = x_i[xi];
        x_elem[1] = x_i[xi + 1];
        tmp[0] = tmp[0] + x_elem[0];
        tmp[1] = tmp[1] + x_elem[1];
      }
      ((float *) sum)[0] = tmp[0];
      ((float *) sum)[1] = tmp[1];



      break;
    }

  case blas_prec_extra:
    {
      int i, xi;
      float *sum_i = (float *) sum;
      const float *x_i = (float *) x;
      float x_elem[2];
      double head_tmp[2], tail_tmp[2];
      FPU_FIX_DECL;

      /* Test the input parameters. */
      if (n < 0)
        BLAS_error(routine_name, -1, n, NULL);
      if (incx == 0)
        BLAS_error(routine_name, -3, incx, NULL);

      /* Immediate return. */
      if (n <= 0) {
        sum_i[0] = sum_i[1] = 0.0;
        return;
      }

      FPU_FIX_START;

      head_tmp[0] = head_tmp[1] = tail_tmp[0] = tail_tmp[1] = 0.0;

      incx *= 2;
      if (incx < 0)
        xi = -(n - 1) * incx;
      else
        xi = 0;

      for (i = 0; i < n; i++, xi += incx) {
        x_elem[0] = x_i[xi];
        x_elem[1] = x_i[xi + 1];
        {
          double cd[2];
          cd[0] = (double) x_elem[0];
          cd[1] = (double) x_elem[1];
          {
            double head_t, tail_t;
            double head_a, tail_a;
            head_a = head_tmp[0];
            tail_a = tail_tmp[0];
            compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_a, tail_a, cd[0]);
            head_tmp[0] = head_t;
            tail_tmp[0] = tail_t;
            head_a = head_tmp[1];
            tail_a = tail_tmp[1];
            compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_a, tail_a, cd[1]);
            head_tmp[1] = head_t;
            tail_tmp[1] = tail_t;
          }
        }
      }
      ((float *) sum)[0] = head_tmp[0];
      ((float *) sum)[1] = head_tmp[1];

      FPU_FIX_STOP;
    }
    break;
  }
}
#endif
