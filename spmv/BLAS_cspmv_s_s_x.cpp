#include "blas_extended.h"
#include "blas_extended_private.h"
#include "spmv/XBLAS_spmv.hpp"

void BLAS_cspmv_s_s_x(enum blas_order_type order,
                      enum blas_uplo_type uplo,
                      int n,
                      const void *alpha,
                      const float *ap,
                      const float *x,
                      int incx,
                      const void *beta,
                      void *y,
                      int incy,
                      enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * ap * x + beta * y, where ap is a symmetric
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Order of ap; row or column major
 *
 * uplo   (input) enum blas_uplo_type
 *        Whether ap is upper or lower
 *
 * n      (input) int
 *        Dimension of ap and the length of vector x
 *
 * alpha  (input) const void*
 *
 * ap     (input) const float*
 *
 * x      (input) const float*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) const void*
 *
 * y      (input/output) void*
 *
 * incy   (input) int
 *        The stride for vector y.
 * prec   (input) enum blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 *
 */
{
//static const char routine_name[] = "BLAS_cspmv_s_s_x";
#if 0
  XBLAS::spmv_x(order,
                uplo,
                n,
                *static_cast<const std::complex<float> *>(alpha),
                ap,
                x,
                incx,
                *static_cast<const std::complex<float> *>(beta),
                static_cast<std::complex<float> *>(y),
                incy,
                prec);
} /* end BLAS_cspmv_s_s_x */
#else
  static const char routine_name[] = "BLAS_cspmv_s_s_x";

  switch (prec) {
  case blas_prec_single:{
      {
        int matrix_row, step, ap_index, ap_start, x_index, x_start;
        int y_start, y_index, incap;
        float *alpha_i = (float *) alpha;
        float *beta_i = (float *) beta;

        const float *ap_i = ap;
        const float *x_i = x;
        float *y_i = (float *) y;
        float rowsum;
        float rowtmp;
        float matval;
        float vecval;
        float resval[2];
        float tmp1[2];
        float tmp2[2];


        incap = 1;


        incy *= 2;

        if (incx < 0)
          x_start = (-n + 1) * incx;
        else
          x_start = 0;
        if (incy < 0)
          y_start = (-n + 1) * incy;
        else
          y_start = 0;

        if (n < 1) {
          return;
        }
        if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
            && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
          return;
        }

        /* Check for error conditions. */
        if (order != blas_colmajor && order != blas_rowmajor) {
          BLAS_error(routine_name, -1, order, NULL);
        }
        if (uplo != blas_upper && uplo != blas_lower) {
          BLAS_error(routine_name, -2, uplo, NULL);
        }
        if (incx == 0) {
          BLAS_error(routine_name, -7, incx, NULL);
        }
        if (incy == 0) {
          BLAS_error(routine_name, -10, incy, NULL);
        }



        if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
          {
            y_index = y_start;
            for (matrix_row = 0; matrix_row < n; matrix_row++) {
              resval[0] = y_i[y_index];
              resval[1] = y_i[y_index + 1];

              {
                tmp2[0] = beta_i[0] * resval[0] - beta_i[1] * resval[1];
                tmp2[1] = beta_i[0] * resval[1] + beta_i[1] * resval[0];
              }


              y_i[y_index] = tmp2[0];
              y_i[y_index + 1] = tmp2[1];

              y_index += incy;
            }
          }
        } else {
          if (uplo == blas_lower)
            order = (order == blas_rowmajor) ? blas_colmajor : blas_rowmajor;
          if (order == blas_rowmajor) {
            if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    tmp1[0] = rowsum;
                    tmp1[1] = 0.0;
                    y_i[y_index] = tmp1[0];
                    y_i[y_index + 1] = tmp1[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    tmp1[0] = rowsum;
                    tmp1[1] = 0.0;
                    {
                      tmp2[0] = beta_i[0] * resval[0] - beta_i[1] * resval[1];
                      tmp2[1] = beta_i[0] * resval[1] + beta_i[1] * resval[0];
                    }

                    tmp2[0] = tmp1[0] + tmp2[0];
                    tmp2[1] = tmp1[1] + tmp2[1];
                    y_i[y_index] = tmp2[0];
                    y_i[y_index + 1] = tmp2[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            } else {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    {
                      tmp1[0] = alpha_i[0] * rowsum;
                      tmp1[1] = alpha_i[1] * rowsum;
                    }
                    y_i[y_index] = tmp1[0];
                    y_i[y_index + 1] = tmp1[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    {
                      tmp1[0] = alpha_i[0] * rowsum;
                      tmp1[1] = alpha_i[1] * rowsum;
                    }
                    {
                      tmp2[0] = beta_i[0] * resval[0] - beta_i[1] * resval[1];
                      tmp2[1] = beta_i[0] * resval[1] + beta_i[1] * resval[0];
                    }

                    tmp2[0] = tmp1[0] + tmp2[0];
                    tmp2[1] = tmp1[1] + tmp2[1];
                    y_i[y_index] = tmp2[0];
                    y_i[y_index + 1] = tmp2[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            }
          } else {
            if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    tmp1[0] = rowsum;
                    tmp1[1] = 0.0;
                    y_i[y_index] = tmp1[0];
                    y_i[y_index + 1] = tmp1[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    tmp1[0] = rowsum;
                    tmp1[1] = 0.0;
                    {
                      tmp2[0] = beta_i[0] * resval[0] - beta_i[1] * resval[1];
                      tmp2[1] = beta_i[0] * resval[1] + beta_i[1] * resval[0];
                    }

                    tmp2[0] = tmp1[0] + tmp2[0];
                    tmp2[1] = tmp1[1] + tmp2[1];
                    y_i[y_index] = tmp2[0];
                    y_i[y_index + 1] = tmp2[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              }
            } else {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    {
                      tmp1[0] = alpha_i[0] * rowsum;
                      tmp1[1] = alpha_i[1] * rowsum;
                    }
                    y_i[y_index] = tmp1[0];
                    y_i[y_index + 1] = tmp1[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = matval * vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    {
                      tmp1[0] = alpha_i[0] * rowsum;
                      tmp1[1] = alpha_i[1] * rowsum;
                    }
                    {
                      tmp2[0] = beta_i[0] * resval[0] - beta_i[1] * resval[1];
                      tmp2[1] = beta_i[0] * resval[1] + beta_i[1] * resval[0];
                    }

                    tmp2[0] = tmp1[0] + tmp2[0];
                    tmp2[1] = tmp1[1] + tmp2[1];
                    y_i[y_index] = tmp2[0];
                    y_i[y_index + 1] = tmp2[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              }
            }
          }                        /* if order == ... */
        }                        /* alpha != 0 */


      }
      break;
    }
  case blas_prec_indigenous:
  case blas_prec_double:{
      {
        int matrix_row, step, ap_index, ap_start, x_index, x_start;
        int y_start, y_index, incap;
        float *alpha_i = (float *) alpha;
        float *beta_i = (float *) beta;

        const float *ap_i = ap;
        const float *x_i = x;
        float *y_i = (float *) y;
        double rowsum;
        double rowtmp;
        float matval;
        float vecval;
        float resval[2];
        double tmp1[2];
        double tmp2[2];


        incap = 1;


        incy *= 2;

        if (incx < 0)
          x_start = (-n + 1) * incx;
        else
          x_start = 0;
        if (incy < 0)
          y_start = (-n + 1) * incy;
        else
          y_start = 0;

        if (n < 1) {
          return;
        }
        if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
            && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
          return;
        }

        /* Check for error conditions. */
        if (order != blas_colmajor && order != blas_rowmajor) {
          BLAS_error(routine_name, -1, order, NULL);
        }
        if (uplo != blas_upper && uplo != blas_lower) {
          BLAS_error(routine_name, -2, uplo, NULL);
        }
        if (incx == 0) {
          BLAS_error(routine_name, -7, incx, NULL);
        }
        if (incy == 0) {
          BLAS_error(routine_name, -10, incy, NULL);
        }



        if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
          {
            y_index = y_start;
            for (matrix_row = 0; matrix_row < n; matrix_row++) {
              resval[0] = y_i[y_index];
              resval[1] = y_i[y_index + 1];

              {
                tmp2[0] =
                  (double) beta_i[0] * resval[0] -
                  (double) beta_i[1] * resval[1];
                tmp2[1] =
                  (double) beta_i[0] * resval[1] +
                  (double) beta_i[1] * resval[0];
              }

              y_i[y_index] = tmp2[0];
              y_i[y_index + 1] = tmp2[1];

              y_index += incy;
            }
          }
        } else {
          if (uplo == blas_lower)
            order = (order == blas_rowmajor) ? blas_colmajor : blas_rowmajor;
          if (order == blas_rowmajor) {
            if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    tmp1[0] = rowsum;
                    tmp1[1] = 0.0;
                    y_i[y_index] = tmp1[0];
                    y_i[y_index + 1] = tmp1[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    tmp1[0] = rowsum;
                    tmp1[1] = 0.0;
                    {
                      tmp2[0] =
                        (double) beta_i[0] * resval[0] -
                        (double) beta_i[1] * resval[1];
                      tmp2[1] =
                        (double) beta_i[0] * resval[1] +
                        (double) beta_i[1] * resval[0];
                    }
                    tmp2[0] = tmp1[0] + tmp2[0];
                    tmp2[1] = tmp1[1] + tmp2[1];
                    y_i[y_index] = tmp2[0];
                    y_i[y_index + 1] = tmp2[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            } else {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    {
                      tmp1[0] = alpha_i[0] * rowsum;
                      tmp1[1] = alpha_i[1] * rowsum;
                    }
                    y_i[y_index] = tmp1[0];
                    y_i[y_index + 1] = tmp1[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    {
                      tmp1[0] = alpha_i[0] * rowsum;
                      tmp1[1] = alpha_i[1] * rowsum;
                    }
                    {
                      tmp2[0] =
                        (double) beta_i[0] * resval[0] -
                        (double) beta_i[1] * resval[1];
                      tmp2[1] =
                        (double) beta_i[0] * resval[1] +
                        (double) beta_i[1] * resval[0];
                    }
                    tmp2[0] = tmp1[0] + tmp2[0];
                    tmp2[1] = tmp1[1] + tmp2[1];
                    y_i[y_index] = tmp2[0];
                    y_i[y_index + 1] = tmp2[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            }
          } else {
            if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    tmp1[0] = rowsum;
                    tmp1[1] = 0.0;
                    y_i[y_index] = tmp1[0];
                    y_i[y_index + 1] = tmp1[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    tmp1[0] = rowsum;
                    tmp1[1] = 0.0;
                    {
                      tmp2[0] =
                        (double) beta_i[0] * resval[0] -
                        (double) beta_i[1] * resval[1];
                      tmp2[1] =
                        (double) beta_i[0] * resval[1] +
                        (double) beta_i[1] * resval[0];
                    }
                    tmp2[0] = tmp1[0] + tmp2[0];
                    tmp2[1] = tmp1[1] + tmp2[1];
                    y_i[y_index] = tmp2[0];
                    y_i[y_index + 1] = tmp2[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              }
            } else {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    {
                      tmp1[0] = alpha_i[0] * rowsum;
                      tmp1[1] = alpha_i[1] * rowsum;
                    }
                    y_i[y_index] = tmp1[0];
                    y_i[y_index + 1] = tmp1[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    rowsum = 0.0;
                    rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      rowtmp = (double) matval *vecval;
                      rowsum = rowsum + rowtmp;
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    {
                      tmp1[0] = alpha_i[0] * rowsum;
                      tmp1[1] = alpha_i[1] * rowsum;
                    }
                    {
                      tmp2[0] =
                        (double) beta_i[0] * resval[0] -
                        (double) beta_i[1] * resval[1];
                      tmp2[1] =
                        (double) beta_i[0] * resval[1] +
                        (double) beta_i[1] * resval[0];
                    }
                    tmp2[0] = tmp1[0] + tmp2[0];
                    tmp2[1] = tmp1[1] + tmp2[1];
                    y_i[y_index] = tmp2[0];
                    y_i[y_index + 1] = tmp2[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              }
            }
          }                        /* if order == ... */
        }                        /* alpha != 0 */


      }
      break;
    }

  case blas_prec_extra:{
      {
        int matrix_row, step, ap_index, ap_start, x_index, x_start;
        int y_start, y_index, incap;
        float *alpha_i = (float *) alpha;
        float *beta_i = (float *) beta;

        const float *ap_i = ap;
        const float *x_i = x;
        float *y_i = (float *) y;
        double head_rowsum, tail_rowsum;
        double head_rowtmp, tail_rowtmp;
        float matval;
        float vecval;
        float resval[2];
        double head_tmp1[2], tail_tmp1[2];
        double head_tmp2[2], tail_tmp2[2];
        FPU_FIX_DECL;

        incap = 1;


        incy *= 2;

        if (incx < 0)
          x_start = (-n + 1) * incx;
        else
          x_start = 0;
        if (incy < 0)
          y_start = (-n + 1) * incy;
        else
          y_start = 0;

        if (n < 1) {
          return;
        }
        if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
            && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
          return;
        }

        /* Check for error conditions. */
        if (order != blas_colmajor && order != blas_rowmajor) {
          BLAS_error(routine_name, -1, order, NULL);
        }
        if (uplo != blas_upper && uplo != blas_lower) {
          BLAS_error(routine_name, -2, uplo, NULL);
        }
        if (incx == 0) {
          BLAS_error(routine_name, -7, incx, NULL);
        }
        if (incy == 0) {
          BLAS_error(routine_name, -10, incy, NULL);
        }

        FPU_FIX_START;

        if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
          {
            y_index = y_start;
            for (matrix_row = 0; matrix_row < n; matrix_row++) {
              resval[0] = y_i[y_index];
              resval[1] = y_i[y_index + 1];

              {
                double head_e1, tail_e1;
                double d1;
                double d2;
                /* Real part */
                d1 = (double) beta_i[0] * resval[0];
                d2 = (double) -beta_i[1] * resval[1];
                compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                head_tmp2[0] = head_e1;
                tail_tmp2[0] = tail_e1;
                /* imaginary part */
                d1 = (double) beta_i[0] * resval[1];
                d2 = (double) beta_i[1] * resval[0];
                compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                head_tmp2[1] = head_e1;
                tail_tmp2[1] = tail_e1;
              }

              y_i[y_index] = head_tmp2[0];
              y_i[y_index + 1] = head_tmp2[1];

              y_index += incy;
            }
          }
        } else {
          if (uplo == blas_lower)
            order = (order == blas_rowmajor) ? blas_colmajor : blas_rowmajor;
          if (order == blas_rowmajor) {
            if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    head_rowsum = tail_rowsum = 0.0;
                    head_rowtmp = tail_rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += incap;
                      x_index += incx;
                    }
                    head_tmp1[0] = head_rowsum;
                    tail_tmp1[0] = tail_rowsum;
                    head_tmp1[1] = tail_tmp1[1] = 0.0;
                    y_i[y_index] = head_tmp1[0];
                    y_i[y_index + 1] = head_tmp1[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    head_rowsum = tail_rowsum = 0.0;
                    head_rowtmp = tail_rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    head_tmp1[0] = head_rowsum;
                    tail_tmp1[0] = tail_rowsum;
                    head_tmp1[1] = tail_tmp1[1] = 0.0;
                    {
                      double head_e1, tail_e1;
                      double d1;
                      double d2;
                      /* Real part */
                      d1 = (double) beta_i[0] * resval[0];
                      d2 = (double) -beta_i[1] * resval[1];
                      compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                      head_tmp2[0] = head_e1;
                      tail_tmp2[0] = tail_e1;
                      /* imaginary part */
                      d1 = (double) beta_i[0] * resval[1];
                      d2 = (double) beta_i[1] * resval[0];
                      compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                      head_tmp2[1] = head_e1;
                      tail_tmp2[1] = tail_e1;
                    }
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
                      head_tmp2[0] = head_t;
                      tail_tmp2[0] = tail_t;
                      /* Imaginary part */
                      head_a = head_tmp1[1];
                      tail_a = tail_tmp1[1];
                      head_b = head_tmp2[1];
                      tail_b = tail_tmp2[1];
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                      head_tmp2[1] = head_t;
                      tail_tmp2[1] = tail_t;
                    }
                    y_i[y_index] = head_tmp2[0];
                    y_i[y_index + 1] = head_tmp2[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            } else {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    head_rowsum = tail_rowsum = 0.0;
                    head_rowtmp = tail_rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += incap;
                      x_index += incx;
                    }
                    {
                      double head_e1, tail_e1;
                      double dt;
                      dt = (double) alpha_i[0];
                      compute_doubledouble_eq_doubledouble_mul_double(&head_e1, &tail_e1, head_rowsum, tail_rowsum, dt);
                      head_tmp1[0] = head_e1;
                      tail_tmp1[0] = tail_e1;
                      dt = (double) alpha_i[1];
                      compute_doubledouble_eq_doubledouble_mul_double(&head_e1, &tail_e1, head_rowsum, tail_rowsum, dt);
                      head_tmp1[1] = head_e1;
                      tail_tmp1[1] = tail_e1;
                    }
                    y_i[y_index] = head_tmp1[0];
                    y_i[y_index + 1] = head_tmp1[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    head_rowsum = tail_rowsum = 0.0;
                    head_rowtmp = tail_rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += (n - step - 1) * incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    {
                      double head_e1, tail_e1;
                      double dt;
                      dt = (double) alpha_i[0];
                      compute_doubledouble_eq_doubledouble_mul_double(&head_e1, &tail_e1, head_rowsum, tail_rowsum, dt);
                      head_tmp1[0] = head_e1;
                      tail_tmp1[0] = tail_e1;
                      dt = (double) alpha_i[1];
                      compute_doubledouble_eq_doubledouble_mul_double(&head_e1, &tail_e1, head_rowsum, tail_rowsum, dt);
                      head_tmp1[1] = head_e1;
                      tail_tmp1[1] = tail_e1;
                    }
                    {
                      double head_e1, tail_e1;
                      double d1;
                      double d2;
                      /* Real part */
                      d1 = (double) beta_i[0] * resval[0];
                      d2 = (double) -beta_i[1] * resval[1];
                      compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                      head_tmp2[0] = head_e1;
                      tail_tmp2[0] = tail_e1;
                      /* imaginary part */
                      d1 = (double) beta_i[0] * resval[1];
                      d2 = (double) beta_i[1] * resval[0];
                      compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                      head_tmp2[1] = head_e1;
                      tail_tmp2[1] = tail_e1;
                    }
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
                      head_tmp2[0] = head_t;
                      tail_tmp2[0] = tail_t;
                      /* Imaginary part */
                      head_a = head_tmp1[1];
                      tail_a = tail_tmp1[1];
                      head_b = head_tmp2[1];
                      tail_b = tail_tmp2[1];
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                      head_tmp2[1] = head_t;
                      tail_tmp2[1] = tail_t;
                    }
                    y_i[y_index] = head_tmp2[0];
                    y_i[y_index + 1] = head_tmp2[1];

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            }
          } else {
            if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    head_rowsum = tail_rowsum = 0.0;
                    head_rowtmp = tail_rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    head_tmp1[0] = head_rowsum;
                    tail_tmp1[0] = tail_rowsum;
                    head_tmp1[1] = tail_tmp1[1] = 0.0;
                    y_i[y_index] = head_tmp1[0];
                    y_i[y_index + 1] = head_tmp1[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    head_rowsum = tail_rowsum = 0.0;
                    head_rowtmp = tail_rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    head_tmp1[0] = head_rowsum;
                    tail_tmp1[0] = tail_rowsum;
                    head_tmp1[1] = tail_tmp1[1] = 0.0;
                    {
                      double head_e1, tail_e1;
                      double d1;
                      double d2;
                      /* Real part */
                      d1 = (double) beta_i[0] * resval[0];
                      d2 = (double) -beta_i[1] * resval[1];
                      compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                      head_tmp2[0] = head_e1;
                      tail_tmp2[0] = tail_e1;
                      /* imaginary part */
                      d1 = (double) beta_i[0] * resval[1];
                      d2 = (double) beta_i[1] * resval[0];
                      compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                      head_tmp2[1] = head_e1;
                      tail_tmp2[1] = tail_e1;
                    }
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
                      head_tmp2[0] = head_t;
                      tail_tmp2[0] = tail_t;
                      /* Imaginary part */
                      head_a = head_tmp1[1];
                      tail_a = tail_tmp1[1];
                      head_b = head_tmp2[1];
                      tail_b = tail_tmp2[1];
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                      head_tmp2[1] = head_t;
                      tail_tmp2[1] = tail_t;
                    }
                    y_i[y_index] = head_tmp2[0];
                    y_i[y_index + 1] = head_tmp2[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              }
            } else {
              if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    head_rowsum = tail_rowsum = 0.0;
                    head_rowtmp = tail_rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    {
                      double head_e1, tail_e1;
                      double dt;
                      dt = (double) alpha_i[0];
                      compute_doubledouble_eq_doubledouble_mul_double(&head_e1, &tail_e1, head_rowsum, tail_rowsum, dt);
                      head_tmp1[0] = head_e1;
                      tail_tmp1[0] = tail_e1;
                      dt = (double) alpha_i[1];
                      compute_doubledouble_eq_doubledouble_mul_double(&head_e1, &tail_e1, head_rowsum, tail_rowsum, dt);
                      head_tmp1[1] = head_e1;
                      tail_tmp1[1] = tail_e1;
                    }
                    y_i[y_index] = head_tmp1[0];
                    y_i[y_index + 1] = head_tmp1[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              } else {
                {
                  y_index = y_start;
                  ap_start = 0;
                  for (matrix_row = 0; matrix_row < n; matrix_row++) {
                    x_index = x_start;
                    ap_index = ap_start;
                    head_rowsum = tail_rowsum = 0.0;
                    head_rowtmp = tail_rowtmp = 0.0;
                    for (step = 0; step < matrix_row; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += incap;
                      x_index += incx;
                    }
                    for (step = matrix_row; step < n; step++) {
                      matval = ap_i[ap_index];
                      vecval = x_i[x_index];
                      head_rowtmp = (double) matval *vecval;
                      tail_rowtmp = 0.0;
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_rowsum, &tail_rowsum, head_rowsum, tail_rowsum, head_rowtmp, tail_rowtmp);
                      ap_index += (step + 1) * incap;
                      x_index += incx;
                    }
                    resval[0] = y_i[y_index];
                    resval[1] = y_i[y_index + 1];
                    {
                      double head_e1, tail_e1;
                      double dt;
                      dt = (double) alpha_i[0];
                      compute_doubledouble_eq_doubledouble_mul_double(&head_e1, &tail_e1, head_rowsum, tail_rowsum, dt);
                      head_tmp1[0] = head_e1;
                      tail_tmp1[0] = tail_e1;
                      dt = (double) alpha_i[1];
                      compute_doubledouble_eq_doubledouble_mul_double(&head_e1, &tail_e1, head_rowsum, tail_rowsum, dt);
                      head_tmp1[1] = head_e1;
                      tail_tmp1[1] = tail_e1;
                    }
                    {
                      double head_e1, tail_e1;
                      double d1;
                      double d2;
                      /* Real part */
                      d1 = (double) beta_i[0] * resval[0];
                      d2 = (double) -beta_i[1] * resval[1];
                      compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                      head_tmp2[0] = head_e1;
                      tail_tmp2[0] = tail_e1;
                      /* imaginary part */
                      d1 = (double) beta_i[0] * resval[1];
                      d2 = (double) beta_i[1] * resval[0];
                      compute_doubledouble_eq_double_add_double(&head_e1, &tail_e1, d1, d2);
                      head_tmp2[1] = head_e1;
                      tail_tmp2[1] = tail_e1;
                    }
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
                      head_tmp2[0] = head_t;
                      tail_tmp2[0] = tail_t;
                      /* Imaginary part */
                      head_a = head_tmp1[1];
                      tail_a = tail_tmp1[1];
                      head_b = head_tmp2[1];
                      tail_b = tail_tmp2[1];
                      compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                      head_tmp2[1] = head_t;
                      tail_tmp2[1] = tail_t;
                    }
                    y_i[y_index] = head_tmp2[0];
                    y_i[y_index + 1] = head_tmp2[1];

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              }
            }
          }                        /* if order == ... */
        }                        /* alpha != 0 */

        FPU_FIX_STOP;
      }
      break;
    }

  }
}
#endif
