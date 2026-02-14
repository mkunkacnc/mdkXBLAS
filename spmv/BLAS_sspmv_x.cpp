#include "blas_extended.h"
#include "blas_extended_private.h"
#include "spmv/XBLAS_spmv.hpp"


void BLAS_sspmv_x(enum blas_order_type order,
                  enum blas_uplo_type uplo,
                  int n,
                  float alpha,
                  const float *ap,
                  const float *x,
                  int incx,
                  float beta,
                  float *y,
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
 * alpha  (input) float
 *
 * ap     (input) const float*
 *
 * x      (input) const float*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) float
 *
 * y      (input/output) float*
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
//static const char routine_name[] = "BLAS_sspmv_x";
#if 0
  XBLAS::spmv_x(order, uplo, n, alpha, ap, x, incx, beta, y, incy, prec);
} /* end BLAS_sspmv_x */
#else
  static const char routine_name[] = "BLAS_sspmv_x";

  switch (prec) {
  case blas_prec_single:{
      {
        int matrix_row, step, ap_index, ap_start, x_index, x_start;
        int y_start, y_index, incap;
        float alpha_i = alpha;
        float beta_i = beta;

        const float *ap_i = ap;
        const float *x_i = x;
        float *y_i = y;
        float rowsum;
        float rowtmp;
        float matval;
        float vecval;
        float resval;
        float tmp1;
        float tmp2;


        incap = 1;




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
        if (alpha_i == 0.0 && beta_i == 1.0) {
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



        if (alpha_i == 0.0) {
          {
            y_index = y_start;
            for (matrix_row = 0; matrix_row < n; matrix_row++) {
              resval = y_i[y_index];

              tmp2 = beta_i * resval;

              y_i[y_index] = tmp2;

              y_index += incy;
            }
          }
        } else {
          if (uplo == blas_lower)
            order = (order == blas_rowmajor) ? blas_colmajor : blas_rowmajor;
          if (order == blas_rowmajor) {
            if (alpha_i == 1.0) {
              if (beta_i == 0.0) {
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
                    tmp1 = rowsum;
                    y_i[y_index] = tmp1;

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
                    resval = y_i[y_index];
                    tmp1 = rowsum;
                    tmp2 = beta_i * resval;
                    tmp2 = tmp1 + tmp2;
                    y_i[y_index] = tmp2;

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            } else {
              if (beta_i == 0.0) {
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
                    tmp1 = rowsum * alpha_i;
                    y_i[y_index] = tmp1;

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
                    resval = y_i[y_index];
                    tmp1 = rowsum * alpha_i;
                    tmp2 = beta_i * resval;
                    tmp2 = tmp1 + tmp2;
                    y_i[y_index] = tmp2;

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            }
          } else {
            if (alpha_i == 1.0) {
              if (beta_i == 0.0) {
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
                    tmp1 = rowsum;
                    y_i[y_index] = tmp1;

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
                    resval = y_i[y_index];
                    tmp1 = rowsum;
                    tmp2 = beta_i * resval;
                    tmp2 = tmp1 + tmp2;
                    y_i[y_index] = tmp2;

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              }
            } else {
              if (beta_i == 0.0) {
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
                    tmp1 = rowsum * alpha_i;
                    y_i[y_index] = tmp1;

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
                    resval = y_i[y_index];
                    tmp1 = rowsum * alpha_i;
                    tmp2 = beta_i * resval;
                    tmp2 = tmp1 + tmp2;
                    y_i[y_index] = tmp2;

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
        float alpha_i = alpha;
        float beta_i = beta;

        const float *ap_i = ap;
        const float *x_i = x;
        float *y_i = y;
        double rowsum;
        double rowtmp;
        float matval;
        float vecval;
        float resval;
        double tmp1;
        double tmp2;


        incap = 1;




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
        if (alpha_i == 0.0 && beta_i == 1.0) {
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



        if (alpha_i == 0.0) {
          {
            y_index = y_start;
            for (matrix_row = 0; matrix_row < n; matrix_row++) {
              resval = y_i[y_index];

              tmp2 = (double) beta_i *resval;

              y_i[y_index] = tmp2;

              y_index += incy;
            }
          }
        } else {
          if (uplo == blas_lower)
            order = (order == blas_rowmajor) ? blas_colmajor : blas_rowmajor;
          if (order == blas_rowmajor) {
            if (alpha_i == 1.0) {
              if (beta_i == 0.0) {
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
                    tmp1 = rowsum;
                    y_i[y_index] = tmp1;

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
                    resval = y_i[y_index];
                    tmp1 = rowsum;
                    tmp2 = (double) beta_i *resval;
                    tmp2 = tmp1 + tmp2;
                    y_i[y_index] = tmp2;

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            } else {
              if (beta_i == 0.0) {
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
                    tmp1 = rowsum * alpha_i;
                    y_i[y_index] = tmp1;

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
                    resval = y_i[y_index];
                    tmp1 = rowsum * alpha_i;
                    tmp2 = (double) beta_i *resval;
                    tmp2 = tmp1 + tmp2;
                    y_i[y_index] = tmp2;

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            }
          } else {
            if (alpha_i == 1.0) {
              if (beta_i == 0.0) {
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
                    tmp1 = rowsum;
                    y_i[y_index] = tmp1;

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
                    resval = y_i[y_index];
                    tmp1 = rowsum;
                    tmp2 = (double) beta_i *resval;
                    tmp2 = tmp1 + tmp2;
                    y_i[y_index] = tmp2;

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              }
            } else {
              if (beta_i == 0.0) {
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
                    tmp1 = rowsum * alpha_i;
                    y_i[y_index] = tmp1;

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
                    resval = y_i[y_index];
                    tmp1 = rowsum * alpha_i;
                    tmp2 = (double) beta_i *resval;
                    tmp2 = tmp1 + tmp2;
                    y_i[y_index] = tmp2;

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
        float alpha_i = alpha;
        float beta_i = beta;

        const float *ap_i = ap;
        const float *x_i = x;
        float *y_i = y;
        double head_rowsum, tail_rowsum;
        double head_rowtmp, tail_rowtmp;
        float matval;
        float vecval;
        float resval;
        double head_tmp1, tail_tmp1;
        double head_tmp2, tail_tmp2;
        FPU_FIX_DECL;

        incap = 1;




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
        if (alpha_i == 0.0 && beta_i == 1.0) {
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

        if (alpha_i == 0.0) {
          {
            y_index = y_start;
            for (matrix_row = 0; matrix_row < n; matrix_row++) {
              resval = y_i[y_index];

              head_tmp2 = (double) beta_i *resval;
              tail_tmp2 = 0.0;

              y_i[y_index] = head_tmp2;

              y_index += incy;
            }
          }
        } else {
          if (uplo == blas_lower)
            order = (order == blas_rowmajor) ? blas_colmajor : blas_rowmajor;
          if (order == blas_rowmajor) {
            if (alpha_i == 1.0) {
              if (beta_i == 0.0) {
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
                    head_tmp1 = head_rowsum;
                    tail_tmp1 = tail_rowsum;
                    y_i[y_index] = head_tmp1;

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
                    resval = y_i[y_index];
                    head_tmp1 = head_rowsum;
                    tail_tmp1 = tail_rowsum;
                    head_tmp2 = (double) beta_i *resval;
                    tail_tmp2 = 0.0;
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp2, &tail_tmp2, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
                    y_i[y_index] = head_tmp2;

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            } else {
              if (beta_i == 0.0) {
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
                      double dt = (double) alpha_i;
                      compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_rowsum, tail_rowsum, dt);
                    }
                    y_i[y_index] = head_tmp1;

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
                    resval = y_i[y_index];
                    {
                      double dt = (double) alpha_i;
                      compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_rowsum, tail_rowsum, dt);
                    }
                    head_tmp2 = (double) beta_i *resval;
                    tail_tmp2 = 0.0;
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp2, &tail_tmp2, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
                    y_i[y_index] = head_tmp2;

                    y_index += incy;
                    ap_start += incap;
                  }
                }
              }
            }
          } else {
            if (alpha_i == 1.0) {
              if (beta_i == 0.0) {
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
                    head_tmp1 = head_rowsum;
                    tail_tmp1 = tail_rowsum;
                    y_i[y_index] = head_tmp1;

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
                    resval = y_i[y_index];
                    head_tmp1 = head_rowsum;
                    tail_tmp1 = tail_rowsum;
                    head_tmp2 = (double) beta_i *resval;
                    tail_tmp2 = 0.0;
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp2, &tail_tmp2, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
                    y_i[y_index] = head_tmp2;

                    y_index += incy;
                    ap_start += (matrix_row + 1) * incap;
                  }
                }
              }
            } else {
              if (beta_i == 0.0) {
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
                      double dt = (double) alpha_i;
                      compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_rowsum, tail_rowsum, dt);
                    }
                    y_i[y_index] = head_tmp1;

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
                    resval = y_i[y_index];
                    {
                      double dt = (double) alpha_i;
                      compute_doubledouble_eq_doubledouble_mul_double(&head_tmp1, &tail_tmp1, head_rowsum, tail_rowsum, dt);
                    }
                    head_tmp2 = (double) beta_i *resval;
                    tail_tmp2 = 0.0;
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_tmp2, &tail_tmp2, head_tmp1, tail_tmp1, head_tmp2, tail_tmp2);
                    y_i[y_index] = head_tmp2;

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
