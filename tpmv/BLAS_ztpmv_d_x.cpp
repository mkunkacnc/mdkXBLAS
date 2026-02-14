#include "blas_extended.h"
#include "blas_extended_private.h"
#include "tpmv/XBLAS_tpmv.hpp"


void BLAS_ztpmv_d_x(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    enum blas_trans_type trans,
                    enum blas_diag_type diag,
                    int n,
                    const void *alpha,
                    const double *tp,
                    void *x,
                    int incx,
                    enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes x = alpha * tp * x, x = alpha * tp_transpose * x,
 * or x = alpha * tp_conjugate_transpose where tp is a triangular
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order        (input) enum blas_order_type
 *              Order of tp; row or column major
 *
 * uplo         (input) enum blas_uplo_type
 *              Whether tp is upper or lower
 *
 * trans        (input) enum blas_trans_type
 *
 * diag         (input) enum blas_diag_type
 *              Whether the diagonal entries of tp are 1
 *
 * n            (input) int
 *              Dimension of tp and the length of vector x
 *
 * alpha        (input) const void*
 *
 * tp           (input) const double*
 *
 * x            (input) void*
 *
 * incx         (input) int
 *              The stride for vector x.
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
//static const char routine_name[] = "BLAS_ztpmv_d_x";
#if 0
  XBLAS::tpmv_x(order,
                uplo,
                trans,
                diag,
                n,
                *static_cast<const std::complex<double> *>(alpha),
                tp,
                static_cast<std::complex<double> *>(x),
                incx,
                prec);
} /* end BLAS_ztpmv_d_x */
#else
  static const char routine_name[] = "BLAS_ztpmv_d_x";


  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      {
        int matrix_row, step, tp_index, tp_start, x_index, x_start;
        int inctp, x_index2, stride, col_index, inctp2;

        double *alpha_i = (double *) alpha;

        const double *tp_i = tp;
        double *x_i = (double *) x;
        double rowsum[2];
        double rowtmp[2];
        double result[2];
        double matval;
        double vecval[2];
        double one;


        one = 1.0;

        inctp = 1;

        incx *= 2;

        if (incx < 0)
          x_start = (-n + 1) * incx;
        else
          x_start = 0;

        if (n < 1) {
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
          BLAS_error(routine_name, -9, incx, NULL);
        }



        {
          if ((uplo == blas_upper &&
               trans == blas_no_trans && order == blas_rowmajor) ||
              (uplo == blas_lower &&
               trans != blas_no_trans && order == blas_colmajor)) {
            tp_start = 0;
            tp_index = tp_start;
            for (matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start + incx * matrix_row;
              x_index2 = x_index;
              col_index = matrix_row;
              rowsum[0] = rowsum[1] = 0.0;
              rowtmp[0] = rowtmp[1] = 0.0;
              result[0] = result[1] = 0.0;
              while (col_index < n) {
                vecval[0] = x_i[x_index];
                vecval[1] = x_i[x_index + 1];
                if ((diag == blas_unit_diag) && (col_index == matrix_row)) {
                  {
                    rowtmp[0] = vecval[0] * one;
                    rowtmp[1] = vecval[1] * one;
                  }
                } else {
                  matval = tp_i[tp_index];
                  {
                    rowtmp[0] = vecval[0] * matval;
                    rowtmp[1] = vecval[1] * matval;
                  }
                }
                rowsum[0] = rowsum[0] + rowtmp[0];
                rowsum[1] = rowsum[1] + rowtmp[1];
                x_index += incx;
                tp_index += inctp;
                col_index++;
              }
              {
                result[0] =
                  (double) rowsum[0] * alpha_i[0] -
                  (double) rowsum[1] * alpha_i[1];
                result[1] =
                  (double) rowsum[0] * alpha_i[1] +
                  (double) rowsum[1] * alpha_i[0];
              }
              x_i[x_index2] = result[0];
              x_i[x_index2 + 1] = result[1];
            }
          } else if ((uplo == blas_upper &&
                      trans == blas_no_trans && order == blas_colmajor) ||
                     (uplo == blas_lower &&
                      trans != blas_no_trans && order == blas_rowmajor)) {
            tp_start = ((n - 1) * n) / 2;
            inctp2 = n - 1;
            x_index2 = x_start;
            for (matrix_row = 0; matrix_row < n; matrix_row++, inctp2 = n - 1) {
              x_index = x_start + incx * (n - 1);
              tp_index = (tp_start + matrix_row) * inctp;
              col_index = (n - 1) - matrix_row;
              rowsum[0] = rowsum[1] = 0.0;
              rowtmp[0] = rowtmp[1] = 0.0;
              result[0] = result[1] = 0.0;
              while (col_index >= 0) {
                vecval[0] = x_i[x_index];
                vecval[1] = x_i[x_index + 1];
                if ((diag == blas_unit_diag) && (col_index == 0)) {
                  {
                    rowtmp[0] = vecval[0] * one;
                    rowtmp[1] = vecval[1] * one;
                  }
                } else {
                  matval = tp_i[tp_index];
                  {
                    rowtmp[0] = vecval[0] * matval;
                    rowtmp[1] = vecval[1] * matval;
                  }
                }
                rowsum[0] = rowsum[0] + rowtmp[0];
                rowsum[1] = rowsum[1] + rowtmp[1];
                x_index -= incx;
                tp_index -= inctp2 * inctp;
                inctp2--;
                col_index--;
              }
              {
                result[0] =
                  (double) rowsum[0] * alpha_i[0] -
                  (double) rowsum[1] * alpha_i[1];
                result[1] =
                  (double) rowsum[0] * alpha_i[1] +
                  (double) rowsum[1] * alpha_i[0];
              }
              x_i[x_index2] = result[0];
              x_i[x_index2 + 1] = result[1];
              x_index2 += incx;
            }
          } else if ((uplo == blas_lower &&
                      trans == blas_no_trans && order == blas_rowmajor) ||
                     (uplo == blas_upper &&
                      trans != blas_no_trans && order == blas_colmajor)) {
            tp_start = (n - 1) + ((n - 1) * n) / 2;
            tp_index = tp_start * inctp;
            x_index = x_start + (n - 1) * incx;

            for (matrix_row = n - 1; matrix_row >= 0; matrix_row--) {
              x_index2 = x_index;
              rowsum[0] = rowsum[1] = 0.0;
              rowtmp[0] = rowtmp[1] = 0.0;
              result[0] = result[1] = 0.0;
              for (step = 0; step <= matrix_row; step++) {
                vecval[0] = x_i[x_index2];
                vecval[1] = x_i[x_index2 + 1];
                if ((diag == blas_unit_diag) && (step == 0)) {
                  {
                    rowtmp[0] = vecval[0] * one;
                    rowtmp[1] = vecval[1] * one;
                  }
                } else {
                  matval = tp_i[tp_index];
                  {
                    rowtmp[0] = vecval[0] * matval;
                    rowtmp[1] = vecval[1] * matval;
                  }
                }
                rowsum[0] = rowsum[0] + rowtmp[0];
                rowsum[1] = rowsum[1] + rowtmp[1];
                x_index2 -= incx;
                tp_index -= inctp;
              }
              {
                result[0] =
                  (double) rowsum[0] * alpha_i[0] -
                  (double) rowsum[1] * alpha_i[1];
                result[1] =
                  (double) rowsum[0] * alpha_i[1] +
                  (double) rowsum[1] * alpha_i[0];
              }
              x_i[x_index] = result[0];
              x_i[x_index + 1] = result[1];
              x_index -= incx;
            }
          } else {
            tp_start = 0;
            x_index = x_start + (n - 1) * incx;
            for (matrix_row = n - 1; matrix_row >= 0; matrix_row--) {
              tp_index = matrix_row * inctp;
              x_index2 = x_start;
              rowsum[0] = rowsum[1] = 0.0;
              rowtmp[0] = rowtmp[1] = 0.0;
              result[0] = result[1] = 0.0;
              stride = n;
              for (step = 0; step <= matrix_row; step++) {
                vecval[0] = x_i[x_index2];
                vecval[1] = x_i[x_index2 + 1];
                if ((diag == blas_unit_diag) && (step == matrix_row)) {
                  {
                    rowtmp[0] = vecval[0] * one;
                    rowtmp[1] = vecval[1] * one;
                  }
                } else {
                  matval = tp_i[tp_index];
                  {
                    rowtmp[0] = vecval[0] * matval;
                    rowtmp[1] = vecval[1] * matval;
                  }
                }
                rowsum[0] = rowsum[0] + rowtmp[0];
                rowsum[1] = rowsum[1] + rowtmp[1];
                stride--;
                tp_index += stride * inctp;
                x_index2 += incx;
              }
              {
                result[0] =
                  (double) rowsum[0] * alpha_i[0] -
                  (double) rowsum[1] * alpha_i[1];
                result[1] =
                  (double) rowsum[0] * alpha_i[1] +
                  (double) rowsum[1] * alpha_i[0];
              }
              x_i[x_index] = result[0];
              x_i[x_index + 1] = result[1];
              x_index -= incx;
            }
          }
        }


      }
      break;
    }

  case blas_prec_extra:{

      {
        int matrix_row, step, tp_index, tp_start, x_index, x_start;
        int inctp, x_index2, stride, col_index, inctp2;

        double *alpha_i = (double *) alpha;

        const double *tp_i = tp;
        double *x_i = (double *) x;
        double head_rowsum[2], tail_rowsum[2];
        double head_rowtmp[2], tail_rowtmp[2];
        double head_result[2], tail_result[2];
        double matval;
        double vecval[2];
        double one;

        FPU_FIX_DECL;
        one = 1.0;

        inctp = 1;

        incx *= 2;

        if (incx < 0)
          x_start = (-n + 1) * incx;
        else
          x_start = 0;

        if (n < 1) {
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
          BLAS_error(routine_name, -9, incx, NULL);
        }
        FPU_FIX_START;


        {
          if ((uplo == blas_upper &&
               trans == blas_no_trans && order == blas_rowmajor) ||
              (uplo == blas_lower &&
               trans != blas_no_trans && order == blas_colmajor)) {
            tp_start = 0;
            tp_index = tp_start;
            for (matrix_row = 0; matrix_row < n; matrix_row++) {
              x_index = x_start + incx * matrix_row;
              x_index2 = x_index;
              col_index = matrix_row;
              head_rowsum[0] = head_rowsum[1] = tail_rowsum[0] =
                tail_rowsum[1] = 0.0;
              head_rowtmp[0] = head_rowtmp[1] = tail_rowtmp[0] =
                tail_rowtmp[1] = 0.0;
              head_result[0] = head_result[1] = tail_result[0] =
                tail_result[1] = 0.0;
              while (col_index < n) {
                vecval[0] = x_i[x_index];
                vecval[1] = x_i[x_index + 1];
                if ((diag == blas_unit_diag) && (col_index == matrix_row)) {
                  {
                    /* Compute complex-extra = complex-double * real. */
                    double head_t, tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, one, vecval[0]);
                    head_rowtmp[0] = head_t;
                    tail_rowtmp[0] = tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, one, vecval[1]);
                    head_rowtmp[1] = head_t;
                    tail_rowtmp[1] = tail_t;
                  }
                } else {
                  matval = tp_i[tp_index];
                  {
                    /* Compute complex-extra = complex-double * real. */
                    double head_t, tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, matval, vecval[0]);
                    head_rowtmp[0] = head_t;
                    tail_rowtmp[0] = tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, matval, vecval[1]);
                    head_rowtmp[1] = head_t;
                    tail_rowtmp[1] = tail_t;
                  }
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_rowsum[0];
                  tail_a = tail_rowsum[0];
                  head_b = head_rowtmp[0];
                  tail_b = tail_rowtmp[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_rowsum[0] = head_t;
                  tail_rowsum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_rowsum[1];
                  tail_a = tail_rowsum[1];
                  head_b = head_rowtmp[1];
                  tail_b = tail_rowtmp[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_rowsum[1] = head_t;
                  tail_rowsum[1] = tail_t;
                }
                x_index += incx;
                tp_index += inctp;
                col_index++;
              }
              {
                /* Compute complex-extra = complex-extra * complex-double. */
                double head_a0, tail_a0;
                double head_a1, tail_a1;
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                head_a0 = head_rowsum[0];
                tail_a0 = tail_rowsum[0];
                head_a1 = head_rowsum[1];
                tail_a1 = tail_rowsum[1];
                /* real part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_result[0] = head_t1;
                tail_result[0] = tail_t1;
                /* imaginary part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_result[1] = head_t1;
                tail_result[1] = tail_t1;
              }

              x_i[x_index2] = head_result[0];
              x_i[x_index2 + 1] = head_result[1];
            }
          } else if ((uplo == blas_upper &&
                      trans == blas_no_trans && order == blas_colmajor) ||
                     (uplo == blas_lower &&
                      trans != blas_no_trans && order == blas_rowmajor)) {
            tp_start = ((n - 1) * n) / 2;
            inctp2 = n - 1;
            x_index2 = x_start;
            for (matrix_row = 0; matrix_row < n; matrix_row++, inctp2 = n - 1) {
              x_index = x_start + incx * (n - 1);
              tp_index = (tp_start + matrix_row) * inctp;
              col_index = (n - 1) - matrix_row;
              head_rowsum[0] = head_rowsum[1] = tail_rowsum[0] =
                tail_rowsum[1] = 0.0;
              head_rowtmp[0] = head_rowtmp[1] = tail_rowtmp[0] =
                tail_rowtmp[1] = 0.0;
              head_result[0] = head_result[1] = tail_result[0] =
                tail_result[1] = 0.0;
              while (col_index >= 0) {
                vecval[0] = x_i[x_index];
                vecval[1] = x_i[x_index + 1];
                if ((diag == blas_unit_diag) && (col_index == 0)) {
                  {
                    /* Compute complex-extra = complex-double * real. */
                    double head_t, tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, one, vecval[0]);
                    head_rowtmp[0] = head_t;
                    tail_rowtmp[0] = tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, one, vecval[1]);
                    head_rowtmp[1] = head_t;
                    tail_rowtmp[1] = tail_t;
                  }
                } else {
                  matval = tp_i[tp_index];
                  {
                    /* Compute complex-extra = complex-double * real. */
                    double head_t, tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, matval, vecval[0]);
                    head_rowtmp[0] = head_t;
                    tail_rowtmp[0] = tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, matval, vecval[1]);
                    head_rowtmp[1] = head_t;
                    tail_rowtmp[1] = tail_t;
                  }
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_rowsum[0];
                  tail_a = tail_rowsum[0];
                  head_b = head_rowtmp[0];
                  tail_b = tail_rowtmp[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_rowsum[0] = head_t;
                  tail_rowsum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_rowsum[1];
                  tail_a = tail_rowsum[1];
                  head_b = head_rowtmp[1];
                  tail_b = tail_rowtmp[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_rowsum[1] = head_t;
                  tail_rowsum[1] = tail_t;
                }
                x_index -= incx;
                tp_index -= inctp2 * inctp;
                inctp2--;
                col_index--;
              }
              {
                /* Compute complex-extra = complex-extra * complex-double. */
                double head_a0, tail_a0;
                double head_a1, tail_a1;
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                head_a0 = head_rowsum[0];
                tail_a0 = tail_rowsum[0];
                head_a1 = head_rowsum[1];
                tail_a1 = tail_rowsum[1];
                /* real part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_result[0] = head_t1;
                tail_result[0] = tail_t1;
                /* imaginary part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_result[1] = head_t1;
                tail_result[1] = tail_t1;
              }

              x_i[x_index2] = head_result[0];
              x_i[x_index2 + 1] = head_result[1];
              x_index2 += incx;
            }
          } else if ((uplo == blas_lower &&
                      trans == blas_no_trans && order == blas_rowmajor) ||
                     (uplo == blas_upper &&
                      trans != blas_no_trans && order == blas_colmajor)) {
            tp_start = (n - 1) + ((n - 1) * n) / 2;
            tp_index = tp_start * inctp;
            x_index = x_start + (n - 1) * incx;

            for (matrix_row = n - 1; matrix_row >= 0; matrix_row--) {
              x_index2 = x_index;
              head_rowsum[0] = head_rowsum[1] = tail_rowsum[0] =
                tail_rowsum[1] = 0.0;
              head_rowtmp[0] = head_rowtmp[1] = tail_rowtmp[0] =
                tail_rowtmp[1] = 0.0;
              head_result[0] = head_result[1] = tail_result[0] =
                tail_result[1] = 0.0;
              for (step = 0; step <= matrix_row; step++) {
                vecval[0] = x_i[x_index2];
                vecval[1] = x_i[x_index2 + 1];
                if ((diag == blas_unit_diag) && (step == 0)) {
                  {
                    /* Compute complex-extra = complex-double * real. */
                    double head_t, tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, one, vecval[0]);
                    head_rowtmp[0] = head_t;
                    tail_rowtmp[0] = tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, one, vecval[1]);
                    head_rowtmp[1] = head_t;
                    tail_rowtmp[1] = tail_t;
                  }
                } else {
                  matval = tp_i[tp_index];
                  {
                    /* Compute complex-extra = complex-double * real. */
                    double head_t, tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, matval, vecval[0]);
                    head_rowtmp[0] = head_t;
                    tail_rowtmp[0] = tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, matval, vecval[1]);
                    head_rowtmp[1] = head_t;
                    tail_rowtmp[1] = tail_t;
                  }
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_rowsum[0];
                  tail_a = tail_rowsum[0];
                  head_b = head_rowtmp[0];
                  tail_b = tail_rowtmp[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_rowsum[0] = head_t;
                  tail_rowsum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_rowsum[1];
                  tail_a = tail_rowsum[1];
                  head_b = head_rowtmp[1];
                  tail_b = tail_rowtmp[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_rowsum[1] = head_t;
                  tail_rowsum[1] = tail_t;
                }
                x_index2 -= incx;
                tp_index -= inctp;
              }
              {
                /* Compute complex-extra = complex-extra * complex-double. */
                double head_a0, tail_a0;
                double head_a1, tail_a1;
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                head_a0 = head_rowsum[0];
                tail_a0 = tail_rowsum[0];
                head_a1 = head_rowsum[1];
                tail_a1 = tail_rowsum[1];
                /* real part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_result[0] = head_t1;
                tail_result[0] = tail_t1;
                /* imaginary part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_result[1] = head_t1;
                tail_result[1] = tail_t1;
              }

              x_i[x_index] = head_result[0];
              x_i[x_index + 1] = head_result[1];
              x_index -= incx;
            }
          } else {
            tp_start = 0;
            x_index = x_start + (n - 1) * incx;
            for (matrix_row = n - 1; matrix_row >= 0; matrix_row--) {
              tp_index = matrix_row * inctp;
              x_index2 = x_start;
              head_rowsum[0] = head_rowsum[1] = tail_rowsum[0] =
                tail_rowsum[1] = 0.0;
              head_rowtmp[0] = head_rowtmp[1] = tail_rowtmp[0] =
                tail_rowtmp[1] = 0.0;
              head_result[0] = head_result[1] = tail_result[0] =
                tail_result[1] = 0.0;
              stride = n;
              for (step = 0; step <= matrix_row; step++) {
                vecval[0] = x_i[x_index2];
                vecval[1] = x_i[x_index2 + 1];
                if ((diag == blas_unit_diag) && (step == matrix_row)) {
                  {
                    /* Compute complex-extra = complex-double * real. */
                    double head_t, tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, one, vecval[0]);
                    head_rowtmp[0] = head_t;
                    tail_rowtmp[0] = tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, one, vecval[1]);
                    head_rowtmp[1] = head_t;
                    tail_rowtmp[1] = tail_t;
                  }
                } else {
                  matval = tp_i[tp_index];
                  {
                    /* Compute complex-extra = complex-double * real. */
                    double head_t, tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, matval, vecval[0]);
                    head_rowtmp[0] = head_t;
                    tail_rowtmp[0] = tail_t;
                    compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, matval, vecval[1]);
                    head_rowtmp[1] = head_t;
                    tail_rowtmp[1] = tail_t;
                  }
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_rowsum[0];
                  tail_a = tail_rowsum[0];
                  head_b = head_rowtmp[0];
                  tail_b = tail_rowtmp[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_rowsum[0] = head_t;
                  tail_rowsum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_rowsum[1];
                  tail_a = tail_rowsum[1];
                  head_b = head_rowtmp[1];
                  tail_b = tail_rowtmp[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_rowsum[1] = head_t;
                  tail_rowsum[1] = tail_t;
                }
                stride--;
                tp_index += stride * inctp;
                x_index2 += incx;
              }
              {
                /* Compute complex-extra = complex-extra * complex-double. */
                double head_a0, tail_a0;
                double head_a1, tail_a1;
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                head_a0 = head_rowsum[0];
                tail_a0 = tail_rowsum[0];
                head_a1 = head_rowsum[1];
                tail_a1 = tail_rowsum[1];
                /* real part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_result[0] = head_t1;
                tail_result[0] = tail_t1;
                /* imaginary part */
                compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
                compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_result[1] = head_t1;
                tail_result[1] = tail_t1;
              }

              x_i[x_index] = head_result[0];
              x_i[x_index + 1] = head_result[1];
              x_index -= incx;
            }
          }
        }


      }
      break;
    }
  }

}
#endif
