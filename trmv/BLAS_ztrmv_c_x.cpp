#include "blas_extended.h"
#include "blas_extended_private.h"
#include "trmv/XBLAS_trmv.hpp"


void BLAS_ztrmv_c_x(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    enum blas_trans_type trans,
                    enum blas_diag_type diag,
                    int n,
                    const void *alpha,
                    const void *T,
                    int ldt,
                    void *x,
                    int incx,
                    enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes x <-- alpha * T * x, where T is a triangular matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        column major, row major
 *
 * uplo   (input) enum blas_uplo_type
 *        upper, lower
 *
 * trans  (input) enum blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) enum blas_diag_type
 *        unit, non unit
 *
 * n      (input) int
 *        the dimension of T
 *
 * alpha  (input) const void*
 *
 * T      (input) const void*
 *        Triangular matrix
 *
 * ldt    (input) int
 *        Leading dimension of T
 *
 * x      (input) void*
 *    Array of length n.
 *
 * incx   (input) int
 *     The stride used to access components x[i].
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
//static const char routine_name[] = "BLAS_ztrmv_c_x";
#if 0
  XBLAS::trmv_x(order,
                uplo,
                trans,
                diag,
                n,
                *static_cast<const std::complex<double> *>(alpha),
                static_cast<const std::complex<float> *>(T),
                ldt,
                static_cast<std::complex<double> *>(x),
                incx,
                prec);
} /* end BLAS_ztrmv_c_x */
#else
  static const char routine_name[] = "BLAS_ztrmv_c_x";

  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, j;                        /* used to idx matrix */
      int xj, xj0;
      int ti, tij, tij0;

      int inc_ti, inc_tij;
      int inc_x;

      const float *T_i = (float *) T;        /* internal matrix T */
      double *x_i = (double *) x;        /* internal x */
      double *alpha_i = (double *) alpha;        /* internal alpha */

      float t_elem[2];
      double x_elem[2];
      double prod[2];
      double sum[2];
      double tmp[2];



      /* all error calls */
      if ((order != blas_rowmajor && order != blas_colmajor) ||
          (uplo != blas_upper && uplo != blas_lower) ||
          (trans != blas_trans &&
           trans != blas_no_trans &&
           trans != blas_conj_trans) ||
          (diag != blas_non_unit_diag && diag != blas_unit_diag) ||
          (ldt < n) || (incx == 0)) {
        BLAS_error(routine_name, 0, 0, NULL);
      } else if (n <= 0) {
        BLAS_error(routine_name, -4, n, NULL);
      } else if (incx == 0) {
        BLAS_error(routine_name, -9, incx, NULL);
      }

      if (trans == blas_no_trans) {
        if (uplo == blas_upper) {
          inc_x = -incx;
          if (order == blas_rowmajor) {
            inc_ti = ldt;
            inc_tij = -1;
          } else {
            inc_ti = 1;
            inc_tij = -ldt;
          }
        } else {
          inc_x = incx;
          if (order == blas_rowmajor) {
            inc_ti = -ldt;
            inc_tij = 1;
          } else {
            inc_ti = -1;
            inc_tij = ldt;
          }
        }
      } else {
        if (uplo == blas_upper) {
          inc_x = incx;
          if (order == blas_rowmajor) {
            inc_ti = -1;
            inc_tij = ldt;
          } else {
            inc_ti = -ldt;
            inc_tij = 1;
          }
        } else {
          inc_x = -incx;
          if (order == blas_rowmajor) {
            inc_ti = 1;
            inc_tij = -ldt;
          } else {
            inc_ti = ldt;
            inc_tij = -1;
          }
        }
      }

      inc_ti *= 2;
      inc_tij *= 2;
      inc_x *= 2;

      xj0 = (inc_x > 0 ? 0 : -(n - 1) * inc_x);
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
        xj = xj0;
        for (j = 0; j < n; j++) {
          x_i[xj] = 0.0;
          x_i[xj + 1] = 0.0;
          xj += inc_x;
        }
      } else {

        if (diag == blas_unit_diag) {
          if (trans == blas_conj_trans) {


            ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
            tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
            for (i = 0; i < n; i++) {

              sum[0] = sum[1] = 0.0;

              xj = xj0;
              tij = ti + tij0;
              for (j = i; j < (n - 1); j++) {

                t_elem[0] = T_i[tij];
                t_elem[1] = T_i[tij + 1];
                t_elem[1] = -t_elem[1];
                x_elem[0] = x_i[xj];
                x_elem[1] = x_i[xj + 1];
                {
                  prod[0] =
                    (double) x_elem[0] * t_elem[0] -
                    (double) x_elem[1] * t_elem[1];
                  prod[1] =
                    (double) x_elem[0] * t_elem[1] +
                    (double) x_elem[1] * t_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];

                xj += inc_x;
                tij += inc_tij;
              }

              x_elem[0] = x_i[xj];
              x_elem[1] = x_i[xj + 1];
              sum[0] = sum[0] + x_elem[0];
              sum[1] = sum[1] + x_elem[1];

              if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
                x_i[xj] = sum[0];
                x_i[xj + 1] = sum[1];
              } else {
                {
                  tmp[0] =
                    (double) sum[0] * alpha_i[0] -
                    (double) sum[1] * alpha_i[1];
                  tmp[1] =
                    (double) sum[0] * alpha_i[1] +
                    (double) sum[1] * alpha_i[0];
                }
                x_i[xj] = tmp[0];
                x_i[xj + 1] = tmp[1];
              }

              ti += inc_ti;
            }

          } else {


            ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
            tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
            for (i = 0; i < n; i++) {

              sum[0] = sum[1] = 0.0;

              xj = xj0;
              tij = ti + tij0;
              for (j = i; j < (n - 1); j++) {

                t_elem[0] = T_i[tij];
                t_elem[1] = T_i[tij + 1];

                x_elem[0] = x_i[xj];
                x_elem[1] = x_i[xj + 1];
                {
                  prod[0] =
                    (double) x_elem[0] * t_elem[0] -
                    (double) x_elem[1] * t_elem[1];
                  prod[1] =
                    (double) x_elem[0] * t_elem[1] +
                    (double) x_elem[1] * t_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];

                xj += inc_x;
                tij += inc_tij;
              }

              x_elem[0] = x_i[xj];
              x_elem[1] = x_i[xj + 1];
              sum[0] = sum[0] + x_elem[0];
              sum[1] = sum[1] + x_elem[1];

              if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
                x_i[xj] = sum[0];
                x_i[xj + 1] = sum[1];
              } else {
                {
                  tmp[0] =
                    (double) sum[0] * alpha_i[0] -
                    (double) sum[1] * alpha_i[1];
                  tmp[1] =
                    (double) sum[0] * alpha_i[1] +
                    (double) sum[1] * alpha_i[0];
                }
                x_i[xj] = tmp[0];
                x_i[xj + 1] = tmp[1];
              }

              ti += inc_ti;
            }

          }
        } else {
          if (trans == blas_conj_trans) {


            ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
            tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
            for (i = 0; i < n; i++) {

              sum[0] = sum[1] = 0.0;

              xj = xj0;
              tij = ti + tij0;
              for (j = i; j < n; j++) {

                t_elem[0] = T_i[tij];
                t_elem[1] = T_i[tij + 1];
                t_elem[1] = -t_elem[1];
                x_elem[0] = x_i[xj];
                x_elem[1] = x_i[xj + 1];
                {
                  prod[0] =
                    (double) x_elem[0] * t_elem[0] -
                    (double) x_elem[1] * t_elem[1];
                  prod[1] =
                    (double) x_elem[0] * t_elem[1] +
                    (double) x_elem[1] * t_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];

                xj += inc_x;
                tij += inc_tij;
              }

              if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
                x_i[xj - inc_x] = sum[0];
                x_i[xj - inc_x + 1] = sum[1];
              } else {
                {
                  tmp[0] =
                    (double) sum[0] * alpha_i[0] -
                    (double) sum[1] * alpha_i[1];
                  tmp[1] =
                    (double) sum[0] * alpha_i[1] +
                    (double) sum[1] * alpha_i[0];
                }
                x_i[xj - inc_x] = tmp[0];
                x_i[xj - inc_x + 1] = tmp[1];
              }

              ti += inc_ti;
            }

          } else {


            ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
            tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
            for (i = 0; i < n; i++) {

              sum[0] = sum[1] = 0.0;

              xj = xj0;
              tij = ti + tij0;
              for (j = i; j < n; j++) {

                t_elem[0] = T_i[tij];
                t_elem[1] = T_i[tij + 1];

                x_elem[0] = x_i[xj];
                x_elem[1] = x_i[xj + 1];
                {
                  prod[0] =
                    (double) x_elem[0] * t_elem[0] -
                    (double) x_elem[1] * t_elem[1];
                  prod[1] =
                    (double) x_elem[0] * t_elem[1] +
                    (double) x_elem[1] * t_elem[0];
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];

                xj += inc_x;
                tij += inc_tij;
              }

              if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
                x_i[xj - inc_x] = sum[0];
                x_i[xj - inc_x + 1] = sum[1];
              } else {
                {
                  tmp[0] =
                    (double) sum[0] * alpha_i[0] -
                    (double) sum[1] * alpha_i[1];
                  tmp[1] =
                    (double) sum[0] * alpha_i[1] +
                    (double) sum[1] * alpha_i[0];
                }
                x_i[xj - inc_x] = tmp[0];
                x_i[xj - inc_x + 1] = tmp[1];
              }

              ti += inc_ti;
            }

          }
        }

      }



      break;
    }

  case blas_prec_extra:{

      int i, j;                        /* used to idx matrix */
      int xj, xj0;
      int ti, tij, tij0;

      int inc_ti, inc_tij;
      int inc_x;

      const float *T_i = (float *) T;        /* internal matrix T */
      double *x_i = (double *) x;        /* internal x */
      double *alpha_i = (double *) alpha;        /* internal alpha */

      float t_elem[2];
      double x_elem[2];
      double head_prod[2], tail_prod[2];
      double head_sum[2], tail_sum[2];
      double head_tmp[2], tail_tmp[2];

      FPU_FIX_DECL;
      FPU_FIX_START;

      /* all error calls */
      if ((order != blas_rowmajor && order != blas_colmajor) ||
          (uplo != blas_upper && uplo != blas_lower) ||
          (trans != blas_trans &&
           trans != blas_no_trans &&
           trans != blas_conj_trans) ||
          (diag != blas_non_unit_diag && diag != blas_unit_diag) ||
          (ldt < n) || (incx == 0)) {
        BLAS_error(routine_name, 0, 0, NULL);
      } else if (n <= 0) {
        BLAS_error(routine_name, -4, n, NULL);
      } else if (incx == 0) {
        BLAS_error(routine_name, -9, incx, NULL);
      }

      if (trans == blas_no_trans) {
        if (uplo == blas_upper) {
          inc_x = -incx;
          if (order == blas_rowmajor) {
            inc_ti = ldt;
            inc_tij = -1;
          } else {
            inc_ti = 1;
            inc_tij = -ldt;
          }
        } else {
          inc_x = incx;
          if (order == blas_rowmajor) {
            inc_ti = -ldt;
            inc_tij = 1;
          } else {
            inc_ti = -1;
            inc_tij = ldt;
          }
        }
      } else {
        if (uplo == blas_upper) {
          inc_x = incx;
          if (order == blas_rowmajor) {
            inc_ti = -1;
            inc_tij = ldt;
          } else {
            inc_ti = -ldt;
            inc_tij = 1;
          }
        } else {
          inc_x = -incx;
          if (order == blas_rowmajor) {
            inc_ti = 1;
            inc_tij = -ldt;
          } else {
            inc_ti = ldt;
            inc_tij = -1;
          }
        }
      }

      inc_ti *= 2;
      inc_tij *= 2;
      inc_x *= 2;

      xj0 = (inc_x > 0 ? 0 : -(n - 1) * inc_x);
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
        xj = xj0;
        for (j = 0; j < n; j++) {
          x_i[xj] = 0.0;
          x_i[xj + 1] = 0.0;
          xj += inc_x;
        }
      } else {

        if (diag == blas_unit_diag) {
          if (trans == blas_conj_trans) {


            ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
            tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
            for (i = 0; i < n; i++) {

              head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

              xj = xj0;
              tij = ti + tij0;
              for (j = i; j < (n - 1); j++) {

                t_elem[0] = T_i[tij];
                t_elem[1] = T_i[tij + 1];
                t_elem[1] = -t_elem[1];
                x_elem[0] = x_i[xj];
                x_elem[1] = x_i[xj + 1];
                {
                  double cd[2];
                  cd[0] = (double) t_elem[0];
                  cd[1] = (double) t_elem[1];
                  {
                    /* Compute complex-extra = complex-double * complex-double. */
                    double head_t1, tail_t1;
                    double head_t2, tail_t2;
                    /* Real part */
                    compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[0], cd[0]);
                    compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[1], cd[1]);
                    head_t2 = -head_t2;
                    tail_t2 = -tail_t2;
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                    head_prod[0] = head_t1;
                    tail_prod[0] = tail_t1;
                    /* Imaginary part */
                    compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[1], cd[0]);
                    compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[0], cd[1]);
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                    head_prod[1] = head_t1;
                    tail_prod[1] = tail_t1;
                  }
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }

                xj += inc_x;
                tij += inc_tij;
              }

              x_elem[0] = x_i[xj];
              x_elem[1] = x_i[xj + 1];
              {
                double head_t, tail_t;
                double head_a, tail_a;
                head_a = head_sum[0];
                tail_a = tail_sum[0];
                compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_a, tail_a, x_elem[0]);
                head_sum[0] = head_t;
                tail_sum[0] = tail_t;
                head_a = head_sum[1];
                tail_a = tail_sum[1];
                compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_a, tail_a, x_elem[1]);
                head_sum[1] = head_t;
                tail_sum[1] = tail_t;
              }

              if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
                x_i[xj] = head_sum[0];
                x_i[xj + 1] = head_sum[1];
              } else {
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_sum[0];
                  tail_a0 = tail_sum[0];
                  head_a1 = head_sum[1];
                  tail_a1 = tail_sum[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp[0] = head_t1;
                  tail_tmp[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp[1] = head_t1;
                  tail_tmp[1] = tail_t1;
                }

                x_i[xj] = head_tmp[0];
                x_i[xj + 1] = head_tmp[1];
              }

              ti += inc_ti;
            }

          } else {


            ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
            tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
            for (i = 0; i < n; i++) {

              head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

              xj = xj0;
              tij = ti + tij0;
              for (j = i; j < (n - 1); j++) {

                t_elem[0] = T_i[tij];
                t_elem[1] = T_i[tij + 1];

                x_elem[0] = x_i[xj];
                x_elem[1] = x_i[xj + 1];
                {
                  double cd[2];
                  cd[0] = (double) t_elem[0];
                  cd[1] = (double) t_elem[1];
                  {
                    /* Compute complex-extra = complex-double * complex-double. */
                    double head_t1, tail_t1;
                    double head_t2, tail_t2;
                    /* Real part */
                    compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[0], cd[0]);
                    compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[1], cd[1]);
                    head_t2 = -head_t2;
                    tail_t2 = -tail_t2;
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                    head_prod[0] = head_t1;
                    tail_prod[0] = tail_t1;
                    /* Imaginary part */
                    compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[1], cd[0]);
                    compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[0], cd[1]);
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                    head_prod[1] = head_t1;
                    tail_prod[1] = tail_t1;
                  }
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }

                xj += inc_x;
                tij += inc_tij;
              }

              x_elem[0] = x_i[xj];
              x_elem[1] = x_i[xj + 1];
              {
                double head_t, tail_t;
                double head_a, tail_a;
                head_a = head_sum[0];
                tail_a = tail_sum[0];
                compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_a, tail_a, x_elem[0]);
                head_sum[0] = head_t;
                tail_sum[0] = tail_t;
                head_a = head_sum[1];
                tail_a = tail_sum[1];
                compute_doubledouble_eq_doubledouble_add_double(&head_t, &tail_t, head_a, tail_a, x_elem[1]);
                head_sum[1] = head_t;
                tail_sum[1] = tail_t;
              }

              if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
                x_i[xj] = head_sum[0];
                x_i[xj + 1] = head_sum[1];
              } else {
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_sum[0];
                  tail_a0 = tail_sum[0];
                  head_a1 = head_sum[1];
                  tail_a1 = tail_sum[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp[0] = head_t1;
                  tail_tmp[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp[1] = head_t1;
                  tail_tmp[1] = tail_t1;
                }

                x_i[xj] = head_tmp[0];
                x_i[xj + 1] = head_tmp[1];
              }

              ti += inc_ti;
            }

          }
        } else {
          if (trans == blas_conj_trans) {


            ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
            tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
            for (i = 0; i < n; i++) {

              head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

              xj = xj0;
              tij = ti + tij0;
              for (j = i; j < n; j++) {

                t_elem[0] = T_i[tij];
                t_elem[1] = T_i[tij + 1];
                t_elem[1] = -t_elem[1];
                x_elem[0] = x_i[xj];
                x_elem[1] = x_i[xj + 1];
                {
                  double cd[2];
                  cd[0] = (double) t_elem[0];
                  cd[1] = (double) t_elem[1];
                  {
                    /* Compute complex-extra = complex-double * complex-double. */
                    double head_t1, tail_t1;
                    double head_t2, tail_t2;
                    /* Real part */
                    compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[0], cd[0]);
                    compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[1], cd[1]);
                    head_t2 = -head_t2;
                    tail_t2 = -tail_t2;
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                    head_prod[0] = head_t1;
                    tail_prod[0] = tail_t1;
                    /* Imaginary part */
                    compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[1], cd[0]);
                    compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[0], cd[1]);
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                    head_prod[1] = head_t1;
                    tail_prod[1] = tail_t1;
                  }
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }

                xj += inc_x;
                tij += inc_tij;
              }

              if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
                x_i[xj - inc_x] = head_sum[0];
                x_i[xj - inc_x + 1] = head_sum[1];
              } else {
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_sum[0];
                  tail_a0 = tail_sum[0];
                  head_a1 = head_sum[1];
                  tail_a1 = tail_sum[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp[0] = head_t1;
                  tail_tmp[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp[1] = head_t1;
                  tail_tmp[1] = tail_t1;
                }

                x_i[xj - inc_x] = head_tmp[0];
                x_i[xj - inc_x + 1] = head_tmp[1];
              }

              ti += inc_ti;
            }

          } else {


            ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
            tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
            for (i = 0; i < n; i++) {

              head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

              xj = xj0;
              tij = ti + tij0;
              for (j = i; j < n; j++) {

                t_elem[0] = T_i[tij];
                t_elem[1] = T_i[tij + 1];

                x_elem[0] = x_i[xj];
                x_elem[1] = x_i[xj + 1];
                {
                  double cd[2];
                  cd[0] = (double) t_elem[0];
                  cd[1] = (double) t_elem[1];
                  {
                    /* Compute complex-extra = complex-double * complex-double. */
                    double head_t1, tail_t1;
                    double head_t2, tail_t2;
                    /* Real part */
                    compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[0], cd[0]);
                    compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[1], cd[1]);
                    head_t2 = -head_t2;
                    tail_t2 = -tail_t2;
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                    head_prod[0] = head_t1;
                    tail_prod[0] = tail_t1;
                    /* Imaginary part */
                    compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[1], cd[0]);
                    compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[0], cd[1]);
                    compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                    head_prod[1] = head_t1;
                    tail_prod[1] = tail_t1;
                  }
                }
                {
                  double head_t, tail_t;
                  double head_a, tail_a;
                  double head_b, tail_b;
                  /* Real part */
                  head_a = head_sum[0];
                  tail_a = tail_sum[0];
                  head_b = head_prod[0];
                  tail_b = tail_prod[0];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[0] = head_t;
                  tail_sum[0] = tail_t;
                  /* Imaginary part */
                  head_a = head_sum[1];
                  tail_a = tail_sum[1];
                  head_b = head_prod[1];
                  tail_b = tail_prod[1];
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                  head_sum[1] = head_t;
                  tail_sum[1] = tail_t;
                }

                xj += inc_x;
                tij += inc_tij;
              }

              if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
                x_i[xj - inc_x] = head_sum[0];
                x_i[xj - inc_x + 1] = head_sum[1];
              } else {
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_sum[0];
                  tail_a0 = tail_sum[0];
                  head_a1 = head_sum[1];
                  tail_a1 = tail_sum[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp[0] = head_t1;
                  tail_tmp[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_tmp[1] = head_t1;
                  tail_tmp[1] = tail_t1;
                }

                x_i[xj - inc_x] = head_tmp[0];
                x_i[xj - inc_x + 1] = head_tmp[1];
              }

              ti += inc_ti;
            }

          }
        }

      }

      FPU_FIX_STOP;

      break;
    }
  }
}
#endif
