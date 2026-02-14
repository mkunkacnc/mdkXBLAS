#include "blas_extended.h"
#include "blas_extended_private.h"
#include "trmv/XBLAS_trmv.hpp"


void BLAS_dtrmv_x(enum blas_order_type order,
                  enum blas_uplo_type uplo,
                  enum blas_trans_type trans,
                  enum blas_diag_type diag,
                  int n,
                  double alpha,
                  const double *T,
                  int ldt,
                  double *x,
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
 * alpha  (input) double
 *
 * T      (input) const double*
 *        Triangular matrix
 *
 * ldt    (input) int
 *        Leading dimension of T
 *
 * x      (input/output) double*
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
//static const char routine_name[] = "BLAS_dtrmv_x";
#if 0
  XBLAS::trmv_x(order, uplo, trans, diag, n, alpha, T, ldt, x, incx, prec);
} /* end BLAS_dtrmv_x */
#else
  static const char routine_name[] = "BLAS_dtrmv_x";

  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, j;                        /* used to idx matrix */
      int xj, xj0;
      int ti, tij, tij0;

      int inc_ti, inc_tij;
      int inc_x;

      const double *T_i = T;        /* internal matrix T */
      double *x_i = x;                /* internal x */
      double alpha_i = alpha;        /* internal alpha */

      double t_elem;
      double x_elem;
      double prod;
      double sum;
      double tmp;



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





      xj0 = (inc_x > 0 ? 0 : -(n - 1) * inc_x);
      if (alpha_i == 0.0) {
        xj = xj0;
        for (j = 0; j < n; j++) {
          x_i[xj] = 0.0;
          xj += inc_x;
        }
      } else {

        if (diag == blas_unit_diag) {


          ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
          tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
          for (i = 0; i < n; i++) {

            sum = 0.0;

            xj = xj0;
            tij = ti + tij0;
            for (j = i; j < (n - 1); j++) {

              t_elem = T_i[tij];

              x_elem = x_i[xj];
              prod = x_elem * t_elem;
              sum = sum + prod;

              xj += inc_x;
              tij += inc_tij;
            }

            x_elem = x_i[xj];
            sum = sum + x_elem;

            if (alpha_i == 1.0) {
              x_i[xj] = sum;
            } else {
              tmp = sum * alpha_i;
              x_i[xj] = tmp;
            }

            ti += inc_ti;
          }

        } else {


          ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
          tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
          for (i = 0; i < n; i++) {

            sum = 0.0;

            xj = xj0;
            tij = ti + tij0;
            for (j = i; j < n; j++) {

              t_elem = T_i[tij];

              x_elem = x_i[xj];
              prod = x_elem * t_elem;
              sum = sum + prod;

              xj += inc_x;
              tij += inc_tij;
            }

            if (alpha_i == 1.0) {
              x_i[xj - inc_x] = sum;
            } else {
              tmp = sum * alpha_i;
              x_i[xj - inc_x] = tmp;
            }

            ti += inc_ti;
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

      const double *T_i = T;        /* internal matrix T */
      double *x_i = x;                /* internal x */
      double alpha_i = alpha;        /* internal alpha */

      double t_elem;
      double x_elem;
      double head_prod, tail_prod;
      double head_sum, tail_sum;
      double head_tmp, tail_tmp;

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





      xj0 = (inc_x > 0 ? 0 : -(n - 1) * inc_x);
      if (alpha_i == 0.0) {
        xj = xj0;
        for (j = 0; j < n; j++) {
          x_i[xj] = 0.0;
          xj += inc_x;
        }
      } else {

        if (diag == blas_unit_diag) {


          ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
          tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
          for (i = 0; i < n; i++) {

            head_sum = tail_sum = 0.0;

            xj = xj0;
            tij = ti + tij0;
            for (j = i; j < (n - 1); j++) {

              t_elem = T_i[tij];

              x_elem = x_i[xj];
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, x_elem, t_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);

              xj += inc_x;
              tij += inc_tij;
            }

            x_elem = x_i[xj];
            compute_doubledouble_eq_doubledouble_add_double(&head_sum, &tail_sum, head_sum, tail_sum, x_elem);

            if (alpha_i == 1.0) {
              x_i[xj] = head_sum;
            } else {
              compute_doubledouble_eq_doubledouble_mul_double(&head_tmp, &tail_tmp, head_sum, tail_sum, alpha_i);
              x_i[xj] = head_tmp;
            }

            ti += inc_ti;
          }

        } else {


          ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
          tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
          for (i = 0; i < n; i++) {

            head_sum = tail_sum = 0.0;

            xj = xj0;
            tij = ti + tij0;
            for (j = i; j < n; j++) {

              t_elem = T_i[tij];

              x_elem = x_i[xj];
              compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, x_elem, t_elem);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);

              xj += inc_x;
              tij += inc_tij;
            }

            if (alpha_i == 1.0) {
              x_i[xj - inc_x] = head_sum;
            } else {
              compute_doubledouble_eq_doubledouble_mul_double(&head_tmp, &tail_tmp, head_sum, tail_sum, alpha_i);
              x_i[xj - inc_x] = head_tmp;
            }

            ti += inc_ti;
          }

        }

      }

      FPU_FIX_STOP;

      break;
    }
  }
}
#endif
