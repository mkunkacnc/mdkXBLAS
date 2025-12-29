#include <math.h>
#include "blas_extended.h"
#include "blas_extended_private.h"
void BLAS_ztbsv_d_x(enum blas_order_type order, enum blas_uplo_type uplo,
                    enum blas_trans_type trans, enum blas_diag_type diag,
                    int n, int k, const void *alpha, const double *t, int ldt,
                    void *x, int incx, enum blas_prec_type prec)

/*
 * Purpose
 * =======
 *
 * This routine solves :
 *
 *     x <- alpha * inverse(t) * x
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        column major, row major (blas_rowmajor, blas_colmajor)
 *
 * uplo   (input) enum blas_uplo_type
 *        upper, lower (blas_upper, blas_lower)
 *
 * trans  (input) enum blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) enum blas_diag_type
 *        unit, non unit (blas_unit_diag, blas_non_unit_diag)
 *
 * n      (input) int
 *        the dimension of t
 *
 * k      (input) int
 *        the number of subdiagonals/superdiagonals of t
 *
 * alpha  (input) const void*
 *
 * t      (input) double*
 *        Triangular Banded matrix
 *
 * x      (input) const void*
 *           Array of length n.
 *
 * incx   (input) int
 *           The stride used to access components x[i].
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
  /* Routine name */
  static const char routine_name[] = "BLAS_ztbsv_d_x";

  int i, j;                        /* used to keep track of loop counts */
  int xi;                        /* used to index vector x */
  int start_xi;                        /* used as the starting idx to vector x */
  int incxi;
  int Tij;                        /* index inside of Banded structure */
  int dot_start, dot_start_inc1, dot_start_inc2, dot_inc;

  const double *t_i = t;        /* internal matrix t */
  double *x_i = (double *) x;        /* internal x */
  double *alpha_i = (double *) alpha;        /* internal alpha */

  if (order != blas_rowmajor && order != blas_colmajor) {
    BLAS_error(routine_name, -1, order, 0);
  }
  if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, 0);
  }
  if ((trans != blas_trans) && (trans != blas_no_trans) &&
      (trans != blas_conj) && (trans != blas_conj_trans)) {
    BLAS_error(routine_name, -2, uplo, 0);
  }
  if (diag != blas_non_unit_diag && diag != blas_unit_diag) {
    BLAS_error(routine_name, -4, diag, 0);
  }
  if (n < 0) {
    BLAS_error(routine_name, -5, n, 0);
  }
  if (k >= n) {
    BLAS_error(routine_name, -6, k, 0);
  }
  if ((ldt < 1) || (ldt <= k)) {
    BLAS_error(routine_name, -9, ldt, 0);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -11, incx, 0);
  }

  if (n <= 0)
    return;

  incxi = incx;
  incxi *= 2;

  /* configuring the vector starting idx */
  if (incxi < 0) {
    start_xi = (1 - n) * incxi;
  } else {
    start_xi = 0;
  }

  /* if alpha is zero, then return x as a zero vector */
  if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
    xi = start_xi;
    for (i = 0; i < n; i++) {
      x_i[xi] = 0.0;
      x_i[xi + 1] = 0.0;
      xi += incxi;
    }
    return;
  }
  /* check to see if k=0.  if so, we can optimize somewhat */
  if (k == 0) {
    if (((alpha_i[0] == 1.0 && alpha_i[1] == 0.0))
        && (diag == blas_unit_diag)) {
      /* nothing to do */
      return;
    } else {
      /* just run the loops as is. */
      /* must set prec to output. Ignore user input of prec */
      prec = blas_prec_double;
    }
  }

  /* get index variables prepared */
  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (order == blas_rowmajor)) {
    dot_start = k;
  } else {
    dot_start = 0;
  }

  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (order == blas_rowmajor)) {
    dot_inc = 1;
    dot_start_inc1 = ldt - 1;
    dot_start_inc2 = ldt;
  } else {
    dot_inc = ldt - 1;
    dot_start_inc1 = 1;
    dot_start_inc2 = ldt;
  }

  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (uplo == blas_lower)) {
    /*start at the first element of x */
    /* substitution will proceed forwards (forwardsubstitution) */
  } else {
    /*start at the last element of x */
    /* substitution will proceed backwards (backsubstitution) */
    dot_inc = -dot_inc;
    dot_start_inc1 = -dot_start_inc1;
    dot_start_inc2 = -dot_start_inc2;
    dot_start = ldt * (n - 1) + k - dot_start;
    /*order of the following 2 statements matters! */
    start_xi = start_xi + (n - 1) * incxi;
    incxi = -incxi;
  }







  switch (prec) {

  case blas_prec_single:
  case blas_prec_indigenous:
  case blas_prec_double:{
      {

        {
          double temp1[2];        /* temporary variable for calculations */
          double temp2[2];        /* temporary variable for calculations */
          double x_elem[2];
          double T_element;






          /*loop 1 */
          xi = start_xi;
          for (j = 0; j < k; j++) {

            /* each time through loop, xi lands on next x to compute. */
            x_elem[0] = x_i[xi];
            x_elem[1] = x_i[xi + 1];
            /* preform the multiplication -
               in this implementation we do not seperate the alpha = 1 case */
            {
              temp1[0] =
                (double) x_elem[0] * alpha_i[0] -
                (double) x_elem[1] * alpha_i[1];
              temp1[1] =
                (double) x_elem[0] * alpha_i[1] +
                (double) x_elem[1] * alpha_i[0];
            }

            xi = start_xi;

            Tij = dot_start;
            dot_start += dot_start_inc1;

            for (i = j; i > 0; i--) {
              T_element = t_i[Tij];

              x_elem[0] = x_i[xi];
              x_elem[1] = x_i[xi + 1];
              {
                temp2[0] = x_elem[0] * T_element;
                temp2[1] = x_elem[1] * T_element;
              }
              temp1[0] = temp1[0] - temp2[0];
              temp1[1] = temp1[1] - temp2[1];
              xi += incxi;
              Tij += dot_inc;
            }                        /* for across row */


            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[Tij];


              temp1[0] = temp1[0] / T_element;
              temp1[1] = temp1[1] / T_element;

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[xi] = temp1[0];
            x_i[xi + 1] = temp1[1];
            xi += incxi;
          }                        /* for j<k */
          /*end loop 1 */

          /*loop 2 continue without changing j to start */
          for (; j < n; j++) {

            /* each time through loop, xi lands on next x to compute. */
            x_elem[0] = x_i[xi];
            x_elem[1] = x_i[xi + 1];
            {
              temp1[0] =
                (double) x_elem[0] * alpha_i[0] -
                (double) x_elem[1] * alpha_i[1];
              temp1[1] =
                (double) x_elem[0] * alpha_i[1] +
                (double) x_elem[1] * alpha_i[0];
            }

            xi = start_xi;
            start_xi += incxi;

            Tij = dot_start;
            dot_start += dot_start_inc2;

            for (i = k; i > 0; i--) {
              T_element = t_i[Tij];

              x_elem[0] = x_i[xi];
              x_elem[1] = x_i[xi + 1];
              {
                temp2[0] = x_elem[0] * T_element;
                temp2[1] = x_elem[1] * T_element;
              }
              temp1[0] = temp1[0] - temp2[0];
              temp1[1] = temp1[1] - temp2[1];
              xi += incxi;
              Tij += dot_inc;
            }                        /* for across row */


            /* if the diagonal entry is not equal to one, then divide by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[Tij];


              temp1[0] = temp1[0] / T_element;
              temp1[1] = temp1[1] / T_element;

            }
            /* if (diag == blas_non_unit_diag) */
            x_i[xi] = temp1[0];
            x_i[xi + 1] = temp1[1];
            xi += incxi;
          }                        /* for j<n */


        }
      }
      break;
    }

  case blas_prec_extra:{
      {

        {
          double head_temp1[2], tail_temp1[2];        /* temporary variable for calculations */
          double head_temp2[2], tail_temp2[2];        /* temporary variable for calculations */
          double head_temp3[2], tail_temp3[2];        /* temporary variable for calculations */
          double x_elem[2];
          double T_element;        /* temporary variable for an element of matrix T */

          int x_inti = 0, inc_x_inti = 1;
          int k_compare = k;        /*used for comparisons with x_inti */
          double *head_x_internal, *tail_x_internal;

          FPU_FIX_DECL;

          k_compare *= 2;
          inc_x_inti *= 2;
          head_x_internal = (double *) blas_malloc(k * sizeof(double) * 2);
          tail_x_internal = (double *) blas_malloc(k * sizeof(double) * 2);
          if (k > 0 && (head_x_internal == NULL || tail_x_internal == NULL)) {
            BLAS_error("blas_malloc", 0, 0, "malloc failed.\n");
          }


          FPU_FIX_START;

          /*loop 1 */
          xi = start_xi;
          /* x_inti already initialized to 0 */
          for (j = 0; j < k; j++) {

            /* each time through loop, xi lands on next x to compute. */
            x_elem[0] = x_i[xi];
            x_elem[1] = x_i[xi + 1];
            /* preform the multiplication -
               in this implementation we do not seperate the alpha = 1 case */
            {
              /* Compute complex-extra = complex-double * complex-double. */
              double head_t1, tail_t1;
              double head_t2, tail_t2;
              /* Real part */
              compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[0], alpha_i[0]);
              compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[1], alpha_i[1]);
              head_t2 = -head_t2;
              tail_t2 = -tail_t2;
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
              head_temp1[0] = head_t1;
              tail_temp1[0] = tail_t1;
              /* Imaginary part */
              compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[1], alpha_i[0]);
              compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[0], alpha_i[1]);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
              head_temp1[1] = head_t1;
              tail_temp1[1] = tail_t1;
            }

            Tij = dot_start;
            dot_start += dot_start_inc1;

            /*start loop buffer over in loop 1 */
            x_inti = 0;
            for (i = j; i > 0; i--) {
              T_element = t_i[Tij];

              head_temp3[0] = head_x_internal[x_inti];
              head_temp3[1] = head_x_internal[1 + x_inti];
              tail_temp3[0] = tail_x_internal[x_inti];
              tail_temp3[1] = tail_x_internal[1 + x_inti];
              {
                /* Compute complex-extra = complex-extra * real. */
                double head_a0, tail_a0;
                double head_a1, tail_a1;
                double head_t, tail_t;
                head_a0 = head_temp3[0];
                tail_a0 = tail_temp3[0];
                head_a1 = head_temp3[1];
                tail_a1 = tail_temp3[1];
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, T_element);
                head_temp2[0] = head_t;
                tail_temp2[0] = tail_t;
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, T_element);
                head_temp2[1] = head_t;
                tail_temp2[1] = tail_t;
              }

              {
                double head_at, tail_at;
                double head_bt, tail_bt;
                double head_ct, tail_ct;

                /* Real part */
                head_at = head_temp1[0];
                tail_at = tail_temp1[0];
                head_bt = -head_temp2[0];
                tail_bt = -tail_temp2[0];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_ct, &tail_ct, head_at, tail_at, head_bt, tail_bt);
                head_temp1[0] = head_ct;
                tail_temp1[0] = tail_ct;
                /* Imaginary part */
                head_at = head_temp1[1];
                tail_at = tail_temp1[1];
                head_bt = -head_temp2[1];
                tail_bt = -tail_temp2[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_ct, &tail_ct, head_at, tail_at, head_bt, tail_bt);
                head_temp1[1] = head_ct;
                tail_temp1[1] = tail_ct;
              }
              x_inti += inc_x_inti;
              Tij += dot_inc;
            }                        /* for across row */


            /* if the diagonal entry is not equal to one, then divide Xj by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[Tij];


              {
                double head_a, tail_a;
                double head_b, tail_b;
                head_a = head_temp1[0];
                tail_a = tail_temp1[0];
                compute_doubledouble_eq_doubledouble_div_double(&head_b, &tail_b, head_a, tail_a, T_element);
                head_temp1[0] = head_b;
                tail_temp1[0] = tail_b;
                head_a = head_temp1[1];
                tail_a = tail_temp1[1];
                compute_doubledouble_eq_doubledouble_div_double(&head_b, &tail_b, head_a, tail_a, T_element);
                head_temp1[1] = head_b;
                tail_temp1[1] = tail_b;
              }

            }

            /* if (diag == blas_non_unit_diag) */
            /* place internal precision result in internal buffer */
            head_x_internal[x_inti] = head_temp1[0];
            tail_x_internal[x_inti] = tail_temp1[0];
            head_x_internal[1 + x_inti] = head_temp1[1];
            tail_x_internal[1 + x_inti] = tail_temp1[1];

            /* place result x in same place as got x this loop */
            x_i[xi] = head_temp1[0];
            x_i[xi + 1] = head_temp1[1];
            xi += incxi;
          }                        /* for j<k */
          /*end loop 1 */


          /* loop2 ***************************** */
          x_inti = 0;
          /*loop 2 continue without changing j to start */
          for (; j < n; j++) {

            /* each time through loop, xi lands on next x to compute. */
            x_elem[0] = x_i[xi];
            x_elem[1] = x_i[xi + 1];
            {
              /* Compute complex-extra = complex-double * complex-double. */
              double head_t1, tail_t1;
              double head_t2, tail_t2;
              /* Real part */
              compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[0], alpha_i[0]);
              compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[1], alpha_i[1]);
              head_t2 = -head_t2;
              tail_t2 = -tail_t2;
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
              head_temp1[0] = head_t1;
              tail_temp1[0] = tail_t1;
              /* Imaginary part */
              compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, x_elem[1], alpha_i[0]);
              compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, x_elem[0], alpha_i[1]);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
              head_temp1[1] = head_t1;
              tail_temp1[1] = tail_t1;
            }


            Tij = dot_start;
            dot_start += dot_start_inc2;

            for (i = k; i > 0 && (x_inti < k_compare); i--) {
              T_element = t_i[Tij];

              head_temp3[0] = head_x_internal[x_inti];
              head_temp3[1] = head_x_internal[1 + x_inti];
              tail_temp3[0] = tail_x_internal[x_inti];
              tail_temp3[1] = tail_x_internal[1 + x_inti];
              {
                /* Compute complex-extra = complex-extra * real. */
                double head_a0, tail_a0;
                double head_a1, tail_a1;
                double head_t, tail_t;
                head_a0 = head_temp3[0];
                tail_a0 = tail_temp3[0];
                head_a1 = head_temp3[1];
                tail_a1 = tail_temp3[1];
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, T_element);
                head_temp2[0] = head_t;
                tail_temp2[0] = tail_t;
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, T_element);
                head_temp2[1] = head_t;
                tail_temp2[1] = tail_t;
              }

              {
                double head_at, tail_at;
                double head_bt, tail_bt;
                double head_ct, tail_ct;

                /* Real part */
                head_at = head_temp1[0];
                tail_at = tail_temp1[0];
                head_bt = -head_temp2[0];
                tail_bt = -tail_temp2[0];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_ct, &tail_ct, head_at, tail_at, head_bt, tail_bt);
                head_temp1[0] = head_ct;
                tail_temp1[0] = tail_ct;
                /* Imaginary part */
                head_at = head_temp1[1];
                tail_at = tail_temp1[1];
                head_bt = -head_temp2[1];
                tail_bt = -tail_temp2[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_ct, &tail_ct, head_at, tail_at, head_bt, tail_bt);
                head_temp1[1] = head_ct;
                tail_temp1[1] = tail_ct;
              }
              x_inti += inc_x_inti;
              Tij += dot_inc;
            }                        /* for across row */
            /*reset index to internal storage loop buffer. */
            x_inti = 0;
            for (; i > 0; i--) {
              T_element = t_i[Tij];

              head_temp3[0] = head_x_internal[x_inti];
              head_temp3[1] = head_x_internal[1 + x_inti];
              tail_temp3[0] = tail_x_internal[x_inti];
              tail_temp3[1] = tail_x_internal[1 + x_inti];
              {
                /* Compute complex-extra = complex-extra * real. */
                double head_a0, tail_a0;
                double head_a1, tail_a1;
                double head_t, tail_t;
                head_a0 = head_temp3[0];
                tail_a0 = tail_temp3[0];
                head_a1 = head_temp3[1];
                tail_a1 = tail_temp3[1];
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, T_element);
                head_temp2[0] = head_t;
                tail_temp2[0] = tail_t;
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, T_element);
                head_temp2[1] = head_t;
                tail_temp2[1] = tail_t;
              }

              {
                double head_at, tail_at;
                double head_bt, tail_bt;
                double head_ct, tail_ct;

                /* Real part */
                head_at = head_temp1[0];
                tail_at = tail_temp1[0];
                head_bt = -head_temp2[0];
                tail_bt = -tail_temp2[0];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_ct, &tail_ct, head_at, tail_at, head_bt, tail_bt);
                head_temp1[0] = head_ct;
                tail_temp1[0] = tail_ct;
                /* Imaginary part */
                head_at = head_temp1[1];
                tail_at = tail_temp1[1];
                head_bt = -head_temp2[1];
                tail_bt = -tail_temp2[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_ct, &tail_ct, head_at, tail_at, head_bt, tail_bt);
                head_temp1[1] = head_ct;
                tail_temp1[1] = tail_ct;
              }
              x_inti += inc_x_inti;
              Tij += dot_inc;
            }                        /* for across row */


            /* if the diagonal entry is not equal to one, then divide by
               the entry */
            if (diag == blas_non_unit_diag) {
              T_element = t_i[Tij];


              {
                double head_a, tail_a;
                double head_b, tail_b;
                head_a = head_temp1[0];
                tail_a = tail_temp1[0];
                compute_doubledouble_eq_doubledouble_div_double(&head_b, &tail_b, head_a, tail_a, T_element);
                head_temp1[0] = head_b;
                tail_temp1[0] = tail_b;
                head_a = head_temp1[1];
                tail_a = tail_temp1[1];
                compute_doubledouble_eq_doubledouble_div_double(&head_b, &tail_b, head_a, tail_a, T_element);
                head_temp1[1] = head_b;
                tail_temp1[1] = tail_b;
              }

            }

            /* if (diag == blas_non_unit_diag) */
            /* place internal precision result in internal buffer */
            head_x_internal[x_inti] = head_temp1[0];
            tail_x_internal[x_inti] = tail_temp1[0];
            head_x_internal[1 + x_inti] = head_temp1[1];
            tail_x_internal[1 + x_inti] = tail_temp1[1];
            x_inti += inc_x_inti;
            if (x_inti >= k_compare)
              x_inti = 0;

            /* place result x in same place as got x this loop */
            x_i[xi] = head_temp1[0];
            x_i[xi + 1] = head_temp1[1];
            xi += incxi;
          }                        /* for j<n */

          FPU_FIX_STOP;

          blas_free(head_x_internal);
          blas_free(tail_x_internal);
        }
      }
      break;
    }

  default:
    BLAS_error(routine_name, -13, prec, 0);
    break;
  }                                /* end prec switch */
}                                /* end BLAS_ztbsv_d_x */
