#include <math.h>
#include "blas_extended.h"
#include "blas_extended_private.h"
#include "trsv/XBLAS_trsv.hpp"


void BLAS_ctrsv_s_x(enum blas_order_type order,
                    enum blas_uplo_type uplo,
                    enum blas_trans_type trans,
                    enum blas_diag_type diag,
                    int n,
                    const void *alpha,
                    const float *T,
                    int ldt,
                    void *x,
                    int incx,
                    enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine solve :
 *
 *     x <- alpha * inverse(T) * x
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
 * T      (input) float*
 *        Triangular matrix
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
  char *routine_name = "BLAS_ctrsv_s";

  int i, j;                        /* used to idx matrix */
  int ix, jx;                        /* used to idx vector x */
  int start_x;                        /* used as the starting idx to vector x */
  const float *T_i = T;                /* internal matrix T */
  float *x_i = (float *) x;        /* internal x */
  float *alpha_i = (float *) alpha;        /* internal alpha */
  float T_element;                /* temporary variable for an element of matrix A */
  int incT = 1;                        /* internal ldt */

  if ((order != blas_rowmajor && order != blas_colmajor) ||
      (uplo != blas_upper && uplo != blas_lower) ||
      (trans != blas_trans && trans !=
       blas_no_trans && trans != blas_conj_trans) ||
      (diag != blas_non_unit_diag && diag != blas_unit_diag) ||
      (ldt < n) || (incx == 0)) {
    BLAS_error(routine_name, 0, 0, NULL);
  }

  if (n <= 0)
    return;


  incx *= 2;
  /* configuring the vector starting idx */
  if (incx <= 0) {
    start_x = -(n - 1) * incx;
  } else {
    start_x = 0;
  }

  /* if alpha is zero, then return x as a zero vector */
  if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
    ix = start_x;
    for (i = 0; i < n; i++) {
      x_i[ix] = 0.0;
      x_i[ix + 1] = 0.0;
      ix += incx;
    }
    return;
  }
  switch (prec) {
  case blas_prec_single:

    {
      float temp1[2];                /* temporary variable for calculations */
      float temp2[2];                /* temporary variable for calculations */
      float temp3[2];                /* temporary variable for calculations */

      if ((order == blas_rowmajor &&
           trans == blas_no_trans && uplo == blas_upper) ||
          (order == blas_colmajor &&
           trans != blas_no_trans && uplo == blas_lower)) {

        jx = start_x + (n - 1) * incx;
        for (j = n - 1; j >= 0; j--) {

          /* compute Xj = alpha*Xj - SUM Tij(or Tji) * Xi
             i=j+1 to n-1           */
          temp3[0] = x_i[jx];
          temp3[1] = x_i[1 + jx];
          {
            temp1[0] = temp3[0] * alpha_i[0] - temp3[1] * alpha_i[1];
            temp1[1] = temp3[0] * alpha_i[1] + temp3[1] * alpha_i[0];
          }


          ix = start_x + (n - 1) * incx;
          for (i = n - 1; i >= j + 1; i--) {
            T_element = T_i[i * incT + j * ldt * incT];

            temp3[0] = x_i[ix];
            temp3[1] = x_i[1 + ix];
            {
              temp2[0] = temp3[0] * T_element;
              temp2[1] = temp3[1] * T_element;
            }
            temp1[0] = temp1[0] - temp2[0];
            temp1[1] = temp1[1] - temp2[1];
            ix -= incx;
          }                        /* for j<n */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = T_i[j * incT + j * ldt * incT];


            temp1[0] = temp1[0] / T_element;
            temp1[1] = temp1[1] / T_element;

          }
          /* if (diag == blas_non_unit_diag) */
          x_i[jx] = temp1[0];
          x_i[jx + 1] = temp1[1];

          jx -= incx;
        }                        /* for j>=0 */
      } else if ((order == blas_rowmajor &&
                  trans == blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans != blas_no_trans && uplo == blas_upper)) {

        jx = start_x;
        for (j = 0; j < n; j++) {

          /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3[0] = x_i[jx];
          temp3[1] = x_i[1 + jx];
          /* multiply by alpha */
          {
            temp1[0] = temp3[0] * alpha_i[0] - temp3[1] * alpha_i[1];
            temp1[1] = temp3[0] * alpha_i[1] + temp3[1] * alpha_i[0];
          }


          ix = start_x;
          for (i = 0; i < j; i++) {
            T_element = T_i[i * incT + j * ldt * incT];

            temp3[0] = x_i[ix];
            temp3[1] = x_i[1 + ix];
            {
              temp2[0] = temp3[0] * T_element;
              temp2[1] = temp3[1] * T_element;
            }
            temp1[0] = temp1[0] - temp2[0];
            temp1[1] = temp1[1] - temp2[1];
            ix += incx;
          }                        /* for i<j */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = T_i[j * incT + j * ldt * incT];


            temp1[0] = temp1[0] / T_element;
            temp1[1] = temp1[1] / T_element;

          }
          /* if (diag == blas_non_unit_diag) */
          x_i[jx] = temp1[0];
          x_i[jx + 1] = temp1[1];
          jx += incx;
        }                        /* for j<n */
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_upper)) {

        jx = start_x + (n - 1) * incx;
        for (j = n - 1; j >= 0; j--) {

          /* compute Xj = alpha*Xj - SUM Tij(or Tji) * Xi
             i=j+1 to n-1           */
          temp3[0] = x_i[jx];
          temp3[1] = x_i[1 + jx];
          {
            temp1[0] = temp3[0] * alpha_i[0] - temp3[1] * alpha_i[1];
            temp1[1] = temp3[0] * alpha_i[1] + temp3[1] * alpha_i[0];
          }


          ix = start_x + (n - 1) * incx;
          for (i = n - 1; i >= j + 1; i--) {
            T_element = T_i[j * incT + i * ldt * incT];

            temp3[0] = x_i[ix];
            temp3[1] = x_i[1 + ix];
            {
              temp2[0] = temp3[0] * T_element;
              temp2[1] = temp3[1] * T_element;
            }
            temp1[0] = temp1[0] - temp2[0];
            temp1[1] = temp1[1] - temp2[1];
            ix -= incx;
          }                        /* for j<n */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = T_i[j * incT + j * ldt * incT];


            temp1[0] = temp1[0] / T_element;
            temp1[1] = temp1[1] / T_element;

          }
          /* if (diag == blas_non_unit_diag) */
          x_i[jx] = temp1[0];
          x_i[jx + 1] = temp1[1];

          jx -= incx;
        }                        /* for j>=0 */
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_upper) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_lower)) {

        jx = start_x;
        for (j = 0; j < n; j++) {

          /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3[0] = x_i[jx];
          temp3[1] = x_i[1 + jx];
          /* multiply by alpha */
          {
            temp1[0] = temp3[0] * alpha_i[0] - temp3[1] * alpha_i[1];
            temp1[1] = temp3[0] * alpha_i[1] + temp3[1] * alpha_i[0];
          }


          ix = start_x;
          for (i = 0; i < j; i++) {
            T_element = T_i[j * incT + i * ldt * incT];

            temp3[0] = x_i[ix];
            temp3[1] = x_i[1 + ix];
            {
              temp2[0] = temp3[0] * T_element;
              temp2[1] = temp3[1] * T_element;
            }
            temp1[0] = temp1[0] - temp2[0];
            temp1[1] = temp1[1] - temp2[1];
            ix += incx;
          }                        /* for i<j */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = T_i[j * incT + j * ldt * incT];


            temp1[0] = temp1[0] / T_element;
            temp1[1] = temp1[1] / T_element;

          }
          /* if (diag == blas_non_unit_diag) */
          x_i[jx] = temp1[0];
          x_i[jx + 1] = temp1[1];
          jx += incx;
        }                        /* for j<n */
      }
    }
    break;
  case blas_prec_double:
  case blas_prec_indigenous:

    {
      int inc_intx;                /* inc for intx */
      double temp1[2];                /* temporary variable for calculations */
      double temp2[2];                /* temporary variable for calculations */
      double temp3[2];                /* temporary variable for calculations */
      double *intx;                /* copy of x used for calculations */

      /* allocate space for intx */
      intx = (double *) blas_malloc(n * sizeof(double) * 2);
      if (n > 0 && intx == NULL) {
        BLAS_error("blas_malloc", 0, 0, "malloc failed.\n");
      }

      /* since intx is for internal usage, set it to 1 and then adjust
         it if necessary */
      inc_intx = 1;
      inc_intx *= 2;

      /* copy x to intx */
      ix = start_x;
      jx = 0;
      for (i = 0; i < n; i++) {
        temp1[0] = x_i[ix];
        temp1[1] = x_i[1 + ix];
        intx[jx] = temp1[0];
        intx[1 + jx] = temp1[1];
        ix += incx;
        jx += inc_intx;
      }

      if ((order == blas_rowmajor &&
           trans == blas_no_trans && uplo == blas_upper) ||
          (order == blas_colmajor &&
           trans != blas_no_trans && uplo == blas_lower)) {

        jx = (n - 1) * inc_intx;
        for (j = n - 1; j >= 0; j--) {

          /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3[0] = intx[jx];
          temp3[1] = intx[1 + jx];
          /* multiply by alpha */
          {
            temp1[0] =
              (double) temp3[0] * alpha_i[0] - (double) temp3[1] * alpha_i[1];
            temp1[1] =
              (double) temp3[0] * alpha_i[1] + (double) temp3[1] * alpha_i[0];
          }

          ix = (n - 1) * inc_intx;
          for (i = n - 1; i >= j + 1; i--) {
            T_element = T_i[i * incT + j * ldt * incT];

            temp3[0] = intx[ix];
            temp3[1] = intx[1 + ix];
            {
              temp2[0] = temp3[0] * T_element;
              temp2[1] = temp3[1] * T_element;
            }
            temp1[0] = temp1[0] - temp2[0];
            temp1[1] = temp1[1] - temp2[1];
            ix -= inc_intx;
          }                        /* for j<n */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = T_i[j * incT + j * ldt * incT];


            {
              double dt = T_element;
              temp1[0] = temp1[0] / dt;
              temp1[1] = temp1[1] / dt;
            }

          }
          /* if (diag == blas_non_unit_diag) */
          intx[jx] = temp1[0];
          intx[1 + jx] = temp1[1];

          jx -= inc_intx;
        }                        /* for j>=0 */
      } else if ((order == blas_rowmajor &&
                  trans == blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans != blas_no_trans && uplo == blas_upper)) {

        jx = 0;
        for (j = 0; j < n; j++) {

          /* compute Xj = Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3[0] = intx[jx];
          temp3[1] = intx[1 + jx];
          /* multiply by alpha */
          {
            temp1[0] =
              (double) temp3[0] * alpha_i[0] - (double) temp3[1] * alpha_i[1];
            temp1[1] =
              (double) temp3[0] * alpha_i[1] + (double) temp3[1] * alpha_i[0];
          }

          ix = 0;
          for (i = 0; i < j; i++) {
            T_element = T_i[i * incT + j * ldt * incT];

            temp3[0] = intx[ix];
            temp3[1] = intx[1 + ix];
            {
              temp2[0] = temp3[0] * T_element;
              temp2[1] = temp3[1] * T_element;
            }
            temp1[0] = temp1[0] - temp2[0];
            temp1[1] = temp1[1] - temp2[1];
            ix += inc_intx;
          }                        /* for i<j */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = T_i[j * incT + j * ldt * incT];


            {
              double dt = T_element;
              temp1[0] = temp1[0] / dt;
              temp1[1] = temp1[1] / dt;
            }

          }
          /* if (diag == blas_non_unit_diag) */
          intx[jx] = temp1[0];
          intx[1 + jx] = temp1[1];
          jx += inc_intx;
        }                        /* for j<n */
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_upper)) {

        jx = (n - 1) * inc_intx;
        for (j = n - 1; j >= 0; j--) {

          /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3[0] = intx[jx];
          temp3[1] = intx[1 + jx];
          /* multiply by alpha */
          {
            temp1[0] =
              (double) temp3[0] * alpha_i[0] - (double) temp3[1] * alpha_i[1];
            temp1[1] =
              (double) temp3[0] * alpha_i[1] + (double) temp3[1] * alpha_i[0];
          }

          ix = (n - 1) * inc_intx;
          for (i = n - 1; i >= j + 1; i--) {
            T_element = T_i[j * incT + i * ldt * incT];

            temp3[0] = intx[ix];
            temp3[1] = intx[1 + ix];
            {
              temp2[0] = temp3[0] * T_element;
              temp2[1] = temp3[1] * T_element;
            }
            temp1[0] = temp1[0] - temp2[0];
            temp1[1] = temp1[1] - temp2[1];
            ix -= inc_intx;
          }                        /* for j<n */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = T_i[j * incT + j * ldt * incT];


            {
              double dt = T_element;
              temp1[0] = temp1[0] / dt;
              temp1[1] = temp1[1] / dt;
            }

          }
          /* if (diag == blas_non_unit_diag) */
          intx[jx] = temp1[0];
          intx[1 + jx] = temp1[1];

          jx -= inc_intx;
        }                        /* for j>=0 */
      } else if ((order == blas_rowmajor &&
                  trans != blas_no_trans && uplo == blas_upper) ||
                 (order == blas_colmajor &&
                  trans == blas_no_trans && uplo == blas_lower)) {

        jx = 0;
        for (j = 0; j < n; j++) {

          /* compute Xj = Xj - SUM Aij(or Aji) * Xi
             i=j+1 to n-1           */
          temp3[0] = intx[jx];
          temp3[1] = intx[1 + jx];
          /* multiply by alpha */
          {
            temp1[0] =
              (double) temp3[0] * alpha_i[0] - (double) temp3[1] * alpha_i[1];
            temp1[1] =
              (double) temp3[0] * alpha_i[1] + (double) temp3[1] * alpha_i[0];
          }

          ix = 0;
          for (i = 0; i < j; i++) {
            T_element = T_i[j * incT + i * ldt * incT];

            temp3[0] = intx[ix];
            temp3[1] = intx[1 + ix];
            {
              temp2[0] = temp3[0] * T_element;
              temp2[1] = temp3[1] * T_element;
            }
            temp1[0] = temp1[0] - temp2[0];
            temp1[1] = temp1[1] - temp2[1];
            ix += inc_intx;
          }                        /* for i<j */

          /* if the diagonal entry is not equal to one, then divide Xj by
             the entry */
          if (diag == blas_non_unit_diag) {
            T_element = T_i[j * incT + j * ldt * incT];


            {
              double dt = T_element;
              temp1[0] = temp1[0] / dt;
              temp1[1] = temp1[1] / dt;
            }

          }
          /* if (diag == blas_non_unit_diag) */
          intx[jx] = temp1[0];
          intx[1 + jx] = temp1[1];
          jx += inc_intx;
        }                        /* for j<n */
      }

      /* copy the final results from intx to x */
      ix = start_x;
      jx = 0;
      for (i = 0; i < n; i++) {
        temp1[0] = intx[jx];
        temp1[1] = intx[1 + jx];
        x_i[ix] = temp1[0];
        x_i[ix + 1] = temp1[1];
        ix += incx;
        jx += inc_intx;
      }

      blas_free(intx);
    }
    break;
  case blas_prec_extra:
    {
      FPU_FIX_DECL;
      FPU_FIX_START;
      {
        {
          int inc_intx;                /* inc for intx */
          double head_temp1[2], tail_temp1[2];        /* temporary variable for calculations */
          double head_temp2[2], tail_temp2[2];        /* temporary variable for calculations */
          double head_temp3[2], tail_temp3[2];        /* temporary variable for calculations */
          double *head_intx, *tail_intx;
          /* copy of x used for calculations */

          /* allocate space for intx */
          head_intx = (double *) blas_malloc(n * sizeof(double) * 2);
          tail_intx = (double *) blas_malloc(n * sizeof(double) * 2);
          if (n > 0 && (head_intx == NULL || tail_intx == NULL)) {
            BLAS_error("blas_malloc", 0, 0, "malloc failed.\n");
          }

          /* since intx is for internal usage, set it to 1 and then adjust
             it if necessary */
          inc_intx = 1;
          inc_intx *= 2;

          /* copy x to intx */
          ix = start_x;
          jx = 0;
          for (i = 0; i < n; i++) {
            head_temp1[0] = x_i[ix];
            tail_temp1[0] = 0.0;
            head_temp1[1] = x_i[1 + ix];
            tail_temp1[1] = 0.0;
            head_intx[jx] = head_temp1[0];
            tail_intx[jx] = tail_temp1[0];
            head_intx[1 + jx] = head_temp1[1];
            tail_intx[1 + jx] = tail_temp1[1];
            ix += incx;
            jx += inc_intx;
          }

          if ((order == blas_rowmajor &&
               trans == blas_no_trans && uplo == blas_upper) ||
              (order == blas_colmajor &&
               trans != blas_no_trans && uplo == blas_lower)) {

            jx = (n - 1) * inc_intx;
            for (j = n - 1; j >= 0; j--) {

              /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
                 i=j+1 to n-1           */
              head_temp3[0] = head_intx[jx];
              head_temp3[1] = head_intx[1 + jx];
              tail_temp3[0] = tail_intx[jx];
              tail_temp3[1] = tail_intx[1 + jx];
              /* multiply by alpha */
              {
                double cd[2];
                cd[0] = (double) alpha_i[0];
                cd[1] = (double) alpha_i[1];
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }

              }

              ix = (n - 1) * inc_intx;
              for (i = n - 1; i >= j + 1; i--) {
                T_element = T_i[i * incT + j * ldt * incT];

                head_temp3[0] = head_intx[ix];
                head_temp3[1] = head_intx[1 + ix];
                tail_temp3[0] = tail_intx[ix];
                tail_temp3[1] = tail_intx[1 + ix];
                {
                  double dt = (double) T_element;
                  {
                    /* Compute complex-extra = complex-extra * real. */
                    double head_a0, tail_a0;
                    double head_a1, tail_a1;
                    double head_t, tail_t;
                    head_a0 = head_temp3[0];
                    tail_a0 = tail_temp3[0];
                    head_a1 = head_temp3[1];
                    tail_a1 = tail_temp3[1];
                    compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, dt);
                    head_temp2[0] = head_t;
                    tail_temp2[0] = tail_t;
                    compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, dt);
                    head_temp2[1] = head_t;
                    tail_temp2[1] = tail_t;
                  }

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
                ix -= inc_intx;
              }                        /* for j<n */

              /* if the diagonal entry is not equal to one, then divide Xj by
                 the entry */
              if (diag == blas_non_unit_diag) {
                T_element = T_i[j * incT + j * ldt * incT];


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
              head_intx[jx] = head_temp1[0];
              tail_intx[jx] = tail_temp1[0];
              head_intx[1 + jx] = head_temp1[1];
              tail_intx[1 + jx] = tail_temp1[1];

              jx -= inc_intx;
            }                        /* for j>=0 */
          } else if ((order == blas_rowmajor &&
                      trans == blas_no_trans && uplo == blas_lower) ||
                     (order == blas_colmajor &&
                      trans != blas_no_trans && uplo == blas_upper)) {

            jx = 0;
            for (j = 0; j < n; j++) {

              /* compute Xj = Xj - SUM Aij(or Aji) * Xi
                 i=j+1 to n-1           */
              head_temp3[0] = head_intx[jx];
              head_temp3[1] = head_intx[1 + jx];
              tail_temp3[0] = tail_intx[jx];
              tail_temp3[1] = tail_intx[1 + jx];
              /* multiply by alpha */
              {
                double cd[2];
                cd[0] = (double) alpha_i[0];
                cd[1] = (double) alpha_i[1];
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }

              }

              ix = 0;
              for (i = 0; i < j; i++) {
                T_element = T_i[i * incT + j * ldt * incT];

                head_temp3[0] = head_intx[ix];
                head_temp3[1] = head_intx[1 + ix];
                tail_temp3[0] = tail_intx[ix];
                tail_temp3[1] = tail_intx[1 + ix];
                {
                  double dt = (double) T_element;
                  {
                    /* Compute complex-extra = complex-extra * real. */
                    double head_a0, tail_a0;
                    double head_a1, tail_a1;
                    double head_t, tail_t;
                    head_a0 = head_temp3[0];
                    tail_a0 = tail_temp3[0];
                    head_a1 = head_temp3[1];
                    tail_a1 = tail_temp3[1];
                    compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, dt);
                    head_temp2[0] = head_t;
                    tail_temp2[0] = tail_t;
                    compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, dt);
                    head_temp2[1] = head_t;
                    tail_temp2[1] = tail_t;
                  }

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
                ix += inc_intx;
              }                        /* for i<j */

              /* if the diagonal entry is not equal to one, then divide Xj by
                 the entry */
              if (diag == blas_non_unit_diag) {
                T_element = T_i[j * incT + j * ldt * incT];


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
              head_intx[jx] = head_temp1[0];
              tail_intx[jx] = tail_temp1[0];
              head_intx[1 + jx] = head_temp1[1];
              tail_intx[1 + jx] = tail_temp1[1];
              jx += inc_intx;
            }                        /* for j<n */
          } else if ((order == blas_rowmajor &&
                      trans != blas_no_trans && uplo == blas_lower) ||
                     (order == blas_colmajor &&
                      trans == blas_no_trans && uplo == blas_upper)) {

            jx = (n - 1) * inc_intx;
            for (j = n - 1; j >= 0; j--) {

              /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
                 i=j+1 to n-1           */
              head_temp3[0] = head_intx[jx];
              head_temp3[1] = head_intx[1 + jx];
              tail_temp3[0] = tail_intx[jx];
              tail_temp3[1] = tail_intx[1 + jx];
              /* multiply by alpha */
              {
                double cd[2];
                cd[0] = (double) alpha_i[0];
                cd[1] = (double) alpha_i[1];
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }

              }

              ix = (n - 1) * inc_intx;
              for (i = n - 1; i >= j + 1; i--) {
                T_element = T_i[j * incT + i * ldt * incT];

                head_temp3[0] = head_intx[ix];
                head_temp3[1] = head_intx[1 + ix];
                tail_temp3[0] = tail_intx[ix];
                tail_temp3[1] = tail_intx[1 + ix];
                {
                  double dt = (double) T_element;
                  {
                    /* Compute complex-extra = complex-extra * real. */
                    double head_a0, tail_a0;
                    double head_a1, tail_a1;
                    double head_t, tail_t;
                    head_a0 = head_temp3[0];
                    tail_a0 = tail_temp3[0];
                    head_a1 = head_temp3[1];
                    tail_a1 = tail_temp3[1];
                    compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, dt);
                    head_temp2[0] = head_t;
                    tail_temp2[0] = tail_t;
                    compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, dt);
                    head_temp2[1] = head_t;
                    tail_temp2[1] = tail_t;
                  }

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
                ix -= inc_intx;
              }                        /* for j<n */

              /* if the diagonal entry is not equal to one, then divide Xj by
                 the entry */
              if (diag == blas_non_unit_diag) {
                T_element = T_i[j * incT + j * ldt * incT];


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
              head_intx[jx] = head_temp1[0];
              tail_intx[jx] = tail_temp1[0];
              head_intx[1 + jx] = head_temp1[1];
              tail_intx[1 + jx] = tail_temp1[1];

              jx -= inc_intx;
            }                        /* for j>=0 */
          } else if ((order == blas_rowmajor &&
                      trans != blas_no_trans && uplo == blas_upper) ||
                     (order == blas_colmajor &&
                      trans == blas_no_trans && uplo == blas_lower)) {

            jx = 0;
            for (j = 0; j < n; j++) {

              /* compute Xj = Xj - SUM Aij(or Aji) * Xi
                 i=j+1 to n-1           */
              head_temp3[0] = head_intx[jx];
              head_temp3[1] = head_intx[1 + jx];
              tail_temp3[0] = tail_intx[jx];
              tail_temp3[1] = tail_intx[1 + jx];
              /* multiply by alpha */
              {
                double cd[2];
                cd[0] = (double) alpha_i[0];
                cd[1] = (double) alpha_i[1];
                {
                  /* Compute complex-extra = complex-extra * complex-double. */
                  double head_a0, tail_a0;
                  double head_a1, tail_a1;
                  double head_t1, tail_t1;
                  double head_t2, tail_t2;
                  head_a0 = head_temp3[0];
                  tail_a0 = tail_temp3[0];
                  head_a1 = head_temp3[1];
                  tail_a1 = tail_temp3[1];
                  /* real part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, cd[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, cd[1]);
                  head_t2 = -head_t2;
                  tail_t2 = -tail_t2;
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[0] = head_t1;
                  tail_temp1[0] = tail_t1;
                  /* imaginary part */
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, cd[0]);
                  compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, cd[1]);
                  compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                  head_temp1[1] = head_t1;
                  tail_temp1[1] = tail_t1;
                }

              }

              ix = 0;
              for (i = 0; i < j; i++) {
                T_element = T_i[j * incT + i * ldt * incT];

                head_temp3[0] = head_intx[ix];
                head_temp3[1] = head_intx[1 + ix];
                tail_temp3[0] = tail_intx[ix];
                tail_temp3[1] = tail_intx[1 + ix];
                {
                  double dt = (double) T_element;
                  {
                    /* Compute complex-extra = complex-extra * real. */
                    double head_a0, tail_a0;
                    double head_a1, tail_a1;
                    double head_t, tail_t;
                    head_a0 = head_temp3[0];
                    tail_a0 = tail_temp3[0];
                    head_a1 = head_temp3[1];
                    tail_a1 = tail_temp3[1];
                    compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a0, tail_a0, dt);
                    head_temp2[0] = head_t;
                    tail_temp2[0] = tail_t;
                    compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_a1, tail_a1, dt);
                    head_temp2[1] = head_t;
                    tail_temp2[1] = tail_t;
                  }

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
                ix += inc_intx;
              }                        /* for i<j */

              /* if the diagonal entry is not equal to one, then divide Xj by
                 the entry */
              if (diag == blas_non_unit_diag) {
                T_element = T_i[j * incT + j * ldt * incT];


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
              head_intx[jx] = head_temp1[0];
              tail_intx[jx] = tail_temp1[0];
              head_intx[1 + jx] = head_temp1[1];
              tail_intx[1 + jx] = tail_temp1[1];
              jx += inc_intx;
            }                        /* for j<n */
          }

          /* copy the final results from intx to x */
          ix = start_x;
          jx = 0;
          for (i = 0; i < n; i++) {
            head_temp1[0] = head_intx[jx];
            head_temp1[1] = head_intx[1 + jx];
            tail_temp1[0] = tail_intx[jx];
            tail_temp1[1] = tail_intx[1 + jx];
            x_i[ix] = head_temp1[0];
            x_i[ix + 1] = head_temp1[1];
            ix += incx;
            jx += inc_intx;
          }

          blas_free(head_intx);
          blas_free(tail_intx);
        }
      }
      FPU_FIX_STOP;
    }
    break;
  }
}
