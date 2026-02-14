#include "blas_extended.h"
#include "blas_extended_private.h"
#include "gemv2/XBLAS_gemv2.hpp"

void BLAS_zgemv2_d_d_x(enum blas_order_type order,
                       enum blas_trans_type trans,
                       int m,
                       int n,
                       const void *alpha,
                       const double *a,
                       int lda,
                       const double *head_x,
                       const double *tail_x,
                       int incx,
                       const void *beta,
                       void *y,
                       int incy,
                       enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * op(A) * head_x + alpha * op(A) * tail_x + beta * y,
 * where A is a general matrix.
 *
 * Arguments
 * =========
 *
 * order        (input) enum blas_order_type
 *              Order of A; row or column major
 *
 * trans        (input) enum blas_trans_type
 *              Transpose of A: no trans, trans, or conjugate trans
 *
 * m            (input) int
 *              Dimension of A
 *
 * n            (input) int
 *              Dimension of A and the length of vector x and z
 *
 * alpha        (input) const void*
 *
 * A            (input) const double*
 *
 * lda          (input) int
 *              Leading dimension of A
 *
 * head_x
 * tail_x       (input) const double*
 *
 * incx         (input) int
 *              The stride for vector x.
 *
 * beta         (input) const void*
 *
 * y            (input/output) void*
 *
 * incy         (input) int
 *              The stride for vector y.
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
//static const char routine_name[] = "BLAS_zgemv2_d_d_x";
#if 0
  XBLAS::gemv2_x(order,
                 trans,
                 m,
                 n,
                 *static_cast<const std::complex<double> *>(alpha),
                 a,
                 lda,
                 head_x,
                 tail_x,
                 incx,
                 *static_cast<const std::complex<double> *>(beta),
                 static_cast<std::complex<double> *>(y),
                 incy,
                 prec);
} /* end BLAS_zgemv2_d_d_x */
#else
  static const char routine_name[] = "BLAS_zgemv2_d_d_x";
  switch (prec) {
  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, j;
      int iy, jx, kx, ky;
      int lenx, leny;
      int ai, aij;
      int incai, incaij;

      const double *a_i = a;
      const double *head_x_i = head_x;
      const double *tail_x_i = tail_x;
      double *y_i = (double *) y;
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;
      double a_elem;
      double x_elem;
      double y_elem[2];
      double prod;
      double sum;
      double sum2;
      double tmp1[2];
      double tmp2[2];


      /* all error calls */
      if (m < 0)
        BLAS_error(routine_name, -3, m, 0);
      else if (n <= 0)
        BLAS_error(routine_name, -4, n, 0);
      else if (incx == 0)
        BLAS_error(routine_name, -10, incx, 0);
      else if (incy == 0)
        BLAS_error(routine_name, -13, incy, 0);

      if ((order == blas_rowmajor) && (trans == blas_no_trans)) {
        lenx = n;
        leny = m;
        incai = lda;
        incaij = 1;
      } else if ((order == blas_rowmajor) && (trans != blas_no_trans)) {
        lenx = m;
        leny = n;
        incai = 1;
        incaij = lda;
      } else if ((order == blas_colmajor) && (trans == blas_no_trans)) {
        lenx = n;
        leny = m;
        incai = 1;
        incaij = lda;
      } else {                        /* colmajor and blas_trans */
        lenx = m;
        leny = n;
        incai = lda;
        incaij = 1;
      }

      if (lda < leny)
        BLAS_error(routine_name, -7, lda, NULL);




      incy *= 2;



      if (incx > 0)
        kx = 0;
      else
        kx = (1 - lenx) * incx;
      if (incy > 0)
        ky = 0;
      else
        ky = (1 - leny) * incy;

      /* No extra-precision needed for alpha = 0 */
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          iy = ky;
          for (i = 0; i < leny; i++) {
            y_i[iy] = 0.0;
            y_i[iy + 1] = 0.0;
            iy += incy;
          }
        } else if (!(beta_i[0] == 0.0 && beta_i[1] == 0.0)) {
          iy = ky;
          for (i = 0; i < leny; i++) {
            y_elem[0] = y_i[iy];
            y_elem[1] = y_i[iy + 1];
            {
              tmp1[0] =
                (double) y_elem[0] * beta_i[0] -
                (double) y_elem[1] * beta_i[1];
              tmp1[1] =
                (double) y_elem[0] * beta_i[1] +
                (double) y_elem[1] * beta_i[0];
            }
            y_i[iy] = tmp1[0];
            y_i[iy + 1] = tmp1[1];
            iy += incy;
          }
        }
      } else {                        /* alpha != 0 */

        /* if beta = 0, we can save m multiplies:
           y = alpha*A*head_x + alpha*A*tail_x  */
        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
            /* save m more multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = 0.0;
              sum2 = 0.0;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                prod = a_elem * x_elem;
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = a_elem * x_elem;
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              sum = sum + sum2;
              tmp1[0] = sum;
              tmp1[1] = 0.0;
              y_i[iy] = tmp1[0];
              y_i[iy + 1] = tmp1[1];
              ai += incai;
              iy += incy;
            }                        /* end for */
          } else {                /* alpha != 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = 0.0;
              sum2 = 0.0;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                prod = a_elem * x_elem;
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = a_elem * x_elem;
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              {
                tmp1[0] = alpha_i[0] * sum;
                tmp1[1] = alpha_i[1] * sum;
              }
              {
                tmp2[0] = alpha_i[0] * sum2;
                tmp2[1] = alpha_i[1] * sum2;
              }
              tmp1[0] = tmp1[0] + tmp2[0];
              tmp1[1] = tmp1[1] + tmp2[1];
              y_i[iy] = tmp1[0];
              y_i[iy + 1] = tmp1[1];
              ai += incai;
              iy += incy;
            }
          }
        } else {                /* beta != 0 */
          if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
            /* save m multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = 0.0;;
              sum2 = 0.0;;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                prod = a_elem * x_elem;
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = a_elem * x_elem;
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              sum = sum + sum2;
              y_elem[0] = y_i[iy];
              y_elem[1] = y_i[iy + 1];
              {
                tmp1[0] =
                  (double) y_elem[0] * beta_i[0] -
                  (double) y_elem[1] * beta_i[1];
                tmp1[1] =
                  (double) y_elem[0] * beta_i[1] +
                  (double) y_elem[1] * beta_i[0];
              }
              tmp2[0] = sum;
              tmp2[1] = 0.0;
              tmp2[0] = tmp2[0] + tmp1[0];
              tmp2[1] = tmp2[1] + tmp1[1];
              y_i[iy] = tmp2[0];
              y_i[iy + 1] = tmp2[1];
              ai += incai;
              iy += incy;
            }
          } else {                /* alpha != 1, the most general form:
                                   y = alpha*A*head_x + alpha*A*tail_x + beta*y */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              sum = 0.0;;
              sum2 = 0.0;;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                prod = a_elem * x_elem;
                sum = sum + prod;
                x_elem = tail_x_i[jx];
                prod = a_elem * x_elem;
                sum2 = sum2 + prod;
                aij += incaij;
                jx += incx;
              }
              {
                tmp1[0] = alpha_i[0] * sum;
                tmp1[1] = alpha_i[1] * sum;
              }
              {
                tmp2[0] = alpha_i[0] * sum2;
                tmp2[1] = alpha_i[1] * sum2;
              }
              tmp1[0] = tmp1[0] + tmp2[0];
              tmp1[1] = tmp1[1] + tmp2[1];
              y_elem[0] = y_i[iy];
              y_elem[1] = y_i[iy + 1];
              {
                tmp2[0] =
                  (double) y_elem[0] * beta_i[0] -
                  (double) y_elem[1] * beta_i[1];
                tmp2[1] =
                  (double) y_elem[0] * beta_i[1] +
                  (double) y_elem[1] * beta_i[0];
              }
              tmp1[0] = tmp1[0] + tmp2[0];
              tmp1[1] = tmp1[1] + tmp2[1];
              y_i[iy] = tmp1[0];
              y_i[iy + 1] = tmp1[1];
              ai += incai;
              iy += incy;
            }
          }
        }

      }



      break;
    }
  case blas_prec_extra:{

      int i, j;
      int iy, jx, kx, ky;
      int lenx, leny;
      int ai, aij;
      int incai, incaij;

      const double *a_i = a;
      const double *head_x_i = head_x;
      const double *tail_x_i = tail_x;
      double *y_i = (double *) y;
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;
      double a_elem;
      double x_elem;
      double y_elem[2];
      double head_prod, tail_prod;
      double head_sum, tail_sum;
      double head_sum2, tail_sum2;
      double head_tmp1[2], tail_tmp1[2];
      double head_tmp2[2], tail_tmp2[2];
      FPU_FIX_DECL;

      /* all error calls */
      if (m < 0)
        BLAS_error(routine_name, -3, m, 0);
      else if (n <= 0)
        BLAS_error(routine_name, -4, n, 0);
      else if (incx == 0)
        BLAS_error(routine_name, -10, incx, 0);
      else if (incy == 0)
        BLAS_error(routine_name, -13, incy, 0);

      if ((order == blas_rowmajor) && (trans == blas_no_trans)) {
        lenx = n;
        leny = m;
        incai = lda;
        incaij = 1;
      } else if ((order == blas_rowmajor) && (trans != blas_no_trans)) {
        lenx = m;
        leny = n;
        incai = 1;
        incaij = lda;
      } else if ((order == blas_colmajor) && (trans == blas_no_trans)) {
        lenx = n;
        leny = m;
        incai = 1;
        incaij = lda;
      } else {                        /* colmajor and blas_trans */
        lenx = m;
        leny = n;
        incai = lda;
        incaij = 1;
      }

      if (lda < leny)
        BLAS_error(routine_name, -7, lda, NULL);

      FPU_FIX_START;


      incy *= 2;



      if (incx > 0)
        kx = 0;
      else
        kx = (1 - lenx) * incx;
      if (incy > 0)
        ky = 0;
      else
        ky = (1 - leny) * incy;

      /* No extra-precision needed for alpha = 0 */
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {
        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          iy = ky;
          for (i = 0; i < leny; i++) {
            y_i[iy] = 0.0;
            y_i[iy + 1] = 0.0;
            iy += incy;
          }
        } else if (!(beta_i[0] == 0.0 && beta_i[1] == 0.0)) {
          iy = ky;
          for (i = 0; i < leny; i++) {
            y_elem[0] = y_i[iy];
            y_elem[1] = y_i[iy + 1];
            {
              /* Compute complex-extra = complex-double * complex-double. */
              double head_t1, tail_t1;
              double head_t2, tail_t2;
              /* Real part */
              compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[0], beta_i[0]);
              compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[1], beta_i[1]);
              head_t2 = -head_t2;
              tail_t2 = -tail_t2;
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
              head_tmp1[0] = head_t1;
              tail_tmp1[0] = tail_t1;
              /* Imaginary part */
              compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[1], beta_i[0]);
              compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[0], beta_i[1]);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
              head_tmp1[1] = head_t1;
              tail_tmp1[1] = tail_t1;
            }
            y_i[iy] = head_tmp1[0];
            y_i[iy + 1] = head_tmp1[1];
            iy += incy;
          }
        }
      } else {                        /* alpha != 0 */

        /* if beta = 0, we can save m multiplies:
           y = alpha*A*head_x + alpha*A*tail_x  */
        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
            /* save m more multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              head_sum = tail_sum = 0.0;
              head_sum2 = tail_sum2 = 0.0;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
                x_elem = tail_x_i[jx];
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod, tail_prod);
                aij += incaij;
                jx += incx;
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_sum2, tail_sum2);
              head_tmp1[0] = head_sum;
              tail_tmp1[0] = tail_sum;
              head_tmp1[1] = tail_tmp1[1] = 0.0;
              y_i[iy] = head_tmp1[0];
              y_i[iy + 1] = head_tmp1[1];
              ai += incai;
              iy += incy;
            }                        /* end for */
          } else {                /* alpha != 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              head_sum = tail_sum = 0.0;
              head_sum2 = tail_sum2 = 0.0;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
                x_elem = tail_x_i[jx];
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod, tail_prod);
                aij += incaij;
                jx += incx;
              }
              {
                /* Compute complex-extra = complex-double * real. */
                double head_t, tail_t;
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sum, tail_sum, alpha_i[0]);
                head_tmp1[0] = head_t;
                tail_tmp1[0] = tail_t;
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sum, tail_sum, alpha_i[1]);
                head_tmp1[1] = head_t;
                tail_tmp1[1] = tail_t;
              }
              {
                /* Compute complex-extra = complex-double * real. */
                double head_t, tail_t;
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sum2, tail_sum2, alpha_i[0]);
                head_tmp2[0] = head_t;
                tail_tmp2[0] = tail_t;
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sum2, tail_sum2, alpha_i[1]);
                head_tmp2[1] = head_t;
                tail_tmp2[1] = tail_t;
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
                head_tmp1[0] = head_t;
                tail_tmp1[0] = tail_t;
                /* Imaginary part */
                head_a = head_tmp1[1];
                tail_a = tail_tmp1[1];
                head_b = head_tmp2[1];
                tail_b = tail_tmp2[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_tmp1[1] = head_t;
                tail_tmp1[1] = tail_t;
              }
              y_i[iy] = head_tmp1[0];
              y_i[iy + 1] = head_tmp1[1];
              ai += incai;
              iy += incy;
            }
          }
        } else {                /* beta != 0 */
          if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {
            /* save m multiplies if alpha = 1 */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              head_sum = tail_sum = 0.0;;
              head_sum2 = tail_sum2 = 0.0;;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
                x_elem = tail_x_i[jx];
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod, tail_prod);
                aij += incaij;
                jx += incx;
              }
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_sum2, tail_sum2);
              y_elem[0] = y_i[iy];
              y_elem[1] = y_i[iy + 1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[0], beta_i[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[1], beta_i[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp1[0] = head_t1;
                tail_tmp1[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[1], beta_i[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[0], beta_i[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp1[1] = head_t1;
                tail_tmp1[1] = tail_t1;
              }
              head_tmp2[0] = head_sum;
              tail_tmp2[0] = tail_sum;
              head_tmp2[1] = tail_tmp2[1] = 0.0;
              {
                double head_t, tail_t;
                double head_a, tail_a;
                double head_b, tail_b;
                /* Real part */
                head_a = head_tmp2[0];
                tail_a = tail_tmp2[0];
                head_b = head_tmp1[0];
                tail_b = tail_tmp1[0];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_tmp2[0] = head_t;
                tail_tmp2[0] = tail_t;
                /* Imaginary part */
                head_a = head_tmp2[1];
                tail_a = tail_tmp2[1];
                head_b = head_tmp1[1];
                tail_b = tail_tmp1[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_tmp2[1] = head_t;
                tail_tmp2[1] = tail_t;
              }
              y_i[iy] = head_tmp2[0];
              y_i[iy + 1] = head_tmp2[1];
              ai += incai;
              iy += incy;
            }
          } else {                /* alpha != 1, the most general form:
                                   y = alpha*A*head_x + alpha*A*tail_x + beta*y */
            ai = 0;
            iy = ky;
            for (i = 0; i < leny; i++) {
              head_sum = tail_sum = 0.0;;
              head_sum2 = tail_sum2 = 0.0;;
              aij = ai;
              jx = kx;
              for (j = 0; j < lenx; j++) {
                a_elem = a_i[aij];

                x_elem = head_x_i[jx];
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum, &tail_sum, head_sum, tail_sum, head_prod, tail_prod);
                x_elem = tail_x_i[jx];
                compute_doubledouble_eq_double_mul_double(&head_prod, &tail_prod, a_elem, x_elem);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_sum2, &tail_sum2, head_sum2, tail_sum2, head_prod, tail_prod);
                aij += incaij;
                jx += incx;
              }
              {
                /* Compute complex-extra = complex-double * real. */
                double head_t, tail_t;
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sum, tail_sum, alpha_i[0]);
                head_tmp1[0] = head_t;
                tail_tmp1[0] = tail_t;
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sum, tail_sum, alpha_i[1]);
                head_tmp1[1] = head_t;
                tail_tmp1[1] = tail_t;
              }
              {
                /* Compute complex-extra = complex-double * real. */
                double head_t, tail_t;
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sum2, tail_sum2, alpha_i[0]);
                head_tmp2[0] = head_t;
                tail_tmp2[0] = tail_t;
                compute_doubledouble_eq_doubledouble_mul_double(&head_t, &tail_t, head_sum2, tail_sum2, alpha_i[1]);
                head_tmp2[1] = head_t;
                tail_tmp2[1] = tail_t;
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
                head_tmp1[0] = head_t;
                tail_tmp1[0] = tail_t;
                /* Imaginary part */
                head_a = head_tmp1[1];
                tail_a = tail_tmp1[1];
                head_b = head_tmp2[1];
                tail_b = tail_tmp2[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_tmp1[1] = head_t;
                tail_tmp1[1] = tail_t;
              }
              y_elem[0] = y_i[iy];
              y_elem[1] = y_i[iy + 1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[0], beta_i[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[1], beta_i[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp2[0] = head_t1;
                tail_tmp2[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, y_elem[1], beta_i[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, y_elem[0], beta_i[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp2[1] = head_t1;
                tail_tmp2[1] = tail_t1;
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
                head_tmp1[0] = head_t;
                tail_tmp1[0] = tail_t;
                /* Imaginary part */
                head_a = head_tmp1[1];
                tail_a = tail_tmp1[1];
                head_b = head_tmp2[1];
                tail_b = tail_tmp2[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_tmp1[1] = head_t;
                tail_tmp1[1] = tail_t;
              }
              y_i[iy] = head_tmp1[0];
              y_i[iy + 1] = head_tmp1[1];
              ai += incai;
              iy += incy;
            }
          }
        }

      }

      FPU_FIX_STOP;
    }
    break;
  }
}
#endif
