#include <blas_extended.h>
#include <blas_extended_private.h>
#include <blas_fpu.h>
void BLAS_zhemv2_z_c_x(enum blas_order_type order,
                       enum blas_uplo_type uplo,
                       int n,
                       const void *alpha,
                       const void *a,
                       int lda,
                       const void *x_head,
                       const void *x_tail,
                       int incx,
                       const void *beta,
                       const void *y,
                       int incy,
                       enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routines computes the matrix product:
 *
 *     y  <-  alpha * A * (x_head + x_tail) + beta * y
 *
 * where A is a complex Hermitian matrix.
 *
 * Arguments
 * =========
 *
 * order   (input) enum blas_order_type
 *         Storage format of input symmetric matrix A.
 *
 * uplo    (input) enum blas_uplo_type
 *         Determines which half of matrix A (upper or lower triangle)
 *         is accessed.
 *
 * n       (input) int
 *         Dimension of A and size of vectors x, y.
 *
 * alpha   (input) const void*
 *
 * a       (input) const void*
 *         Matrix A.
 *
 * lda     (input) int
 *         Leading dimension of matrix A.
 *
 * x_head  (input) const void*
 *         Vector x_head
 *
 * x_tail  (input) const void*
 *         Vector x_tail
 *
 * incx    (input) int
 *         Stride for vector x.
 *
 * beta    (input) const void*
 *
 * y       (input) const void*
 *         Vector y.
 *
 * incy    (input) int
 *         Stride for vector y.
 *
 * prec    (input) enum blas_prec_type
 *         Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 */
{
//static const char routine_name[] = "BLAS_zhemv2_z_c_x";
#if 0
  XBLAS::hemv2_x(order,
                 uplo,
                 n,
                 *static_cast<const std::complex<double> *>(alpha),
                 static_cast<const std::complex<double> *>(a),
                 lda,
                 static_cast<const std::complex<float> *>(x_head),
                 static_cast<const std::complex<float> *>(x_tail),
                 incx,
                 *static_cast<const std::complex<double> *>(beta),
                 static_cast<const std::complex<double> *>(y),
                 incy,
                 prec);
} /* end BLAS_zhemv2_z_c_x */
#else
  /* Routine name */
  const char routine_name[] = "BLAS_zhemv2_z_c_x";
  switch (prec) {

  case blas_prec_single:{

      int i, j;
      int xi, yi, xi0, yi0;
      int aij, ai;
      int incai;
      int incaij, incaij2;

      const double *a_i = (double *) a;
      const float *x_head_i = (float *) x_head;
      const float *x_tail_i = (float *) x_tail;
      double *y_i = (double *) y;
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;
      double a_elem[2];
      float x_elem[2];
      double y_elem[2];
      double diag_elem;
      double prod1[2];
      double prod2[2];
      double sum1[2];
      double sum2[2];
      double tmp1[2];
      double tmp2[2];
      double tmp3[2];



      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
          && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
        return;
      }

      /* Check for error conditions. */
      if (n < 0) {
        BLAS_error(routine_name, -3, n, NULL);
      }
      if (lda < n) {
        BLAS_error(routine_name, -6, n, NULL);
      }
      if (incx == 0) {
        BLAS_error(routine_name, -9, incx, NULL);
      }
      if (incy == 0) {
        BLAS_error(routine_name, -12, incy, NULL);
      }

      if ((order == blas_colmajor && uplo == blas_upper) ||
          (order == blas_rowmajor && uplo == blas_lower)) {
        incai = lda;
        incaij = 1;
        incaij2 = lda;
      } else {
        incai = 1;
        incaij = lda;
        incaij2 = 1;
      }

      incx *= 2;
      incy *= 2;
      incai *= 2;
      incaij *= 2;
      incaij2 *= 2;
      xi0 = (incx > 0) ? 0 : ((-n + 1) * incx);
      yi0 = (incy > 0) ? 0 : ((-n + 1) * incy);



      /* The most general form,   y <--- alpha * A * (x_head + x_tail) + beta * y   */
      if (uplo == blas_lower) {
        for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
          sum1[0] = sum1[1] = 0.0;
          sum2[0] = sum2[1] = 0.0;
          for (j = 0, aij = ai, xi = xi0; j < i;
               j++, aij += incaij, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              prod1[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod1[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum1[0] = sum1[0] + prod1[0];
            sum1[1] = sum1[1] + prod1[1];
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              prod2[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod2[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum2[0] = sum2[0] + prod2[0];
            sum2[1] = sum2[1] + prod2[1];
          }

          diag_elem = a_i[aij];
          x_elem[0] = x_head_i[xi];
          x_elem[1] = x_head_i[xi + 1];
          {
            prod1[0] = x_elem[0] * diag_elem;
            prod1[1] = x_elem[1] * diag_elem;
          }
          sum1[0] = sum1[0] + prod1[0];
          sum1[1] = sum1[1] + prod1[1];
          x_elem[0] = x_tail_i[xi];
          x_elem[1] = x_tail_i[xi + 1];
          {
            prod2[0] = x_elem[0] * diag_elem;
            prod2[1] = x_elem[1] * diag_elem;
          }
          sum2[0] = sum2[0] + prod2[0];
          sum2[1] = sum2[1] + prod2[1];
          j++;
          aij += incaij2;
          xi += incx;

          for (; j < n; j++, aij += incaij2, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            a_elem[1] = -a_elem[1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              prod1[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod1[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum1[0] = sum1[0] + prod1[0];
            sum1[1] = sum1[1] + prod1[1];
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              prod2[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod2[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum2[0] = sum2[0] + prod2[0];
            sum2[1] = sum2[1] + prod2[1];
          }
          sum1[0] = sum1[0] + sum2[0];
          sum1[1] = sum1[1] + sum2[1];
          {
            tmp1[0] =
              (double) sum1[0] * alpha_i[0] - (double) sum1[1] * alpha_i[1];
            tmp1[1] =
              (double) sum1[0] * alpha_i[1] + (double) sum1[1] * alpha_i[0];
          }
          y_elem[0] = y_i[yi];
          y_elem[1] = y_i[yi + 1];
          {
            tmp2[0] =
              (double) y_elem[0] * beta_i[0] - (double) y_elem[1] * beta_i[1];
            tmp2[1] =
              (double) y_elem[0] * beta_i[1] + (double) y_elem[1] * beta_i[0];
          }
          tmp3[0] = tmp1[0] + tmp2[0];
          tmp3[1] = tmp1[1] + tmp2[1];
          y_i[yi] = tmp3[0];
          y_i[yi + 1] = tmp3[1];
        }
      } else {
        /* uplo == blas_upper */
        for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
          sum1[0] = sum1[1] = 0.0;
          sum2[0] = sum2[1] = 0.0;

          for (j = 0, aij = ai, xi = xi0; j < i;
               j++, aij += incaij, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            a_elem[1] = -a_elem[1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              prod1[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod1[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum1[0] = sum1[0] + prod1[0];
            sum1[1] = sum1[1] + prod1[1];
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              prod2[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod2[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum2[0] = sum2[0] + prod2[0];
            sum2[1] = sum2[1] + prod2[1];
          }

          diag_elem = a_i[aij];
          x_elem[0] = x_head_i[xi];
          x_elem[1] = x_head_i[xi + 1];
          {
            prod1[0] = x_elem[0] * diag_elem;
            prod1[1] = x_elem[1] * diag_elem;
          }
          sum1[0] = sum1[0] + prod1[0];
          sum1[1] = sum1[1] + prod1[1];
          x_elem[0] = x_tail_i[xi];
          x_elem[1] = x_tail_i[xi + 1];
          {
            prod2[0] = x_elem[0] * diag_elem;
            prod2[1] = x_elem[1] * diag_elem;
          }
          sum2[0] = sum2[0] + prod2[0];
          sum2[1] = sum2[1] + prod2[1];
          j++;
          aij += incaij2;
          xi += incx;

          for (; j < n; j++, aij += incaij2, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              prod1[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod1[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum1[0] = sum1[0] + prod1[0];
            sum1[1] = sum1[1] + prod1[1];
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              prod2[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod2[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum2[0] = sum2[0] + prod2[0];
            sum2[1] = sum2[1] + prod2[1];
          }
          sum1[0] = sum1[0] + sum2[0];
          sum1[1] = sum1[1] + sum2[1];
          {
            tmp1[0] =
              (double) sum1[0] * alpha_i[0] - (double) sum1[1] * alpha_i[1];
            tmp1[1] =
              (double) sum1[0] * alpha_i[1] + (double) sum1[1] * alpha_i[0];
          }
          y_elem[0] = y_i[yi];
          y_elem[1] = y_i[yi + 1];
          {
            tmp2[0] =
              (double) y_elem[0] * beta_i[0] - (double) y_elem[1] * beta_i[1];
            tmp2[1] =
              (double) y_elem[0] * beta_i[1] + (double) y_elem[1] * beta_i[0];
          }
          tmp3[0] = tmp1[0] + tmp2[0];
          tmp3[1] = tmp1[1] + tmp2[1];
          y_i[yi] = tmp3[0];
          y_i[yi + 1] = tmp3[1];
        }
      }



      break;
    }

  case blas_prec_double:
  case blas_prec_indigenous:{

      int i, j;
      int xi, yi, xi0, yi0;
      int aij, ai;
      int incai;
      int incaij, incaij2;

      const double *a_i = (double *) a;
      const float *x_head_i = (float *) x_head;
      const float *x_tail_i = (float *) x_tail;
      double *y_i = (double *) y;
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;
      double a_elem[2];
      float x_elem[2];
      double y_elem[2];
      double diag_elem;
      double prod1[2];
      double prod2[2];
      double sum1[2];
      double sum2[2];
      double tmp1[2];
      double tmp2[2];
      double tmp3[2];



      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
          && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
        return;
      }

      /* Check for error conditions. */
      if (n < 0) {
        BLAS_error(routine_name, -3, n, NULL);
      }
      if (lda < n) {
        BLAS_error(routine_name, -6, n, NULL);
      }
      if (incx == 0) {
        BLAS_error(routine_name, -9, incx, NULL);
      }
      if (incy == 0) {
        BLAS_error(routine_name, -12, incy, NULL);
      }

      if ((order == blas_colmajor && uplo == blas_upper) ||
          (order == blas_rowmajor && uplo == blas_lower)) {
        incai = lda;
        incaij = 1;
        incaij2 = lda;
      } else {
        incai = 1;
        incaij = lda;
        incaij2 = 1;
      }

      incx *= 2;
      incy *= 2;
      incai *= 2;
      incaij *= 2;
      incaij2 *= 2;
      xi0 = (incx > 0) ? 0 : ((-n + 1) * incx);
      yi0 = (incy > 0) ? 0 : ((-n + 1) * incy);



      /* The most general form,   y <--- alpha * A * (x_head + x_tail) + beta * y   */
      if (uplo == blas_lower) {
        for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
          sum1[0] = sum1[1] = 0.0;
          sum2[0] = sum2[1] = 0.0;
          for (j = 0, aij = ai, xi = xi0; j < i;
               j++, aij += incaij, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              prod1[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod1[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum1[0] = sum1[0] + prod1[0];
            sum1[1] = sum1[1] + prod1[1];
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              prod2[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod2[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum2[0] = sum2[0] + prod2[0];
            sum2[1] = sum2[1] + prod2[1];
          }

          diag_elem = a_i[aij];
          x_elem[0] = x_head_i[xi];
          x_elem[1] = x_head_i[xi + 1];
          {
            prod1[0] = x_elem[0] * diag_elem;
            prod1[1] = x_elem[1] * diag_elem;
          }
          sum1[0] = sum1[0] + prod1[0];
          sum1[1] = sum1[1] + prod1[1];
          x_elem[0] = x_tail_i[xi];
          x_elem[1] = x_tail_i[xi + 1];
          {
            prod2[0] = x_elem[0] * diag_elem;
            prod2[1] = x_elem[1] * diag_elem;
          }
          sum2[0] = sum2[0] + prod2[0];
          sum2[1] = sum2[1] + prod2[1];
          j++;
          aij += incaij2;
          xi += incx;

          for (; j < n; j++, aij += incaij2, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            a_elem[1] = -a_elem[1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              prod1[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod1[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum1[0] = sum1[0] + prod1[0];
            sum1[1] = sum1[1] + prod1[1];
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              prod2[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod2[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum2[0] = sum2[0] + prod2[0];
            sum2[1] = sum2[1] + prod2[1];
          }
          sum1[0] = sum1[0] + sum2[0];
          sum1[1] = sum1[1] + sum2[1];
          {
            tmp1[0] =
              (double) sum1[0] * alpha_i[0] - (double) sum1[1] * alpha_i[1];
            tmp1[1] =
              (double) sum1[0] * alpha_i[1] + (double) sum1[1] * alpha_i[0];
          }
          y_elem[0] = y_i[yi];
          y_elem[1] = y_i[yi + 1];
          {
            tmp2[0] =
              (double) y_elem[0] * beta_i[0] - (double) y_elem[1] * beta_i[1];
            tmp2[1] =
              (double) y_elem[0] * beta_i[1] + (double) y_elem[1] * beta_i[0];
          }
          tmp3[0] = tmp1[0] + tmp2[0];
          tmp3[1] = tmp1[1] + tmp2[1];
          y_i[yi] = tmp3[0];
          y_i[yi + 1] = tmp3[1];
        }
      } else {
        /* uplo == blas_upper */
        for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
          sum1[0] = sum1[1] = 0.0;
          sum2[0] = sum2[1] = 0.0;

          for (j = 0, aij = ai, xi = xi0; j < i;
               j++, aij += incaij, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            a_elem[1] = -a_elem[1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              prod1[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod1[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum1[0] = sum1[0] + prod1[0];
            sum1[1] = sum1[1] + prod1[1];
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              prod2[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod2[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum2[0] = sum2[0] + prod2[0];
            sum2[1] = sum2[1] + prod2[1];
          }

          diag_elem = a_i[aij];
          x_elem[0] = x_head_i[xi];
          x_elem[1] = x_head_i[xi + 1];
          {
            prod1[0] = x_elem[0] * diag_elem;
            prod1[1] = x_elem[1] * diag_elem;
          }
          sum1[0] = sum1[0] + prod1[0];
          sum1[1] = sum1[1] + prod1[1];
          x_elem[0] = x_tail_i[xi];
          x_elem[1] = x_tail_i[xi + 1];
          {
            prod2[0] = x_elem[0] * diag_elem;
            prod2[1] = x_elem[1] * diag_elem;
          }
          sum2[0] = sum2[0] + prod2[0];
          sum2[1] = sum2[1] + prod2[1];
          j++;
          aij += incaij2;
          xi += incx;

          for (; j < n; j++, aij += incaij2, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              prod1[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod1[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum1[0] = sum1[0] + prod1[0];
            sum1[1] = sum1[1] + prod1[1];
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              prod2[0] =
                (double) a_elem[0] * x_elem[0] -
                (double) a_elem[1] * x_elem[1];
              prod2[1] =
                (double) a_elem[0] * x_elem[1] +
                (double) a_elem[1] * x_elem[0];
            }
            sum2[0] = sum2[0] + prod2[0];
            sum2[1] = sum2[1] + prod2[1];
          }
          sum1[0] = sum1[0] + sum2[0];
          sum1[1] = sum1[1] + sum2[1];
          {
            tmp1[0] =
              (double) sum1[0] * alpha_i[0] - (double) sum1[1] * alpha_i[1];
            tmp1[1] =
              (double) sum1[0] * alpha_i[1] + (double) sum1[1] * alpha_i[0];
          }
          y_elem[0] = y_i[yi];
          y_elem[1] = y_i[yi + 1];
          {
            tmp2[0] =
              (double) y_elem[0] * beta_i[0] - (double) y_elem[1] * beta_i[1];
            tmp2[1] =
              (double) y_elem[0] * beta_i[1] + (double) y_elem[1] * beta_i[0];
          }
          tmp3[0] = tmp1[0] + tmp2[0];
          tmp3[1] = tmp1[1] + tmp2[1];
          y_i[yi] = tmp3[0];
          y_i[yi + 1] = tmp3[1];
        }
      }



      break;
    }

  case blas_prec_extra:{

      int i, j;
      int xi, yi, xi0, yi0;
      int aij, ai;
      int incai;
      int incaij, incaij2;

      const double *a_i = (double *) a;
      const float *x_head_i = (float *) x_head;
      const float *x_tail_i = (float *) x_tail;
      double *y_i = (double *) y;
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;
      double a_elem[2];
      float x_elem[2];
      double y_elem[2];
      double diag_elem;
      double head_prod1[2], tail_prod1[2];
      double head_prod2[2], tail_prod2[2];
      double head_sum1[2], tail_sum1[2];
      double head_sum2[2], tail_sum2[2];
      double head_tmp1[2], tail_tmp1[2];
      double head_tmp2[2], tail_tmp2[2];
      double head_tmp3[2], tail_tmp3[2];

      FPU_FIX_DECL;

      /* Test for no-op */
      if (n <= 0) {
        return;
      }
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
          && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
        return;
      }

      /* Check for error conditions. */
      if (n < 0) {
        BLAS_error(routine_name, -3, n, NULL);
      }
      if (lda < n) {
        BLAS_error(routine_name, -6, n, NULL);
      }
      if (incx == 0) {
        BLAS_error(routine_name, -9, incx, NULL);
      }
      if (incy == 0) {
        BLAS_error(routine_name, -12, incy, NULL);
      }

      if ((order == blas_colmajor && uplo == blas_upper) ||
          (order == blas_rowmajor && uplo == blas_lower)) {
        incai = lda;
        incaij = 1;
        incaij2 = lda;
      } else {
        incai = 1;
        incaij = lda;
        incaij2 = 1;
      }

      incx *= 2;
      incy *= 2;
      incai *= 2;
      incaij *= 2;
      incaij2 *= 2;
      xi0 = (incx > 0) ? 0 : ((-n + 1) * incx);
      yi0 = (incy > 0) ? 0 : ((-n + 1) * incy);

      FPU_FIX_START;

      /* The most general form,   y <--- alpha * A * (x_head + x_tail) + beta * y   */
      if (uplo == blas_lower) {
        for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
          head_sum1[0] = head_sum1[1] = tail_sum1[0] = tail_sum1[1] = 0.0;
          head_sum2[0] = head_sum2[1] = tail_sum2[0] = tail_sum2[1] = 0.0;
          for (j = 0, aij = ai, xi = xi0; j < i;
               j++, aij += incaij, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              double cd[2];
              cd[0] = (double) x_elem[0];
              cd[1] = (double) x_elem[1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[0], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[1], cd[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod1[0] = head_t1;
                tail_prod1[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[1], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[0], cd[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod1[1] = head_t1;
                tail_prod1[1] = tail_t1;
              }
            }
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_sum1[0];
              tail_a = tail_sum1[0];
              head_b = head_prod1[0];
              tail_b = tail_prod1[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum1[0] = head_t;
              tail_sum1[0] = tail_t;
              /* Imaginary part */
              head_a = head_sum1[1];
              tail_a = tail_sum1[1];
              head_b = head_prod1[1];
              tail_b = tail_prod1[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum1[1] = head_t;
              tail_sum1[1] = tail_t;
            }
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              double cd[2];
              cd[0] = (double) x_elem[0];
              cd[1] = (double) x_elem[1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[0], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[1], cd[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod2[0] = head_t1;
                tail_prod2[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[1], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[0], cd[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod2[1] = head_t1;
                tail_prod2[1] = tail_t1;
              }
            }
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_sum2[0];
              tail_a = tail_sum2[0];
              head_b = head_prod2[0];
              tail_b = tail_prod2[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum2[0] = head_t;
              tail_sum2[0] = tail_t;
              /* Imaginary part */
              head_a = head_sum2[1];
              tail_a = tail_sum2[1];
              head_b = head_prod2[1];
              tail_b = tail_prod2[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum2[1] = head_t;
              tail_sum2[1] = tail_t;
            }
          }

          diag_elem = a_i[aij];
          x_elem[0] = x_head_i[xi];
          x_elem[1] = x_head_i[xi + 1];
          {
            head_prod1[0] = (double) x_elem[0] * diag_elem;
            tail_prod1[0] = 0.0;
            head_prod1[1] = (double) x_elem[1] * diag_elem;
            tail_prod1[1] = 0.0;
          }
          {
            double head_t, tail_t;
            double head_a, tail_a;
            double head_b, tail_b;
            /* Real part */
            head_a = head_sum1[0];
            tail_a = tail_sum1[0];
            head_b = head_prod1[0];
            tail_b = tail_prod1[0];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum1[0] = head_t;
            tail_sum1[0] = tail_t;
            /* Imaginary part */
            head_a = head_sum1[1];
            tail_a = tail_sum1[1];
            head_b = head_prod1[1];
            tail_b = tail_prod1[1];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum1[1] = head_t;
            tail_sum1[1] = tail_t;
          }
          x_elem[0] = x_tail_i[xi];
          x_elem[1] = x_tail_i[xi + 1];
          {
            head_prod2[0] = (double) x_elem[0] * diag_elem;
            tail_prod2[0] = 0.0;
            head_prod2[1] = (double) x_elem[1] * diag_elem;
            tail_prod2[1] = 0.0;
          }
          {
            double head_t, tail_t;
            double head_a, tail_a;
            double head_b, tail_b;
            /* Real part */
            head_a = head_sum2[0];
            tail_a = tail_sum2[0];
            head_b = head_prod2[0];
            tail_b = tail_prod2[0];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum2[0] = head_t;
            tail_sum2[0] = tail_t;
            /* Imaginary part */
            head_a = head_sum2[1];
            tail_a = tail_sum2[1];
            head_b = head_prod2[1];
            tail_b = tail_prod2[1];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum2[1] = head_t;
            tail_sum2[1] = tail_t;
          }
          j++;
          aij += incaij2;
          xi += incx;

          for (; j < n; j++, aij += incaij2, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            a_elem[1] = -a_elem[1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              double cd[2];
              cd[0] = (double) x_elem[0];
              cd[1] = (double) x_elem[1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[0], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[1], cd[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod1[0] = head_t1;
                tail_prod1[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[1], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[0], cd[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod1[1] = head_t1;
                tail_prod1[1] = tail_t1;
              }
            }
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_sum1[0];
              tail_a = tail_sum1[0];
              head_b = head_prod1[0];
              tail_b = tail_prod1[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum1[0] = head_t;
              tail_sum1[0] = tail_t;
              /* Imaginary part */
              head_a = head_sum1[1];
              tail_a = tail_sum1[1];
              head_b = head_prod1[1];
              tail_b = tail_prod1[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum1[1] = head_t;
              tail_sum1[1] = tail_t;
            }
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              double cd[2];
              cd[0] = (double) x_elem[0];
              cd[1] = (double) x_elem[1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[0], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[1], cd[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod2[0] = head_t1;
                tail_prod2[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[1], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[0], cd[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod2[1] = head_t1;
                tail_prod2[1] = tail_t1;
              }
            }
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_sum2[0];
              tail_a = tail_sum2[0];
              head_b = head_prod2[0];
              tail_b = tail_prod2[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum2[0] = head_t;
              tail_sum2[0] = tail_t;
              /* Imaginary part */
              head_a = head_sum2[1];
              tail_a = tail_sum2[1];
              head_b = head_prod2[1];
              tail_b = tail_prod2[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum2[1] = head_t;
              tail_sum2[1] = tail_t;
            }
          }
          {
            double head_t, tail_t;
            double head_a, tail_a;
            double head_b, tail_b;
            /* Real part */
            head_a = head_sum1[0];
            tail_a = tail_sum1[0];
            head_b = head_sum2[0];
            tail_b = tail_sum2[0];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum1[0] = head_t;
            tail_sum1[0] = tail_t;
            /* Imaginary part */
            head_a = head_sum1[1];
            tail_a = tail_sum1[1];
            head_b = head_sum2[1];
            tail_b = tail_sum2[1];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum1[1] = head_t;
            tail_sum1[1] = tail_t;
          }
          {
            /* Compute complex-extra = complex-extra * complex-double. */
            double head_a0, tail_a0;
            double head_a1, tail_a1;
            double head_t1, tail_t1;
            double head_t2, tail_t2;
            head_a0 = head_sum1[0];
            tail_a0 = tail_sum1[0];
            head_a1 = head_sum1[1];
            tail_a1 = tail_sum1[1];
            /* real part */
            compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
            compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
            head_t2 = -head_t2;
            tail_t2 = -tail_t2;
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
            head_tmp1[0] = head_t1;
            tail_tmp1[0] = tail_t1;
            /* imaginary part */
            compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
            compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
            head_tmp1[1] = head_t1;
            tail_tmp1[1] = tail_t1;
          }

          y_elem[0] = y_i[yi];
          y_elem[1] = y_i[yi + 1];
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
            head_tmp3[0] = head_t;
            tail_tmp3[0] = tail_t;
            /* Imaginary part */
            head_a = head_tmp1[1];
            tail_a = tail_tmp1[1];
            head_b = head_tmp2[1];
            tail_b = tail_tmp2[1];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_tmp3[1] = head_t;
            tail_tmp3[1] = tail_t;
          }
          y_i[yi] = head_tmp3[0];
          y_i[yi + 1] = head_tmp3[1];
        }
      } else {
        /* uplo == blas_upper */
        for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
          head_sum1[0] = head_sum1[1] = tail_sum1[0] = tail_sum1[1] = 0.0;
          head_sum2[0] = head_sum2[1] = tail_sum2[0] = tail_sum2[1] = 0.0;

          for (j = 0, aij = ai, xi = xi0; j < i;
               j++, aij += incaij, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            a_elem[1] = -a_elem[1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              double cd[2];
              cd[0] = (double) x_elem[0];
              cd[1] = (double) x_elem[1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[0], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[1], cd[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod1[0] = head_t1;
                tail_prod1[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[1], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[0], cd[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod1[1] = head_t1;
                tail_prod1[1] = tail_t1;
              }
            }
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_sum1[0];
              tail_a = tail_sum1[0];
              head_b = head_prod1[0];
              tail_b = tail_prod1[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum1[0] = head_t;
              tail_sum1[0] = tail_t;
              /* Imaginary part */
              head_a = head_sum1[1];
              tail_a = tail_sum1[1];
              head_b = head_prod1[1];
              tail_b = tail_prod1[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum1[1] = head_t;
              tail_sum1[1] = tail_t;
            }
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              double cd[2];
              cd[0] = (double) x_elem[0];
              cd[1] = (double) x_elem[1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[0], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[1], cd[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod2[0] = head_t1;
                tail_prod2[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[1], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[0], cd[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod2[1] = head_t1;
                tail_prod2[1] = tail_t1;
              }
            }
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_sum2[0];
              tail_a = tail_sum2[0];
              head_b = head_prod2[0];
              tail_b = tail_prod2[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum2[0] = head_t;
              tail_sum2[0] = tail_t;
              /* Imaginary part */
              head_a = head_sum2[1];
              tail_a = tail_sum2[1];
              head_b = head_prod2[1];
              tail_b = tail_prod2[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum2[1] = head_t;
              tail_sum2[1] = tail_t;
            }
          }

          diag_elem = a_i[aij];
          x_elem[0] = x_head_i[xi];
          x_elem[1] = x_head_i[xi + 1];
          {
            head_prod1[0] = (double) x_elem[0] * diag_elem;
            tail_prod1[0] = 0.0;
            head_prod1[1] = (double) x_elem[1] * diag_elem;
            tail_prod1[1] = 0.0;
          }
          {
            double head_t, tail_t;
            double head_a, tail_a;
            double head_b, tail_b;
            /* Real part */
            head_a = head_sum1[0];
            tail_a = tail_sum1[0];
            head_b = head_prod1[0];
            tail_b = tail_prod1[0];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum1[0] = head_t;
            tail_sum1[0] = tail_t;
            /* Imaginary part */
            head_a = head_sum1[1];
            tail_a = tail_sum1[1];
            head_b = head_prod1[1];
            tail_b = tail_prod1[1];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum1[1] = head_t;
            tail_sum1[1] = tail_t;
          }
          x_elem[0] = x_tail_i[xi];
          x_elem[1] = x_tail_i[xi + 1];
          {
            head_prod2[0] = (double) x_elem[0] * diag_elem;
            tail_prod2[0] = 0.0;
            head_prod2[1] = (double) x_elem[1] * diag_elem;
            tail_prod2[1] = 0.0;
          }
          {
            double head_t, tail_t;
            double head_a, tail_a;
            double head_b, tail_b;
            /* Real part */
            head_a = head_sum2[0];
            tail_a = tail_sum2[0];
            head_b = head_prod2[0];
            tail_b = tail_prod2[0];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum2[0] = head_t;
            tail_sum2[0] = tail_t;
            /* Imaginary part */
            head_a = head_sum2[1];
            tail_a = tail_sum2[1];
            head_b = head_prod2[1];
            tail_b = tail_prod2[1];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum2[1] = head_t;
            tail_sum2[1] = tail_t;
          }
          j++;
          aij += incaij2;
          xi += incx;

          for (; j < n; j++, aij += incaij2, xi += incx) {
            a_elem[0] = a_i[aij];
            a_elem[1] = a_i[aij + 1];
            x_elem[0] = x_head_i[xi];
            x_elem[1] = x_head_i[xi + 1];
            {
              double cd[2];
              cd[0] = (double) x_elem[0];
              cd[1] = (double) x_elem[1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[0], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[1], cd[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod1[0] = head_t1;
                tail_prod1[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[1], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[0], cd[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod1[1] = head_t1;
                tail_prod1[1] = tail_t1;
              }
            }
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_sum1[0];
              tail_a = tail_sum1[0];
              head_b = head_prod1[0];
              tail_b = tail_prod1[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum1[0] = head_t;
              tail_sum1[0] = tail_t;
              /* Imaginary part */
              head_a = head_sum1[1];
              tail_a = tail_sum1[1];
              head_b = head_prod1[1];
              tail_b = tail_prod1[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum1[1] = head_t;
              tail_sum1[1] = tail_t;
            }
            x_elem[0] = x_tail_i[xi];
            x_elem[1] = x_tail_i[xi + 1];
            {
              double cd[2];
              cd[0] = (double) x_elem[0];
              cd[1] = (double) x_elem[1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[0], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[1], cd[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod2[0] = head_t1;
                tail_prod2[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, a_elem[1], cd[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, a_elem[0], cd[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_prod2[1] = head_t1;
                tail_prod2[1] = tail_t1;
              }
            }
            {
              double head_t, tail_t;
              double head_a, tail_a;
              double head_b, tail_b;
              /* Real part */
              head_a = head_sum2[0];
              tail_a = tail_sum2[0];
              head_b = head_prod2[0];
              tail_b = tail_prod2[0];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum2[0] = head_t;
              tail_sum2[0] = tail_t;
              /* Imaginary part */
              head_a = head_sum2[1];
              tail_a = tail_sum2[1];
              head_b = head_prod2[1];
              tail_b = tail_prod2[1];
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
              head_sum2[1] = head_t;
              tail_sum2[1] = tail_t;
            }
          }
          {
            double head_t, tail_t;
            double head_a, tail_a;
            double head_b, tail_b;
            /* Real part */
            head_a = head_sum1[0];
            tail_a = tail_sum1[0];
            head_b = head_sum2[0];
            tail_b = tail_sum2[0];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum1[0] = head_t;
            tail_sum1[0] = tail_t;
            /* Imaginary part */
            head_a = head_sum1[1];
            tail_a = tail_sum1[1];
            head_b = head_sum2[1];
            tail_b = tail_sum2[1];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_sum1[1] = head_t;
            tail_sum1[1] = tail_t;
          }
          {
            /* Compute complex-extra = complex-extra * complex-double. */
            double head_a0, tail_a0;
            double head_a1, tail_a1;
            double head_t1, tail_t1;
            double head_t2, tail_t2;
            head_a0 = head_sum1[0];
            tail_a0 = tail_sum1[0];
            head_a1 = head_sum1[1];
            tail_a1 = tail_sum1[1];
            /* real part */
            compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a0, tail_a0, alpha_i[0]);
            compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a1, tail_a1, alpha_i[1]);
            head_t2 = -head_t2;
            tail_t2 = -tail_t2;
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
            head_tmp1[0] = head_t1;
            tail_tmp1[0] = tail_t1;
            /* imaginary part */
            compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
            compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
            head_tmp1[1] = head_t1;
            tail_tmp1[1] = tail_t1;
          }

          y_elem[0] = y_i[yi];
          y_elem[1] = y_i[yi + 1];
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
            head_tmp3[0] = head_t;
            tail_tmp3[0] = tail_t;
            /* Imaginary part */
            head_a = head_tmp1[1];
            tail_a = tail_tmp1[1];
            head_b = head_tmp2[1];
            tail_b = tail_tmp2[1];
            compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
            head_tmp3[1] = head_t;
            tail_tmp3[1] = tail_t;
          }
          y_i[yi] = head_tmp3[0];
          y_i[yi + 1] = head_tmp3[1];
        }
      }

      FPU_FIX_STOP;

      break;
    }
  }
}                                /* end BLAS_zhemv2_z_c_x */
#endif
