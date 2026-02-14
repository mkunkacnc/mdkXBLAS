#include "blas_extended.h"
#include "blas_extended_private.h"
#include "gemm/XBLAS_gemm.hpp"

void BLAS_zgemm_z_d_x(enum blas_order_type order,
                      enum blas_trans_type transa,
                      enum blas_trans_type transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *a,
                      int lda,
                      const double *b,
                      int ldb,
                      const void *beta,
                      void *c,
                      int ldc,
                      enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine computes the matrix product:
 *
 *      C   <-  alpha * op(A) * op(B)  +  beta * C .
 *
 * where op(M) represents either M, M transpose,
 * or M conjugate transpose.
 *
 * Arguments
 * =========
 *
 * order   (input) enum blas_order_type
 *         Storage format of input matrices A, B, and C.
 *
 * transa  (input) enum blas_trans_type
 *         Operation to be done on matrix A before multiplication.
 *           Can be no operation, transposition, or conjugate transposition.
 *
 * transb  (input) enum blas_trans_type
 *         Operation to be done on matrix B before multiplication.
 *           Can be no operation, transposition, or conjugate transposition.
 *
 * m n k   (input) int
 *         The dimensions of matrices A, B, and C.
 *         Matrix C is m-by-n matrix.
 *         Matrix A is m-by-k if A is not transposed,
 *                     k-by-m otherwise.
 *         Matrix B is k-by-n if B is not transposed,
 *                     n-by-k otherwise.
 *
 * alpha   (input) const void*
 *
 * a       (input) const void*
 *         matrix A.
 *
 * lda     (input) int
 *         leading dimension of A.
 *
 * b       (input) const double*
 *         matrix B
 *
 * ldb     (input) int
 *         leading dimension of B.
 *
 * beta    (input) const void*
 *
 * c       (input/output) void*
 *         matrix C
 *
 * ldc     (input) int
 *         leading dimension of C.
 *
 * prec    (input) enum blas_prec_type
 *         Specifies the internal precision to be used.
 *         = blas_prec_single: single precision.
 *         = blas_prec_double: double precision.
 *         = blas_prec_extra : anything at least 1.5 times as accurate
 *                             than double, and wider than 80-bits.
 *                             We use double-double in our implementation.
 *
 */
{
//static const char routine_name[] = "BLAS_zgemm_z_d_x";
#if 0
  XBLAS::gemm_x(order,
                transa,
                transb,
                m,
                n,
                k,
                *static_cast<const std::complex<double> *>(alpha),
                static_cast<const std::complex<double> *>(a),
                lda,
                b,
                ldb,
                *static_cast<const std::complex<double> *>(beta),
                static_cast<std::complex<double> *>(c),
                ldc,
                prec);
} /* end BLAS_zgemm_z_d_x */
#else
  static const char routine_name[] = "BLAS_zgemm_z_d_x";
  switch (prec) {

  case blas_prec_single:
  case blas_prec_double:
  case blas_prec_indigenous:{


      /* Integer Index Variables */
      int i, j, h;

      int ai, bj, ci;
      int aih, bhj, cij;        /* Index into matrices a, b, c during multiply */

      int incai, incaih;        /* Index increments for matrix a */
      int incbj, incbhj;        /* Index increments for matrix b */
      int incci, inccij;        /* Index increments for matrix c */

      /* Input Matrices */
      const double *a_i = (double *) a;
      const double *b_i = b;

      /* Output Matrix */
      double *c_i = (double *) c;

      /* Input Scalars */
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;

      /* Temporary Floating-Point Variables */
      double a_elem[2];
      double b_elem;
      double c_elem[2];
      double prod[2];
      double sum[2];
      double tmp1[2];
      double tmp2[2];



      /* Test for error conditions */
      if (m < 0)
        BLAS_error(routine_name, -4, m, NULL);
      if (n < 0)
        BLAS_error(routine_name, -5, n, NULL);
      if (k < 0)
        BLAS_error(routine_name, -6, k, NULL);

      if (order == blas_colmajor) {

        if (ldc < m)
          BLAS_error(routine_name, -14, ldc, NULL);

        if (transa == blas_no_trans) {
          if (lda < m)
            BLAS_error(routine_name, -9, lda, NULL);
        } else {
          if (lda < k)
            BLAS_error(routine_name, -9, lda, NULL);
        }

        if (transb == blas_no_trans) {
          if (ldb < k)
            BLAS_error(routine_name, -11, ldb, NULL);
        } else {
          if (ldb < n)
            BLAS_error(routine_name, -11, ldb, NULL);
        }

      } else {
        /* row major */
        if (ldc < n)
          BLAS_error(routine_name, -14, ldc, NULL);

        if (transa == blas_no_trans) {
          if (lda < k)
            BLAS_error(routine_name, -9, lda, NULL);
        } else {
          if (lda < m)
            BLAS_error(routine_name, -9, lda, NULL);
        }

        if (transb == blas_no_trans) {
          if (ldb < n)
            BLAS_error(routine_name, -11, ldb, NULL);
        } else {
          if (ldb < k)
            BLAS_error(routine_name, -11, ldb, NULL);
        }
      }

      /* Test for no-op */
      if (n == 0 || m == 0 || k == 0)
        return;
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
          && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
        return;
      }

      /* Set Index Parameters */
      if (order == blas_colmajor) {
        incci = 1;
        inccij = ldc;

        if (transa == blas_no_trans) {
          incai = 1;
          incaih = lda;
        } else {
          incai = lda;
          incaih = 1;
        }

        if (transb == blas_no_trans) {
          incbj = ldb;
          incbhj = 1;
        } else {
          incbj = 1;
          incbhj = ldb;
        }

      } else {
        /* row major */
        incci = ldc;
        inccij = 1;

        if (transa == blas_no_trans) {
          incai = lda;
          incaih = 1;
        } else {
          incai = 1;
          incaih = lda;
        }

        if (transb == blas_no_trans) {
          incbj = 1;
          incbhj = ldb;
        } else {
          incbj = ldb;
          incbhj = 1;
        }

      }



      /* Ajustment to increments */
      incci *= 2;
      inccij *= 2;
      incai *= 2;
      incaih *= 2;



      /* alpha = 0.  In this case, just return beta * C */
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {

        ci = 0;
        for (i = 0; i < m; i++, ci += incci) {
          cij = ci;
          for (j = 0; j < n; j++, cij += inccij) {
            c_elem[0] = c_i[cij];
            c_elem[1] = c_i[cij + 1];
            {
              tmp1[0] =
                (double) c_elem[0] * beta_i[0] -
                (double) c_elem[1] * beta_i[1];
              tmp1[1] =
                (double) c_elem[0] * beta_i[1] +
                (double) c_elem[1] * beta_i[0];
            }
            c_i[cij] = tmp1[0];
            c_i[cij + 1] = tmp1[1];
          }
        }

      } else if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {

        /* Case alpha == 1. */

        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          /* Case alpha == 1, beta == 0.   We compute  C <--- A * B */

          ci = 0;
          ai = 0;
          for (i = 0; i < m; i++, ci += incci, ai += incai) {

            cij = ci;
            bj = 0;

            for (j = 0; j < n; j++, cij += inccij, bj += incbj) {

              aih = ai;
              bhj = bj;

              sum[0] = sum[1] = 0.0;

              for (h = 0; h < k; h++, aih += incaih, bhj += incbhj) {
                a_elem[0] = a_i[aih];
                a_elem[1] = a_i[aih + 1];
                b_elem = b_i[bhj];
                if (transa == blas_conj_trans) {
                  a_elem[1] = -a_elem[1];
                }
                if (transb == blas_conj_trans) {

                }
                {
                  prod[0] = a_elem[0] * b_elem;
                  prod[1] = a_elem[1] * b_elem;
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];
              }
              c_i[cij] = sum[0];
              c_i[cij + 1] = sum[1];
            }
          }

        } else {
          /* Case alpha == 1, but beta != 0.
             We compute   C <--- A * B + beta * C   */

          ci = 0;
          ai = 0;
          for (i = 0; i < m; i++, ci += incci, ai += incai) {

            cij = ci;
            bj = 0;

            for (j = 0; j < n; j++, cij += inccij, bj += incbj) {

              aih = ai;
              bhj = bj;

              sum[0] = sum[1] = 0.0;

              for (h = 0; h < k; h++, aih += incaih, bhj += incbhj) {
                a_elem[0] = a_i[aih];
                a_elem[1] = a_i[aih + 1];
                b_elem = b_i[bhj];
                if (transa == blas_conj_trans) {
                  a_elem[1] = -a_elem[1];
                }
                if (transb == blas_conj_trans) {

                }
                {
                  prod[0] = a_elem[0] * b_elem;
                  prod[1] = a_elem[1] * b_elem;
                }
                sum[0] = sum[0] + prod[0];
                sum[1] = sum[1] + prod[1];
              }

              c_elem[0] = c_i[cij];
              c_elem[1] = c_i[cij + 1];
              {
                tmp2[0] =
                  (double) c_elem[0] * beta_i[0] -
                  (double) c_elem[1] * beta_i[1];
                tmp2[1] =
                  (double) c_elem[0] * beta_i[1] +
                  (double) c_elem[1] * beta_i[0];
              }
              tmp1[0] = sum[0];
              tmp1[1] = sum[1];
              tmp1[0] = tmp2[0] + tmp1[0];
              tmp1[1] = tmp2[1] + tmp1[1];
              c_i[cij] = tmp1[0];
              c_i[cij + 1] = tmp1[1];
            }
          }
        }

      } else {

        /* The most general form,   C <-- alpha * A * B + beta * C  */
        ci = 0;
        ai = 0;
        for (i = 0; i < m; i++, ci += incci, ai += incai) {

          cij = ci;
          bj = 0;

          for (j = 0; j < n; j++, cij += inccij, bj += incbj) {

            aih = ai;
            bhj = bj;

            sum[0] = sum[1] = 0.0;

            for (h = 0; h < k; h++, aih += incaih, bhj += incbhj) {
              a_elem[0] = a_i[aih];
              a_elem[1] = a_i[aih + 1];
              b_elem = b_i[bhj];
              if (transa == blas_conj_trans) {
                a_elem[1] = -a_elem[1];
              }
              if (transb == blas_conj_trans) {

              }
              {
                prod[0] = a_elem[0] * b_elem;
                prod[1] = a_elem[1] * b_elem;
              }
              sum[0] = sum[0] + prod[0];
              sum[1] = sum[1] + prod[1];
            }

            {
              tmp1[0] =
                (double) sum[0] * alpha_i[0] - (double) sum[1] * alpha_i[1];
              tmp1[1] =
                (double) sum[0] * alpha_i[1] + (double) sum[1] * alpha_i[0];
            }
            c_elem[0] = c_i[cij];
            c_elem[1] = c_i[cij + 1];
            {
              tmp2[0] =
                (double) c_elem[0] * beta_i[0] -
                (double) c_elem[1] * beta_i[1];
              tmp2[1] =
                (double) c_elem[0] * beta_i[1] +
                (double) c_elem[1] * beta_i[0];
            }
            tmp1[0] = tmp1[0] + tmp2[0];
            tmp1[1] = tmp1[1] + tmp2[1];
            c_i[cij] = tmp1[0];
            c_i[cij + 1] = tmp1[1];
          }
        }

      }



      break;
    }

  case blas_prec_extra:{


      /* Integer Index Variables */
      int i, j, h;

      int ai, bj, ci;
      int aih, bhj, cij;        /* Index into matrices a, b, c during multiply */

      int incai, incaih;        /* Index increments for matrix a */
      int incbj, incbhj;        /* Index increments for matrix b */
      int incci, inccij;        /* Index increments for matrix c */

      /* Input Matrices */
      const double *a_i = (double *) a;
      const double *b_i = b;

      /* Output Matrix */
      double *c_i = (double *) c;

      /* Input Scalars */
      double *alpha_i = (double *) alpha;
      double *beta_i = (double *) beta;

      /* Temporary Floating-Point Variables */
      double a_elem[2];
      double b_elem;
      double c_elem[2];
      double head_prod[2], tail_prod[2];
      double head_sum[2], tail_sum[2];
      double head_tmp1[2], tail_tmp1[2];
      double head_tmp2[2], tail_tmp2[2];

      FPU_FIX_DECL;

      /* Test for error conditions */
      if (m < 0)
        BLAS_error(routine_name, -4, m, NULL);
      if (n < 0)
        BLAS_error(routine_name, -5, n, NULL);
      if (k < 0)
        BLAS_error(routine_name, -6, k, NULL);

      if (order == blas_colmajor) {

        if (ldc < m)
          BLAS_error(routine_name, -14, ldc, NULL);

        if (transa == blas_no_trans) {
          if (lda < m)
            BLAS_error(routine_name, -9, lda, NULL);
        } else {
          if (lda < k)
            BLAS_error(routine_name, -9, lda, NULL);
        }

        if (transb == blas_no_trans) {
          if (ldb < k)
            BLAS_error(routine_name, -11, ldb, NULL);
        } else {
          if (ldb < n)
            BLAS_error(routine_name, -11, ldb, NULL);
        }

      } else {
        /* row major */
        if (ldc < n)
          BLAS_error(routine_name, -14, ldc, NULL);

        if (transa == blas_no_trans) {
          if (lda < k)
            BLAS_error(routine_name, -9, lda, NULL);
        } else {
          if (lda < m)
            BLAS_error(routine_name, -9, lda, NULL);
        }

        if (transb == blas_no_trans) {
          if (ldb < n)
            BLAS_error(routine_name, -11, ldb, NULL);
        } else {
          if (ldb < k)
            BLAS_error(routine_name, -11, ldb, NULL);
        }
      }

      /* Test for no-op */
      if (n == 0 || m == 0 || k == 0)
        return;
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0
          && (beta_i[0] == 1.0 && beta_i[1] == 0.0)) {
        return;
      }

      /* Set Index Parameters */
      if (order == blas_colmajor) {
        incci = 1;
        inccij = ldc;

        if (transa == blas_no_trans) {
          incai = 1;
          incaih = lda;
        } else {
          incai = lda;
          incaih = 1;
        }

        if (transb == blas_no_trans) {
          incbj = ldb;
          incbhj = 1;
        } else {
          incbj = 1;
          incbhj = ldb;
        }

      } else {
        /* row major */
        incci = ldc;
        inccij = 1;

        if (transa == blas_no_trans) {
          incai = lda;
          incaih = 1;
        } else {
          incai = 1;
          incaih = lda;
        }

        if (transb == blas_no_trans) {
          incbj = 1;
          incbhj = ldb;
        } else {
          incbj = ldb;
          incbhj = 1;
        }

      }

      FPU_FIX_START;

      /* Ajustment to increments */
      incci *= 2;
      inccij *= 2;
      incai *= 2;
      incaih *= 2;



      /* alpha = 0.  In this case, just return beta * C */
      if (alpha_i[0] == 0.0 && alpha_i[1] == 0.0) {

        ci = 0;
        for (i = 0; i < m; i++, ci += incci) {
          cij = ci;
          for (j = 0; j < n; j++, cij += inccij) {
            c_elem[0] = c_i[cij];
            c_elem[1] = c_i[cij + 1];
            {
              /* Compute complex-extra = complex-double * complex-double. */
              double head_t1, tail_t1;
              double head_t2, tail_t2;
              /* Real part */
              compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, c_elem[0], beta_i[0]);
              compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, c_elem[1], beta_i[1]);
              head_t2 = -head_t2;
              tail_t2 = -tail_t2;
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
              head_tmp1[0] = head_t1;
              tail_tmp1[0] = tail_t1;
              /* Imaginary part */
              compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, c_elem[1], beta_i[0]);
              compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, c_elem[0], beta_i[1]);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
              head_tmp1[1] = head_t1;
              tail_tmp1[1] = tail_t1;
            }
            c_i[cij] = head_tmp1[0];
            c_i[cij + 1] = head_tmp1[1];
          }
        }

      } else if ((alpha_i[0] == 1.0 && alpha_i[1] == 0.0)) {

        /* Case alpha == 1. */

        if (beta_i[0] == 0.0 && beta_i[1] == 0.0) {
          /* Case alpha == 1, beta == 0.   We compute  C <--- A * B */

          ci = 0;
          ai = 0;
          for (i = 0; i < m; i++, ci += incci, ai += incai) {

            cij = ci;
            bj = 0;

            for (j = 0; j < n; j++, cij += inccij, bj += incbj) {

              aih = ai;
              bhj = bj;

              head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

              for (h = 0; h < k; h++, aih += incaih, bhj += incbhj) {
                a_elem[0] = a_i[aih];
                a_elem[1] = a_i[aih + 1];
                b_elem = b_i[bhj];
                if (transa == blas_conj_trans) {
                  a_elem[1] = -a_elem[1];
                }
                if (transb == blas_conj_trans) {

                }
                {
                  /* Compute complex-extra = complex-double * real. */
                  double head_t, tail_t;
                  compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, b_elem, a_elem[0]);
                  head_prod[0] = head_t;
                  tail_prod[0] = tail_t;
                  compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, b_elem, a_elem[1]);
                  head_prod[1] = head_t;
                  tail_prod[1] = tail_t;
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
              }
              c_i[cij] = head_sum[0];
              c_i[cij + 1] = head_sum[1];
            }
          }

        } else {
          /* Case alpha == 1, but beta != 0.
             We compute   C <--- A * B + beta * C   */

          ci = 0;
          ai = 0;
          for (i = 0; i < m; i++, ci += incci, ai += incai) {

            cij = ci;
            bj = 0;

            for (j = 0; j < n; j++, cij += inccij, bj += incbj) {

              aih = ai;
              bhj = bj;

              head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

              for (h = 0; h < k; h++, aih += incaih, bhj += incbhj) {
                a_elem[0] = a_i[aih];
                a_elem[1] = a_i[aih + 1];
                b_elem = b_i[bhj];
                if (transa == blas_conj_trans) {
                  a_elem[1] = -a_elem[1];
                }
                if (transb == blas_conj_trans) {

                }
                {
                  /* Compute complex-extra = complex-double * real. */
                  double head_t, tail_t;
                  compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, b_elem, a_elem[0]);
                  head_prod[0] = head_t;
                  tail_prod[0] = tail_t;
                  compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, b_elem, a_elem[1]);
                  head_prod[1] = head_t;
                  tail_prod[1] = tail_t;
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
              }

              c_elem[0] = c_i[cij];
              c_elem[1] = c_i[cij + 1];
              {
                /* Compute complex-extra = complex-double * complex-double. */
                double head_t1, tail_t1;
                double head_t2, tail_t2;
                /* Real part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, c_elem[0], beta_i[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, c_elem[1], beta_i[1]);
                head_t2 = -head_t2;
                tail_t2 = -tail_t2;
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp2[0] = head_t1;
                tail_tmp2[0] = tail_t1;
                /* Imaginary part */
                compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, c_elem[1], beta_i[0]);
                compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, c_elem[0], beta_i[1]);
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
                head_tmp2[1] = head_t1;
                tail_tmp2[1] = tail_t1;
              }
              head_tmp1[0] = head_sum[0];
              tail_tmp1[0] = tail_sum[0];
              head_tmp1[1] = head_sum[1];
              tail_tmp1[1] = tail_sum[1];
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
                head_tmp1[0] = head_t;
                tail_tmp1[0] = tail_t;
                /* Imaginary part */
                head_a = head_tmp2[1];
                tail_a = tail_tmp2[1];
                head_b = head_tmp1[1];
                tail_b = tail_tmp1[1];
                compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t, &tail_t, head_a, tail_a, head_b, tail_b);
                head_tmp1[1] = head_t;
                tail_tmp1[1] = tail_t;
              }
              c_i[cij] = head_tmp1[0];
              c_i[cij + 1] = head_tmp1[1];
            }
          }
        }

      } else {

        /* The most general form,   C <-- alpha * A * B + beta * C  */
        ci = 0;
        ai = 0;
        for (i = 0; i < m; i++, ci += incci, ai += incai) {

          cij = ci;
          bj = 0;

          for (j = 0; j < n; j++, cij += inccij, bj += incbj) {

            aih = ai;
            bhj = bj;

            head_sum[0] = head_sum[1] = tail_sum[0] = tail_sum[1] = 0.0;

            for (h = 0; h < k; h++, aih += incaih, bhj += incbhj) {
              a_elem[0] = a_i[aih];
              a_elem[1] = a_i[aih + 1];
              b_elem = b_i[bhj];
              if (transa == blas_conj_trans) {
                a_elem[1] = -a_elem[1];
              }
              if (transb == blas_conj_trans) {

              }
              {
                /* Compute complex-extra = complex-double * real. */
                double head_t, tail_t;
                compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, b_elem, a_elem[0]);
                head_prod[0] = head_t;
                tail_prod[0] = tail_t;
                compute_doubledouble_eq_double_mul_double(&head_t, &tail_t, b_elem, a_elem[1]);
                head_prod[1] = head_t;
                tail_prod[1] = tail_t;
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
            }

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
              head_tmp1[0] = head_t1;
              tail_tmp1[0] = tail_t1;
              /* imaginary part */
              compute_doubledouble_eq_doubledouble_mul_double(&head_t1, &tail_t1, head_a1, tail_a1, alpha_i[0]);
              compute_doubledouble_eq_doubledouble_mul_double(&head_t2, &tail_t2, head_a0, tail_a0, alpha_i[1]);
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
              head_tmp1[1] = head_t1;
              tail_tmp1[1] = tail_t1;
            }

            c_elem[0] = c_i[cij];
            c_elem[1] = c_i[cij + 1];
            {
              /* Compute complex-extra = complex-double * complex-double. */
              double head_t1, tail_t1;
              double head_t2, tail_t2;
              /* Real part */
              compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, c_elem[0], beta_i[0]);
              compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, c_elem[1], beta_i[1]);
              head_t2 = -head_t2;
              tail_t2 = -tail_t2;
              compute_doubledouble_eq_doubledouble_add_doubledouble(&head_t1, &tail_t1, head_t1, tail_t1, head_t2, tail_t2);
              head_tmp2[0] = head_t1;
              tail_tmp2[0] = tail_t1;
              /* Imaginary part */
              compute_doubledouble_eq_double_mul_double(&head_t1, &tail_t1, c_elem[1], beta_i[0]);
              compute_doubledouble_eq_double_mul_double(&head_t2, &tail_t2, c_elem[0], beta_i[1]);
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
            c_i[cij] = head_tmp1[0];
            c_i[cij + 1] = head_tmp1[1];
          }
        }

      }

      FPU_FIX_STOP;

      break;
    }
  }
}
#endif
