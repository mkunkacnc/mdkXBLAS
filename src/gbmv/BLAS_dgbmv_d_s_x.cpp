#include "blas_extended.h"
#include "gbmv.hpp"

void BLAS_dgbmv_d_s_x(enum blas_order_type order,
                      enum blas_trans_type trans,
                      int m,
                      int n,
                      int kl,
                      int ku,
                      double alpha,
                      const double *a,
                      int lda,
                      const float *x,
                      int incx,
                      double beta,
                      double *y,
                      int incy,
                      enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 *  gbmv computes y = alpha * A * x + beta * y, where
 *
 *  A is a m x n banded matrix
 *  x is a n x 1 vector
 *  y is a m x 1 vector
 *  alpha and beta are scalars
 *
 *
 * Arguments
 * =========
 *
 * order  (input) enum blas_order_type
 *        Order of AP; row or column major
 *
 * trans  (input) enum blas_trans_type
 *        Transpose of AB; no trans,
 *          trans, or conjugate trans
 *
 * m      (input) int
 *        Dimension of AB
 *
 * n      (input) int
 *        Dimension of AB and the length of vector x
 *
 * kl     (input) int
 *        Number of lower diagonals of AB
 *
 * ku     (input) int
 *        Number of upper diagonals of AB
 *
 * alpha  (input) double
 *
 * AB     (input) const double*
 *
 * lda    (input) int
 *        Leading dimension of AB
 *          lda >= ku + kl + 1
 *
 * x      (input) const float*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) double
 *
 * y      (input/output) double*
 *
 * incy   (input) int
 *        The stride for vector y.
 *
 * prec   (input) enum blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 *
 * LOCAL VARIABLES
 * ===============
 *
 *  As an example, these variables are described on the mxn, column
 *  major, banded matrix described in section 2.2.3 of the specification
 *
 *  astart      indexes first element in A where computation begins
 *
 *  incai1      indexes first element in row where row is less than lbound
 *
 *  incai2      indexes first element in row where row exceeds lbound
 *
 *  lbound      denotes the number of rows before  first element shifts
 *
 *  rbound      denotes the columns where there is blank space
 *
 *  ra          index of the rightmost element for a given row
 *
 *  la          index of leftmost  elements for a given row
 *
 *  ra - la     width of a row
 *
 *                        rbound
 *            la   ra    ____|_____
 *             |    |   |          |
 *         |  a00  a01   *    *   *
 * lbound -|  a10  a11  a12   *   *
 *         |  a20  a21  a22  a23  *
 *             *   a31  a32  a33 a34
 *             *    *   a42  a43 a44
 *
 *  Variations on order and transpose have been implemented by modifying these
 *  local variables.
 *
 */
{
//static const char *routine_name = "BLAS_dgbmv_d_s_x";
  XBLAS::gbmv_x(order, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy, prec);
} /* end BLAS_dgbmv_d_s_x */
