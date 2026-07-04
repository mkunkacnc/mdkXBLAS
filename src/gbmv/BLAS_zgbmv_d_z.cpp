#include "blas_extended.h"
#include "gbmv.hpp"

void BLAS_zgbmv_d_z(enum blas_order_type order,
                    enum blas_trans_type trans,
                    int m,
                    int n,
                    int kl,
                    int ku,
                    const void *alpha,
                    const double *a,
                    int lda,
                    const void *x,
                    int incx,
                    const void *beta,
                    void *y,
                    int incy)
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
 * alpha  (input) const void*
 *
 * AB     (input) const double*
 *
 * lda    (input) int
 *        Leading dimension of AB
 *          lda >= ku + kl + 1
 *
 * x      (input) const void*
 *
 * incx   (input) int
 *        The stride for vector x.
 *
 * beta   (input) const void*
 *
 * y      (input/output) void*
 *
 * incy   (input) int
 *        The stride for vector y.
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
//static const char *routine_name = "BLAS_zgbmv_d_z";
  XBLAS::gbmv(order,
              trans,
              m,
              n,
              kl,
              ku,
              *static_cast<const std::complex<double> *>(alpha),
              a,
              lda,
              static_cast<const std::complex<double> *>(x),
              incx,
              *static_cast<const std::complex<double> *>(beta),
              static_cast<std::complex<double> *>(y),
              incy);
} /* end BLAS_zgbmv_d_z */
