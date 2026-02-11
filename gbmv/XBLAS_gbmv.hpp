#ifndef XBLAS_GBMV_HPP
#define XBLAS_GBMV_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename A,
         typename X,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void gbmv(blas_order_type order,
                    blas_trans_type trans,
                    IdxType m,
                    IdxType n,
                    IdxType kl,
                    IdxType ku,
                    T alpha,
                    const A *a,
                    IdxType lda,
                    const X *x,
                    IdxType incx,
                    T beta,
                    T *y,
                    IdxType incy)
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
 * order        (input) blas_order_type
 *              Order of AB; row or column major
 *
 * trans        (input) blas_trans_type
 *              Transpose of AB; no trans,
 *              trans, or conjugate trans
 *
 * m            (input) IdxType
 *              Dimension of AB
 *
 * n            (input) IdxType
 *              Dimension of AB and the length of vector x
 *
 * kl           (input) IdxType
 *              Number of lower diagonals of AB
 *
 * ku           (input) IdxType
 *              Number of upper diagonals of AB
 *
 * alpha        (input) T
 *
 * AB           (input) A*
 *
 * lda          (input) int
 *              Leading dimension of AB
 *              lda >= ku + kl + 1
 *
 * x            (input) X*
 *
 * incx         (input) IdxType
 *              The stride for vector x.
 *
 * beta         (input) T
 *
 * y            (input/output) T*
 *
 * incy         (input) IdxType
 *              The stride for vector y.
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
 *  la          index of leftmost elements for a given row
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
  static const char routine_name[] = "XBLAS::gbmv";

  IdxType ky, iy, kx, jx, j, i, rbound, lbound, ra, la, lenx, leny;
  IdxType incaij, aij, incai1, incai2, astart, ai;
  T *y_i = y;
  const A *a_i = a;
  const X *x_i = x;
  T alpha_i = alpha;
  T beta_i = beta;
  TmpType tmp1;
  TmpType tmp2;
  T result;
  TmpType sum;
  TmpType prod;
  T a_elem;
  X x_elem;
  T y_elem;
  FPU_FIX_DECL;

  if (order != blas_colmajor && order != blas_rowmajor)
    BLAS_error(routine_name, -1, order, NULL);
  if (trans != blas_no_trans &&
      trans != blas_trans && trans != blas_conj_trans) {
    BLAS_error(routine_name, -2, trans, NULL);
  }
  if (m < 0)
    BLAS_error(routine_name, -3, m, NULL);
  if (n < 0)
    BLAS_error(routine_name, -4, n, NULL);
  if (kl < 0 || kl >= m)
    BLAS_error(routine_name, -5, kl, NULL);
  if (ku < 0 || ku >= n)
    BLAS_error(routine_name, -6, ku, NULL);
  if (lda < kl + ku + 1)
    BLAS_error(routine_name, -9, lda, NULL);
  if (incx == 0)
    BLAS_error(routine_name, -11, incx, NULL);
  if (incy == 0)
    BLAS_error(routine_name, -14, incy, NULL);

  if ((m == 0) || (n == 0) || (((alpha_i == 0.0) && (beta_i == 1.0))))
    return;

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if (trans == blas_no_trans) {
    lenx = n;
    leny = m;
  } else {                        /* change back */
    lenx = m;
    leny = n;
  }

  if (incx < 0) {
    kx = -(lenx - 1) * incx;
  } else {
    kx = 0;
  }

  if (incy < 0) {
    ky = -(leny - 1) * incy;
  } else {
    ky = 0;
  }

  /* if alpha = 0, return y = y*beta */
  if ((order == blas_colmajor) && (trans == blas_no_trans)) {
    astart = ku;
    incai1 = 1;
    incai2 = lda;
    incaij = lda - 1;
    lbound = kl;
    rbound = n - ku - 1;
    ra = ku;
  } else if ((order == blas_colmajor) && (trans != blas_no_trans)) {
    astart = ku;
    incai1 = lda - 1;
    incai2 = lda;
    incaij = 1;
    lbound = ku;
    rbound = m - kl - 1;
    ra = kl;
  } else if ((order == blas_rowmajor) && (trans == blas_no_trans)) {
    astart = kl;
    incai1 = lda - 1;
    incai2 = lda;
    incaij = 1;
    lbound = kl;
    rbound = n - ku - 1;
    ra = ku;
  } else {                        /* rowmajor and blas_trans */
    astart = kl;
    incai1 = 1;
    incai2 = lda;
    incaij = lda - 1;
    lbound = ku;
    rbound = m - kl - 1;
    ra = kl;
  }

  la = 0;
  ai = astart;
  iy = ky;
  for (i = 0; i < leny; i++) {
    sum = 0.0;
    aij = ai;
    jx = kx;

    for (j = ra - la; j >= 0; j--) {
      x_elem = x_i[jx];
      a_elem = a_i[aij];
      prod = x_elem * a_elem;
      sum = sum + prod;
      aij += incaij;
      jx += incx;
    }

    tmp1 = sum * alpha_i;
    y_elem = y_i[iy];
    tmp2 = beta_i * y_elem;
    result = tmp1 + tmp2;
    y_i[iy] = result;
    iy += incy;
    if (i >= lbound) {
      kx += incx;
      ai += incai2;
      la++;
    } else {
      ai += incai1;
    }
    if (i < rbound) {
      ra++;
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::gbmv */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_GBMV_HPP
