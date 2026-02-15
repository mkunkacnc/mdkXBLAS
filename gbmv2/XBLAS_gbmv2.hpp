#ifndef XBLAS_GBMV2_HPP
#define XBLAS_GBMV2_HPP

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
constexpr void gbmv2(blas_order_type order,
                     blas_trans_type trans,
                     IdxType m,
                     IdxType n,
                     IdxType kl,
                     IdxType ku,
                     T alpha,
                     const A *a,
                     IdxType lda,
                     const X *head_x,
                     const X *tail_x,
                     IdxType incx,
                     T beta,
                     T *y,
                     IdxType incy)
/*
 * Purpose
 * =======
 *
 * This routines computes the matrix product:
 *
 *     y  <-  alpha * op(A) * (x_head + x_tail) + beta * y
 *
 * where
 *
 *  A is a m x n banded matrix
 *  x is a n x 1 vector
 *  y is a m x 1 vector
 *  alpha and beta are scalars
 *
 * Arguments
 * =========
 *
 * order   (input) blas_order_type
 *         Order of AB; row or column major
 *
 * trans   (input) blas_trans_type
 *         Transpose of AB; no trans,
 *           trans, or conjugate trans
 *
 * m       (input) IdxType
 *         Dimension of AB
 *
 * n       (input) IdxType
 *         Dimension of AB and the length of vector x and z
 *
 * kl      (input) IdxType
 *         Number of lower diagonals of AB
 *
 * ku      (input) IdxType
 *         Number of upper diagonals of AB
 *
 * alpha   (input) T
 *
 * AB      (input) const A*
 *
 * lda     (input) IdxType
 *         Leading dimension of AB
 *           lda >= ku + kl + 1
 *
 * head_x
 * tail_x  (input) const X*
 *
 * incx    (input) IdxType
 *         The stride for vector x.
 *
 * beta    (input) T
 *
 * y       (input/output) T*
 *
 * incy    (input) IdxType
 *         The stride for vector y.
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
  static const char routine_name[] = "XBLAS::gbmv2";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

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
    BLAS_error(routine_name, -12, incx, NULL);
  if (incy == 0)
    BLAS_error(routine_name, -15, incy, NULL);

  if (m == 0 || n == 0)
    return;
  if (alpha == T(0) && beta == T(1))
    return;

  IdxType lenx, leny;
  if (trans == blas_no_trans) {
    lenx = n;
    leny = m;
  } else {
    lenx = m;
    leny = n;
  }

  IdxType ix0 = (incx > 0) ? 0 : -(lenx - 1) * incx;
  IdxType iy0 = (incy > 0) ? 0 : -(leny - 1) * incy;

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  /* if alpha = 0, return y = y*beta (not implemented as a special case?) */

  IdxType astart, incai1, incai2, incaij, lbound, rbound, ra;
  if (order == blas_colmajor && trans == blas_no_trans) {
    astart = ku;
    incai1 = 1;
    incai2 = lda;
    incaij = lda - 1;
    lbound = kl;
    rbound = n - ku - 1;
    ra = ku;
  } else if (order == blas_colmajor && trans != blas_no_trans) {
    astart = ku;
    incai1 = lda - 1;
    incai2 = lda;
    incaij = 1;
    lbound = ku;
    rbound = m - kl - 1;
    ra = kl;
  } else if (order == blas_rowmajor && trans == blas_no_trans) {
    astart = kl;
    incai1 = lda - 1;
    incai2 = lda;
    incaij = 1;
    lbound = kl;
    rbound = n - ku - 1;
    ra = ku;
  } else { /* rowmajor and blas_trans */
    astart = kl;
    incai1 = 1;
    incai2 = lda;
    incaij = lda - 1;
    lbound = ku;
    rbound = m - kl - 1;
    ra = kl;
  }

  IdxType la = 0;
  IdxType ai = astart;
  IdxType iy = iy0;

  for (IdxType i = 0; i < leny; ++i) {
    PrdType sum1 = impl::zero_v<PrdType>;
    PrdType sum2 = impl::zero_v<PrdType>;
    IdxType aij = ai;
    IdxType jx = ix0;

    if constexpr (impl::is_complex_v<A>) {
      if (trans != blas_conj_trans) {
        for (IdxType j = ra - la; j >= 0; --j) {
          sum1 += impl::mul<PrdType>(a[aij], head_x[jx]);
          sum2 += impl::mul<PrdType>(a[aij], tail_x[jx]);
          aij += incaij;
          jx += incx;
        }
      } else {
        for (IdxType j = ra - la; j >= 0; --j) {
          A a_elem = impl::Conj::func(a[aij]);
          sum1 += impl::mul<PrdType>(a_elem, head_x[jx]);
          sum2 += impl::mul<PrdType>(a_elem, tail_x[jx]);
          aij += incaij;
          jx += incx;
        }
      }
    } else {
      for (IdxType j = ra - la; j >= 0; --j) {
        sum1 += impl::mul<PrdType>(a[aij], head_x[jx]);
        sum2 += impl::mul<PrdType>(a[aij], tail_x[jx]);
        aij += incaij;
        jx += incx;
      }
    }

    TmpType tmp1 = impl::mul<TmpType>(sum1, alpha);
    TmpType tmp2 = impl::mul<TmpType>(sum2, alpha);
    TmpType tmp3 = tmp1 + tmp2;
    TmpType tmp4 = impl::mul<TmpType>(beta, y[iy]);
    y[iy] = impl::add<T>(tmp4, tmp3);

    iy += incy;
    if (i >= lbound) {
      ix0 += incx;
      ai += incai2;
      ++la;
    } else {
      ai += incai1;
    }
    if (i < rbound) {
      ++ra;
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::gbmv2 */

//-----------------

template<typename T,
         typename A,
         typename X,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void gbmv2_x(blas_order_type order,
                       blas_trans_type trans,
                       IdxType m,
                       IdxType n,
                       IdxType kl,
                       IdxType ku,
                       T alpha,
                       const A *a,
                       IdxType lda,
                       const X *head_x,
                       const X *tail_x,
                       IdxType incx,
                       T beta,
                       T *y,
                       IdxType incy,
                       blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routines computes the matrix product:
 *
 *     y  <-  alpha * op(A) * (x_head + x_tail) + beta * y
 *
 * where
 *
 *  A is a m x n banded matrix
 *  x is a n x 1 vector
 *  y is a m x 1 vector
 *  alpha and beta are scalars
 *
 * Arguments
 * =========
 *
 * order   (input) blas_order_type
 *         Order of AB; row or column major
 *
 * trans   (input) blas_trans_type
 *         Transpose of AB; no trans,
 *           trans, or conjugate trans
 *
 * m       (input) IdxType
 *         Dimension of AB
 *
 * n       (input) IdxType
 *         Dimension of AB and the length of vector x and z
 *
 * kl      (input) IdxType
 *         Number of lower diagonals of AB
 *
 * ku      (input) IdxType
 *         Number of upper diagonals of AB
 *
 * alpha   (input) T
 *
 * AB      (input) const A*
 *
 * lda     (input) IdxType
 *         Leading dimension of AB
 *           lda >= ku + kl + 1
 *
 * head_x
 * tail_x  (input) const X*
 *
 * incx    (input) IdxType
 *         The stride for vector x.
 *
 * beta    (input) T
 *
 * y       (input/output) T*
 *
 * incy    (input) IdxType
 *         The stride for vector y.
 *
 * prec    (input) blas_prec_type
 *         Specifies the internal precision to be used.
 *         = blas_prec_single: single precision.
 *         = blas_prec_double: double precision.
 *         = blas_prec_extra : anything at least 1.5 times as accurate
 *                             than double, and wider than 80-bits.
 *                             We use double-double in our implementation.
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
//static const char routine_name[] = "XBLAS::gbmv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::gbmv2<T, A, X, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, trans, m, n, kl, ku, alpha, a, lda, head_x, tail_x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::gbmv2<T, A, X, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, trans, m, n, kl, ku, alpha, a, lda, head_x, tail_x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::gbmv2<T, A, X, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, trans, m, n, kl, ku, alpha, a, lda, head_x, tail_x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::gbmv2<T, A, X, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, trans, m, n, kl, ku, alpha, a, lda, head_x, tail_x, incx, beta, y, incy);
    break;
  }
} /* end XBLAS::gbmv2_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_GBMV2_HPP
