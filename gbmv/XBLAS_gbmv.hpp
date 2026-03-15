#ifndef XBLAS_GBMV_HPP
#define XBLAS_GBMV_HPP

#include "blas_enum.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

template<int do_conj,
         typename A,
         typename X,
         typename N,
         typename PrdType,
         typename IdxType>
constexpr void gbmv_impl(const A *a,
                         const X *x,
                         N incx,
                         PrdType& sum,
                         IdxType ra,
                         IdxType la,
                         IdxType aij,
                         IdxType jx,
                         IdxType incaij)
{
  for (IdxType j = ra - la; j >= 0; --j) {
    PrdType prod = impl::mul<PrdType>(x[jx], impl::Conj_h<do_conj>::func(a[aij]));
    sum += prod;
    aij += incaij;
    jx += incx;
  }
} /* end XBLAS::impl::gbmv_impl */

//-----------------
} // namespace impl
//-----------------

template<typename T,
         typename A,
         typename X,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType>)
constexpr void gbmv(blas_order_type order,
                    blas_trans_type trans,
                    N m,
                    N n,
                    N kl,
                    N ku,
                    T alpha,
                    const A *a,
                    N lda,
                    const X *x,
                    N incx,
                    T beta,
                    T *y,
                    N incy)
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
 * order  (input) blas_order_type
 *        Order of AB; row or column major
 *
 * trans  (input) blas_trans_type
 *        Transpose of AB; no trans,
 *          trans, or conjugate trans
 *
 * m      (input) N
 *        Dimension of AB
 *
 * n      (input) N
 *        Dimension of AB and the length of vector x
 *
 * kl     (input) N
 *        Number of lower diagonals of AB
 *
 * ku     (input) N
 *        Number of upper diagonals of AB
 *
 * alpha  (input) T
 *
 * AB     (input) const A*
 *
 * lda    (input) N
 *        Leading dimension of AB
 *          lda >= ku + kl + 1
 *
 * x      (input) const X*
 *
 * incx   (input) N
 *        The stride for vector x.
 *
 * beta   (input) T
 *
 * y      (input/output) T*
 *
 * incy   (input) N
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

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  if (order != blas_colmajor && order != blas_rowmajor)
    BLAS_error(routine_name, -1, order, nullptr);
  if (trans != blas_no_trans &&
      trans != blas_trans && trans != blas_conj_trans) {
    BLAS_error(routine_name, -2, trans, nullptr);
  }
  if (m < 0)
    BLAS_error(routine_name, -3, m, nullptr);
  if (n < 0)
    BLAS_error(routine_name, -4, n, nullptr);
  if (kl < 0 || kl >= m)
    BLAS_error(routine_name, -5, kl, nullptr);
  if (ku < 0 || ku >= n)
    BLAS_error(routine_name, -6, ku, nullptr);
  if (lda < kl + ku + 1)
    BLAS_error(routine_name, -9, lda, nullptr);
  if (incx == 0)
    BLAS_error(routine_name, -11, incx, nullptr);
  if (incy == 0)
    BLAS_error(routine_name, -14, incy, nullptr);

  if (m == 0 || n == 0 || (alpha == T(0) && beta == T(1)))
    return;

  IdxType lenx, leny;
  if (trans == blas_no_trans) {
    lenx = n;
    leny = m;
  } else { /* change back */
    lenx = m;
    leny = n;
  }

  IdxType kx;
  if (incx < 0) {
    kx = -(lenx - 1) * incx;
  } else {
    kx = 0;
  }

  IdxType ky;
  if (incy < 0) {
    ky = -(leny - 1) * incy;
  } else {
    ky = 0;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  /* if alpha = 0, return y = y*beta (not implemented as a special case?) */

  IdxType astart, incai1, incai2, incaij, rbound, lbound, ra;
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
  IdxType iy = ky;

  for (IdxType i = 0; i < leny; ++i) {
    PrdType sum = impl::zero_v<PrdType>;
    IdxType aij = ai;
    IdxType jx = kx;

    if constexpr (impl::is_complex_v<A>) {
      if (trans == blas_conj_trans) {
        impl::gbmv_impl<1>(a, x, incx, sum, ra, la, aij, jx, incaij);
      } else {
        impl::gbmv_impl<0>(a, x, incx, sum, ra, la, aij, jx, incaij);
      }
    } else {
      impl::gbmv_impl<0>(a, x, incx, sum, ra, la, aij, jx, incaij);
    }

    TmpType tmp1 = impl::mul<TmpType>(sum, alpha);
    TmpType tmp2 = impl::mul<TmpType>(beta, y[iy]);
    y[iy] = impl::add<T>(tmp1, tmp2);

    iy += incy;
    if (i >= lbound) {
      kx += incx;
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
} /* end XBLAS::gbmv */

//-----------------

template<typename T,
         typename A,
         typename X,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType>)
constexpr void gbmv_x(blas_order_type order,
                      blas_trans_type trans,
                      N m,
                      N n,
                      N kl,
                      N ku,
                      T alpha,
                      const A *a,
                      N lda,
                      const X *x,
                      N incx,
                      T beta,
                      T *y,
                      N incy,
                      blas_prec_type prec)
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
 * order  (input) blas_order_type
 *        Order of AP; row or column major
 *
 * trans  (input) blas_trans_type
 *        Transpose of AB; no trans,
 *          trans, or conjugate trans
 *
 * m      (input) N
 *        Dimension of AB
 *
 * n      (input) N
 *        Dimension of AB and the length of vector x
 *
 * kl     (input) N
 *        Number of lower diagonals of AB
 *
 * ku     (input) N
 *        Number of upper diagonals of AB
 *
 * alpha  (input) T
 *
 * AB     (input) const A*
 *
 * lda    (input) N
 *        Leading dimension of AB
 *          lda >= ku + kl + 1
 *
 * x      (input) const X*
 *
 * incx   (input) N
 *        The stride for vector x.
 *
 * beta   (input) T
 *
 * y      (input/output) T*
 *
 * incy   (input) N
 *        The stride for vector y.
 *
 * prec   (input) blas_prec_type
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
  static const char routine_name[] = "XBLAS::gbmv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::gbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::gbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::gbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::gbmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    break;
  default:
    BLAS_error(routine_name, -15, prec, nullptr);
    break;
  }
} /* end XBLAS::gbmv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_GBMV_HPP
