#ifndef XBLAS_GE_SUM_MV_HPP
#define XBLAS_GE_SUM_MV_HPP

#include "impl.hpp"

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

template<int need_alpha,
         int need_beta,
         typename TmpType,
         typename PrdType,
         typename T,
         typename A,
         typename X,
         typename N,
         typename IdxType>
constexpr void ge_sum_mv_impl(N m,
                              N n,
                              T alpha,
                              const A *a,
                              const X *x,
                              T beta,
                              const A *b,
                              T *y,
                              IdxType incai,
                              IdxType incaij,
                              IdxType x_starti,
                              IdxType incxi,
                              IdxType incbi,
                              IdxType incbij,
                              IdxType y_starti,
                              IdxType incyi)
{
  IdxType ai = 0;
  IdxType bi = 0;

  for (IdxType i = 0, yi = y_starti; i < m; ++i, yi += incyi) {
    PrdType sumA = impl::zero_v<PrdType>;
    IdxType aij = ai;
    PrdType sumB = impl::zero_v<PrdType>;
    IdxType bij = bi;

    for (IdxType j = 0, xi = x_starti; j < n; ++j, xi += incxi) {
      if (need_alpha != 0) {
        sumA += impl::mul<PrdType>(a[aij], x[xi]);
        aij += incaij;
      }
      if (need_beta != 0) {
        sumB += impl::mul<PrdType>(b[bij], x[xi]);
        bij += incbij;
      }
    }

    /* now put the result into y */
    if constexpr (need_alpha == 0) {
      if constexpr (need_beta == 0) {
        y[yi] = T(0); // not actually used.
      } else if constexpr (need_beta == 1) {
        y[yi] = impl::to<T>(sumB);
      } else {
        TmpType tmp2 = impl::mul<TmpType>(sumB, beta);
        y[yi] = impl::to<T>(tmp2);
      }
    } else if constexpr (need_alpha == 1) {
      if constexpr (need_beta == 0) {
        y[yi] = impl::to<T>(sumA);
      } else {
        TmpType tmp1 = sumA;
        TmpType tmp2 = (need_beta == 1 ? sumB : impl::mul<TmpType>(sumB, beta));
        y[yi] = impl::add<T>(tmp1, tmp2);
      }
    } else {
      if constexpr (need_beta == 0) {
        TmpType tmp1 = impl::mul<TmpType>(sumA, alpha);
        y[yi] = impl::to<T>(tmp1);
      } else {
        TmpType tmp1 = impl::mul<TmpType>(sumA, alpha);
        TmpType tmp2 = (need_beta == 1 ? sumB : impl::mul<TmpType>(sumB, beta));
        y[yi] = impl::add<T>(tmp1, tmp2);
      }
    }

    if constexpr (need_alpha != 0)
      ai += incai;
    if constexpr (need_beta != 0)
      bi += incbi;
  }
} /* end XBLAS::impl::ge_sum_mv_impl */

//-----------------
} // namespace impl
//-----------------

template<typename T,
         typename A,
         typename X,
         typename N,
         typename TmpType = T,
         typename IdxType = impl::internal_index_type_t<N>>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void ge_sum_mv(blas_order_type order,
                         N m,
                         N n,
                         T alpha,
                         const A *a,
                         N lda,
                         const X *x,
                         N incx,
                         T beta,
                         const A *b,
                         N ldb,
                         T *y,
                         N incy)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * A * x + beta * B * y,
 *     where A, B are general matrices.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Order of A; row or column major
 *
 * m      (input) N
 *        Row Dimension of A, B, length of output vector y
 *
 * n      (input) N
 *        Column Dimension of A, B and the length of vector x
 *
 * alpha  (input) T
 *
 * A      (input) const A*
 *
 * lda    (input) N
 *        Leading dimension of A
 *
 * x      (input) const X*
 *
 * incx   (input) N
 *        The stride for vector x.
 *
 * beta   (input) T
 *
 * b      (input) const A*
 *
 * ldb    (input) N
 *        Leading dimension of B
 *
 * y      (input/output) T*
 *
 * incy   (input) N
 *        The stride for vector y.
 *
 */
{
  static const char *routine_name = "XBLAS::ge_sum_mv";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  /* m is number of rows */
  /* n is number of columns */

  if (m == 0 || n == 0)
    return;

  /* all error calls */
  IdxType lda_min, incai, incbi, incaij, incbij;
  if (order == blas_rowmajor) {
    lda_min = n;
    incai = lda; /* row stride */
    incbi = ldb;
    incaij = 1; /* column stride */
    incbij = 1;
  } else if (order == blas_colmajor) {
    lda_min = m;
    incai = 1; /*row stride */
    incbi = 1;
    incaij = lda; /* column stride */
    incbij = ldb;
  } else {
    /* error, order not blas_colmajor not blas_rowmajor */
    BLAS_error(routine_name, -1, order, nullptr);
    return;
  }

  if (m < 0)
    BLAS_error(routine_name, -2, m, nullptr);
  else if (n < 0)
    BLAS_error(routine_name, -3, n, nullptr);
  if (lda < lda_min)
    BLAS_error(routine_name, -6, lda, nullptr);
  else if (ldb < lda_min)
    BLAS_error(routine_name, -11, ldb, nullptr);
  else if (incx == 0)
    BLAS_error(routine_name, -8, incx, nullptr);
  else if (incy == 0)
    BLAS_error(routine_name, -13, incy, nullptr);

  IdxType incxi = incx;
  IdxType incyi = incy;

  IdxType x_starti;
  if (incxi > 0)
    x_starti = 0;
  else
    x_starti = (1 - n) * incxi;

  IdxType y_starti;
  if (incyi > 0)
    y_starti = 0;
  else
    y_starti = (1 - m) * incyi;

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if (alpha == T(0)) {
    if (beta == T(0)) {
      /* alpha, beta are 0 */
      for (IdxType i = 0, yi = y_starti; i < m; ++i, yi += incyi) {
        y[yi] = T(0);
      }
    } else if (beta == T(1)) {
      /* alpha is 0, beta is 1 */
      impl::ge_sum_mv_impl<0, 1, TmpType, PrdType>(m, n, alpha, a, x, beta, b, y, incai, incaij, x_starti, incxi, incbi, incbij, y_starti, incyi);
    } else {
      /* alpha is 0, beta is other */
      impl::ge_sum_mv_impl<0, -1, TmpType, PrdType>(m, n, alpha, a, x, beta, b, y, incai, incaij, x_starti, incxi, incbi, incbij, y_starti, incyi);
    }
  } else if (alpha == T(1)) {
    if (beta == T(0)) {
      /* alpha is 1, beta is 0 */
      impl::ge_sum_mv_impl<1, 0, TmpType, PrdType>(m, n, alpha, a, x, beta, b, y, incai, incaij, x_starti, incxi, incbi, incbij, y_starti, incyi);
    } else if (beta == T(1)) {
      /* alpha is 1, beta is 1 */
      impl::ge_sum_mv_impl<1, 1, TmpType, PrdType>(m, n, alpha, a, x, beta, b, y, incai, incaij, x_starti, incxi, incbi, incbij, y_starti, incyi);
    } else {
      /* alpha is 1, beta is other */
      impl::ge_sum_mv_impl<1, -1, TmpType, PrdType>(m, n, alpha, a, x, beta, b, y, incai, incaij, x_starti, incxi, incbi, incbij, y_starti, incyi);
    }
  } else {
    if (beta == T(0)) {
      /* alpha is other, beta is 0 */
      impl::ge_sum_mv_impl<-1, 0, TmpType, PrdType>(m, n, alpha, a, x, beta, b, y, incai, incaij, x_starti, incxi, incbi, incbij, y_starti, incyi);
    } else if (beta == T(1)) {
      /* alpha is other, beta is 1 */
      impl::ge_sum_mv_impl<-1, 1, TmpType, PrdType>(m, n, alpha, a, x, beta, b, y, incai, incaij, x_starti, incxi, incbi, incbij, y_starti, incyi);
    } else {
      /* most general form, alpha, beta are other */
      impl::ge_sum_mv_impl<-1, -1, TmpType, PrdType>(m, n, alpha, a, x, beta, b, y, incai, incaij, x_starti, incxi, incbi, incbij, y_starti, incyi);
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::ge_sum_mv */

//-----------------

template<typename T,
         typename A,
         typename X,
         typename N,
         typename TmpType = T,
         typename IdxType = impl::internal_index_type_t<N>>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<X, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void ge_sum_mv_x(blas_order_type order,
                           N m,
                           N n,
                           T alpha,
                           const A *a,
                           N lda,
                           const X *x,
                           N incx,
                           T beta,
                           const A *b,
                           N ldb,
                           T *y,
                           N incy,
                           blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * A * x + beta * B * y,
 *     where A, B are general matrices.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Order of A; row or column major
 *
 * m      (input) N
 *        Row Dimension of A, B, length of output vector y
 *
 * n      (input) N
 *        Column Dimension of A, B and the length of vector x
 *
 * alpha  (input) T
 *
 * A      (input) const A*
 *
 * lda    (input) N
 *        Leading dimension of A
 *
 * x      (input) const X*
 *
 * incx   (input) N
 *        The stride for vector x.
 *
 * beta   (input) T
 *
 * b      (input) const A*
 *
 * ldb    (input) N
 *        Leading dimension of B
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
 */
{
  static const char *routine_name = "XBLAS::ge_sum_mv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::ge_sum_mv<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, m, n, alpha, a, lda, x, incx, beta, b, ldb, y, incy);
    break;
  case blas_prec_double:
    XBLAS::ge_sum_mv<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, m, n, alpha, a, lda, x, incx, beta, b, ldb, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::ge_sum_mv<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, m, n, alpha, a, lda, x, incx, beta, b, ldb, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::ge_sum_mv<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, m, n, alpha, a, lda, x, incx, beta, b, ldb, y, incy);
    break;
  default:
    BLAS_error(routine_name, -14, prec, nullptr);
    break;
  }
} /* end XBLAS::ge_sum_mv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_GE_SUM_MV_HPP
