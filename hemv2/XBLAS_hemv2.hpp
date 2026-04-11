#ifndef XBLAS_HEMV2_HPP
#define XBLAS_HEMV2_HPP

#include "blas_enum.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

template<int do_conj1,
         int do_conj2,
         typename TmpType,
         typename PrdType,
         typename T,
         typename A,
         typename X,
         typename N,
         typename IdxType>
constexpr void hemv2_impl(N n,
                          T alpha,
                          const A *a,
                          const X *x_head,
                          const X *x_tail,
                          N incx,
                          T beta,
                          T *y,
                          N incy,
                          IdxType xi0,
                          IdxType yi0,
                          IdxType incai,
                          IdxType incaij,
                          IdxType incaij2)
{
  IdxType yi = yi0;
  IdxType ai = 0;
  for (IdxType i = 0; i < n; i++) {
    PrdType sum1 = impl::zero_v<PrdType>;
    PrdType sum2 = impl::zero_v<PrdType>;
    IdxType aij = ai;
    IdxType xi = xi0;
    IdxType j = 0;

    for (; j < i; j++) {
      A a_elem = impl::Conj_h<do_conj1>::func(a[aij]);
      PrdType prod1 = impl::mul<PrdType>(a_elem, x_head[xi]);
      sum1 += prod1;
      PrdType prod2 = impl::mul<PrdType>(a_elem, x_tail[xi]);
      sum2 += prod2;
      aij += incaij;
      xi += incx;
    }

    impl::inner_type_t<A> diag_elem = impl::Real::func(a[aij]);
    PrdType prod1 = impl::mul<PrdType>(diag_elem, x_head[xi]);
    sum1 += prod1;
    PrdType prod2 = impl::mul<PrdType>(diag_elem, x_tail[xi]);
    sum2 += prod2;
    j++;
    aij += incaij2;
    xi += incx;

    for (; j < n; j++) {
      A a_elem = impl::Conj_h<do_conj2>::func(a[aij]);
      PrdType prod1 = impl::mul<PrdType>(a_elem, x_head[xi]);
      sum1 += prod1;
      PrdType prod2 = impl::mul<PrdType>(a_elem, x_tail[xi]);
      sum2 += prod2;
      aij += incaij2;
      xi += incx;
    }

    sum1 += sum2;
    TmpType tmp1 = impl::mul<TmpType>(sum1, alpha);
    TmpType tmp2 = impl::mul<TmpType>(y[yi], beta);
    tmp1 += tmp2;
    y[yi] = impl::to<T>(tmp1);

    yi += incy;
    ai += incai;
  }
} /* end XBLAS::impl::hemv2_impl */

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
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void hemv2(blas_order_type order,
                     blas_uplo_type uplo,
                     N n,
                     T alpha,
                     const A *a,
                     N lda,
                     const X *x_head,
                     const X *x_tail,
                     N incx,
                     T beta,
                     T *y,
                     N incy)
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
 * order   (input) blas_order_type
 *         Storage format of input symmetric matrix A.
 *
 * uplo    (input) blas_uplo_type
 *         Determines which half of matrix A (upper or lower triangle)
 *           is accessed.
 *
 * n       (input) N
 *         Dimension of A and size of vectors x, y.
 *
 * alpha   (input) T
 *
 * a       (input) const A*
 *         Matrix A.
 *
 * lda     (input) N
 *         Leading dimension of matrix A.
 *
 * x_head  (input) const X*
 *         Vector x_head
 *
 * x_tail  (input) const X*
 *         Vector x_tail
 *
 * incx    (input) N
 *         Stride for vector x.
 *
 * beta    (input) T
 *
 * y       (input) T*
 *         Vector y.
 *
 * incy    (input) N
 *         Stride for vector y.
 *
 */
{
  /* Routine name */
  const char routine_name[] = "XBLAS::hemv2";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  /* Check for error conditions. */
  if (n < 0) {
    BLAS_error(routine_name, -3, n, nullptr);
  }
  if (lda < n) {
    BLAS_error(routine_name, -6, n, nullptr);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -9, incx, nullptr);
  }
  if (incy == 0) {
    BLAS_error(routine_name, -12, incy, nullptr);
  }

  /* Test for no-op */
  if (n == 0) {
    return;
  }
  if (alpha == T(0) && beta == T(1)) {
    return;
  }

  IdxType incai, incaij, incaij2;
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

  IdxType xi0 = (incx > 0) ? 0 : ((-n + 1) * incx);
  IdxType yi0 = (incy > 0) ? 0 : ((-n + 1) * incy);

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  // TODO: simplifications based on alpha, beta = 0 or 1?

  /* The most general form,   y <--- alpha * A * (x_head + x_tail) + beta * y   */
  if (uplo == blas_lower) {
    impl::hemv2_impl<0, 1, TmpType, PrdType>(n, alpha, a, x_head, x_tail, incx, beta, y, incy,
                                             xi0, yi0, incai, incaij, incaij2);
  } else {
    /* uplo == blas_upper */
    impl::hemv2_impl<1, 0, TmpType, PrdType>(n, alpha, a, x_head, x_tail, incx, beta, y, incy,
                                             xi0, yi0, incai, incaij, incaij2);
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::hemv2 */

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
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void hemv2_x(blas_order_type order,
                       blas_uplo_type uplo,
                       N n,
                       T alpha,
                       const A *a,
                       N lda,
                       const X *x_head,
                       const X *x_tail,
                       N incx,
                       T beta,
                       T *y,
                       N incy,
                       blas_prec_type prec)
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
 * order   (input) blas_order_type
 *         Storage format of input symmetric matrix A.
 *
 * uplo    (input) blas_uplo_type
 *         Determines which half of matrix A (upper or lower triangle)
 *           is accessed.
 *
 * n       (input) N
 *         Dimension of A and size of vectors x, y.
 *
 * alpha   (input) T
 *
 * a       (input) const A*
 *         Matrix A.
 *
 * lda     (input) N
 *         Leading dimension of matrix A.
 *
 * x_head  (input) const X*
 *         Vector x_head
 *
 * x_tail  (input) const X*
 *         Vector x_tail
 *
 * incx    (input) N
 *         Stride for vector x.
 *
 * beta    (input) T
 *
 * y       (input) T*
 *         Vector y.
 *
 * incy    (input) N
 *         Stride for vector y.
 *
 * prec    (input) blas_prec_type
 *         Specifies the internal precision to be used.
 *         = blas_prec_single: single precision.
 *         = blas_prec_double: double precision.
 *         = blas_prec_extra : anything at least 1.5 times as accurate
 *                             than double, and wider than 80-bits.
 *                             We use double-double in our implementation.
 *
 */
{
  static const char *routine_name = "XBLAS::hemv2_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::hemv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::hemv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::hemv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::hemv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  default:
    BLAS_error(routine_name, -13, prec, nullptr);
    break;
  }
} /* end XBLAS::hemv2_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_HEMV_HPP
