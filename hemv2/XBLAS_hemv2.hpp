#ifndef XBLAS_HEMV2_HPP
#define XBLAS_HEMV2_HPP

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
constexpr void hemv2(blas_order_type order,
                     blas_uplo_type uplo,
                     IdxType n,
                     T alpha,
                     const A *a,
                     IdxType lda,
                     const X *x_head,
                     const X *x_tail,
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
 * n       (input) IdxType
 *         Dimension of A and size of vectors x, y.
 *
 * alpha   (input) T
 *
 * a       (input) const A*
 *         Matrix A.
 *
 * lda     (input) IdxType
 *         Leading dimension of matrix A.
 *
 * x_head  (input) const X*
 *         Vector x_head
 *
 * x_tail  (input) const X*
 *         Vector x_tail
 *
 * incx    (input) IdxType
 *         Stride for vector x.
 *
 * beta    (input) T
 *
 * y       (input) T*
 *         Vector y.
 *
 * incy    (input) IdxType
 *         Stride for vector y.
 *
 */
{
  /* Routine name */
  const char routine_name[] = "XBLAS::hemv2";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  IdxType i, j;
  IdxType xi, yi, xi0, yi0;
  IdxType aij, ai;
  IdxType incai;
  IdxType incaij, incaij2;

  const A *a_i = a;
  const X *x_head_i = x_head;
  const X *x_tail_i = x_tail;
  T *y_i = y;
  T alpha_i = alpha;
  T beta_i = beta;
  A a_elem;
  X x_elem;
  T y_elem;
  impl::inner_type_t<A> diag_elem;
  PrdType prod1;
  PrdType prod2;
  PrdType sum1;
  PrdType sum2;
  TmpType tmp1;
  TmpType tmp2;
  TmpType tmp3;

  /* Test for no-op */
  if (n <= 0) {
    return;
  }
  if (alpha_i == T(0) && beta_i == T(1)) {
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

  xi0 = (incx > 0) ? 0 : ((-n + 1) * incx);
  yi0 = (incy > 0) ? 0 : ((-n + 1) * incy);

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  /* The most general form,   y <--- alpha * A * (x_head + x_tail) + beta * y   */
  if (uplo == blas_lower) {
    for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
      sum1 = impl::zero_v<PrdType>;
      sum2 = impl::zero_v<PrdType>;
      for (j = 0, aij = ai, xi = xi0; j < i; j++, aij += incaij, xi += incx) {
        a_elem = a_i[aij];
        x_elem = x_head_i[xi];
        prod1 = impl::mul<PrdType>(a_elem, x_elem);
        sum1 = sum1 + prod1;
        x_elem = x_tail_i[xi];
        prod2 = impl::mul<PrdType>(a_elem, x_elem);
        sum2 = sum2 + prod2;
      }
      diag_elem = std::real(a_i[aij]);
      x_elem = x_head_i[xi];
      prod1 = impl::mul<PrdType>(diag_elem, x_elem);
      sum1 = sum1 + prod1;
      x_elem = x_tail_i[xi];
      prod2 = impl::mul<PrdType>(diag_elem, x_elem);
      sum2 = sum2 + prod2;
      j++;
      aij += incaij2;
      xi += incx;
      for (; j < n; j++, aij += incaij2, xi += incx) {
        a_elem = impl::Conj::func(a_i[aij]);
        x_elem = x_head_i[xi];
        prod1 = impl::mul<PrdType>(a_elem, x_elem);
        sum1 = sum1 + prod1;
        x_elem = x_tail_i[xi];
        prod2 = impl::mul<PrdType>(a_elem, x_elem);
        sum2 = sum2 + prod2;
      }
      sum1 = sum1 + sum2;
      tmp1 = impl::mul<TmpType>(sum1, alpha_i);
      y_elem = y_i[yi];
      tmp2 = impl::mul<TmpType>(y_elem, beta_i);
      tmp3 = tmp1 + tmp2;
      y_i[yi] = impl::to<T>(tmp3);
    }
  } else {
    /* uplo == blas_upper */
    for (i = 0, yi = yi0, ai = 0; i < n; i++, yi += incy, ai += incai) {
      sum1 = impl::zero_v<PrdType>;
      sum2 = impl::zero_v<PrdType>;
      for (j = 0, aij = ai, xi = xi0; j < i; j++, aij += incaij, xi += incx) {
        a_elem = impl::Conj::func(a_i[aij]);
        x_elem = x_head_i[xi];
        prod1 = impl::mul<PrdType>(a_elem, x_elem);
        sum1 = sum1 + prod1;
        x_elem = x_tail_i[xi];
        prod2 = impl::mul<PrdType>(a_elem, x_elem);
        sum2 = sum2 + prod2;
      }
      diag_elem = std::real(a_i[aij]);
      x_elem = x_head_i[xi];
      prod1 = impl::mul<PrdType>(diag_elem, x_elem);
      sum1 = sum1 + prod1;
      x_elem = x_tail_i[xi];
      prod2 = impl::mul<PrdType>(diag_elem, x_elem);
      sum2 = sum2 + prod2;
      j++;
      aij += incaij2;
      xi += incx;
      for (; j < n; j++, aij += incaij2, xi += incx) {
        a_elem = a_i[aij];
        x_elem = x_head_i[xi];
        prod1 = impl::mul<PrdType>(a_elem, x_elem);
        sum1 = sum1 + prod1;
        x_elem = x_tail_i[xi];
        prod2 = impl::mul<PrdType>(a_elem, x_elem);
        sum2 = sum2 + prod2;
      }
      sum1 = sum1 + sum2;
      tmp1 = impl::mul<TmpType>(sum1, alpha_i);
      y_elem = y_i[yi];
      tmp2 = impl::mul<TmpType>(y_elem, beta_i);
      tmp3 = tmp1 + tmp2;
      y_i[yi] = impl::to<T>(tmp3);
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }

} /* end XBLAS::hemv2 */

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
constexpr void hemv2_x(blas_order_type order,
                       blas_uplo_type uplo,
                       IdxType n,
                       T alpha,
                       const A *a,
                       IdxType lda,
                       const X *x_head,
                       const X *x_tail,
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
 * n       (input) IdxType
 *         Dimension of A and size of vectors x, y.
 *
 * alpha   (input) T
 *
 * a       (input) const A*
 *         Matrix A.
 *
 * lda     (input) IdxType
 *         Leading dimension of matrix A.
 *
 * x_head  (input) const X*
 *         Vector x_head
 *
 * x_tail  (input) const X*
 *         Vector x_tail
 *
 * incx    (input) IdxType
 *         Stride for vector x.
 *
 * beta    (input) T
 *
 * y       (input) T*
 *         Vector y.
 *
 * incy    (input) IdxType
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
//static const char routine_name[] = "XBLAS::hemv2_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::hemv2<T, A, X, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::hemv2<T, A, X, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::hemv2<T, A, X, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::hemv2<T, A, X, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  }
} /* end XBLAS::hemv2_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_HEMV_HPP
