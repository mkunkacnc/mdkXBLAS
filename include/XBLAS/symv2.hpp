#ifndef XBLAS_SYMV2_HPP
#define XBLAS_SYMV2_HPP

#include "hemv2.hpp"

//---------------
namespace XBLAS {
//---------------

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
constexpr void symv2(blas_order_type order,
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
 * where A is a symmetric matrix.
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
 * y       (input/output) T*
 *         Vector y.
 *
 * incy    (input) N
 *         Stride for vector y.
 *
 */
{
  /* Routine name */
  static const char *routine_name = "XBLAS::symv2";

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
  impl::hemv2_impl<0, 0, TmpType, PrdType>(n, alpha, a, x_head, x_tail, incx, beta, y, incy,
                                           xi0, yi0, incai, incaij, incaij2);

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::symv2 */

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
constexpr void symv2_x(blas_order_type order,
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
 * where A is a symmetric matrix.
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
 * y       (input/output) T*
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
  static const char *routine_name = "XBLAS::symv2_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::symv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::symv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::symv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::symv2<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, n, alpha, a, lda, x_head, x_tail, incx, beta, y, incy);
    break;
  default:
    BLAS_error(routine_name, -13, prec, nullptr);
    break;
  }
} /* end XBLAS::symv2_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_SYMV_HPP
