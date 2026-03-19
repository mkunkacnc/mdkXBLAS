#ifndef XBLAS_DOT_HPP
#define XBLAS_DOT_HPP

#include "blas_enum.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

template<int do_conj,
         typename X,
         typename Y,
         typename N,
         typename PrdType,
         typename IdxType>
constexpr void dot_impl(N n,
                        const X *x,
                        N incx,
                        const Y *y,
                        N incy,
                        IdxType ix,
                        IdxType iy,
                        PrdType& sum)
{
  for (IdxType i = 0; i < n; ++i) {
    sum += impl::mul<PrdType>(impl::Conj_h<do_conj>::func(x[ix]), y[iy]);
    ix += incx;
    iy += incy;
  }
} /* end XBLAS::impl::dot_impl */

//-----------------
} // namespace impl
//-----------------

template<typename T,
         typename X,
         typename Y,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<X, T> &&
          impl::size_le_v<Y, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void dot(blas_conj_type conj,
                   N n,
                   T alpha,
                   const X *x,
                   N incx,
                   T beta,
                   const Y *y,
                   N incy,
                   T *r)
/*
 * Purpose
 * =======
 *
 * This routine computes the inner product:
 *
 *     r <- beta * r + alpha * SUM_{i=0, n-1} x[i] * y[i].
 *
 * Arguments
 * =========
 *
 * conj   (input) blas_conj_type
 *        When x and y are complex vectors, specifies whether vector
 *          components x[i] are used unconjugated or conjugated.
 *
 * n      (input) N
 *        The length of vectors x and y.
 *
 * alpha  (input) T
 *
 * x      (input) const X*
 *        Array of length n.
 *
 * incx   (input) N
 *        The stride used to access components x[i].
 *
 * beta   (input) T
 *
 * y      (input) const Y*
 *        Array of length n.
 *
 * incy   (input) N
 *        The stride used to access components y[i].
 *
 * r      (input/output) T*
 *
 */
{
  static const char routine_name[] = "XBLAS::dot";

  using PrdType = impl::get_inner_type_t<X, Y, TmpType>;

  FPU_FIX_DECL;

  /* Test the input parameters. */
  if (n < 0)
    BLAS_error(routine_name, -2, n, nullptr);
  else if (incx == 0)
    BLAS_error(routine_name, -5, incx, nullptr);
  else if (incy == 0)
    BLAS_error(routine_name, -8, incy, nullptr);

  /* Immediate return. */
  if (beta == T(1) && (n == 0 || alpha == T(0)))
    return;

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  T r_v = *r;
  PrdType sum = impl::zero_v<PrdType>;

  IdxType ix = 0;
  if (incx < 0)
    ix = (-n + 1) * incx;
  IdxType iy = 0;
  if (incy < 0)
    iy = (-n + 1) * incy;

  if constexpr (impl::is_complex_v<X>) {
    if (conj == blas_conj) {
      impl::dot_impl<1>(n, x, incx, y, incy, ix, iy, sum);
    } else {
      /* do not conjugate */
      impl::dot_impl<0>(n, x, incx, y, incy, ix, iy, sum);
    }
  } else {
    impl::dot_impl<0>(n, x, incx, y, incy, ix, iy, sum);
  }

  TmpType tmp1 = impl::mul<TmpType>(sum, alpha);
  TmpType tmp2 = impl::mul<TmpType>(r_v, beta);
  *r = impl::add<T>(tmp1, tmp2);

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::dot */

//-----------------

template<typename T,
         typename X,
         typename Y,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<X, T> &&
          impl::size_le_v<Y, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void dot_x(blas_conj_type conj,
                     N n,
                     T alpha,
                     const X *x,
                     N incx,
                     T beta,
                     const Y *y,
                     N incy,
                     T *r,
                     blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine computes the inner product:
 *
 *     r <- beta * r + alpha * SUM_{i=0, n-1} x[i] * y[i].
 *
 * Arguments
 * =========
 *
 * conj   (input) blas_conj_type
 *        When x and y are complex vectors, specifies whether vector
 *          components x[i] are used unconjugated or conjugated.
 *
 * n      (input) N
 *        The length of vectors x and y.
 *
 * alpha  (input) T
 *
 * x      (input) const X*
 *        Array of length n.
 *
 * incx   (input) N
 *        The stride used to access components x[i].
 *
 * beta   (input) T
 *
 * y      (input) const Y*
 *        Array of length n.
 *
 * incy   (input) N
 *        The stride used to access components y[i].
 *
 * r      (input/output) T*
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
  static const char routine_name[] = "XBLAS::dot_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::dot<T, X, Y, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  case blas_prec_double:
    XBLAS::dot<T, X, Y, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  case blas_prec_indigenous:
    XBLAS::dot<T, X, Y, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  case blas_prec_extra:
    XBLAS::dot<T, X, Y, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  default:
    BLAS_error(routine_name, -10, prec, nullptr);
    break;
  }
} /* end XBLAS::dot_x */

//-----------------
} // namespace XBLAS
//-----------------

#endif // XBLAS_DOT_HPP
