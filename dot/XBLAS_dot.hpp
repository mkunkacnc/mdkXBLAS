#ifndef XBLAS_DOT_HPP
#define XBLAS_DOT_HPP

#include "blas_extended_private.h"
#include "blas_enum.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename X,
         typename Y,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<X, T> &&
          impl::size_le_v<Y, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void dot(blas_conj_type conj,
                   IdxType n,
                   T alpha,
                   const X *x,
                   IdxType incx,
                   T beta,
                   const Y *y,
                   IdxType incy,
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
 * conj   (input) enum blas_conj_type
 *        When x and y are complex vectors, specifies whether vector
 *        components x[i] are used unconjugated or conjugated.
 *
 * n      (input) IdxType
 *        The length of vectors x and y.
 *
 * alpha  (input) T
 *
 * x      (input) const X*
 *        Array of length n.
 *
 * incx   (input) IdxType
 *        The stride used to access components x[i].
 *
 * beta   (input) T
 *
 * y      (input) const Y*
 *        Array of length n.
 *
 * incy   (input) IdxType
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
    BLAS_error(routine_name, -2, n, NULL);
  else if (incx == 0)
    BLAS_error(routine_name, -5, incx, NULL);
  else if (incy == 0)
    BLAS_error(routine_name, -8, incy, NULL);

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
      for (IdxType i = 0; i < n; ++i) {
        PrdType prod = impl::mul<PrdType>(impl::Conj::func(x[ix]), y[iy]); /* prod = x[i]*y[i] */
        sum = sum + prod;                                                  /* sum = sum+prod */
        ix += incx;
        iy += incy;
      } /* endfor */
    } else {
      /* do not conjugate */
      for (IdxType i = 0; i < n; ++i) {
        PrdType prod = impl::mul<PrdType>(x[ix], y[iy]); /* prod = x[i]*y[i] */
        sum = sum + prod;                                /* sum = sum+prod */
        ix += incx;
        iy += incy;
      } /* endfor */
    }
  } else {
    for (IdxType i = 0; i < n; ++i) {
      PrdType prod = impl::mul<PrdType>(x[ix], y[iy]); /* prod = x[i]*y[i] */
      sum = sum + prod;                                /* sum = sum+prod */
      ix += incx;
      iy += incy;
    } /* endfor */
  }

  TmpType tmp1 = impl::mul<TmpType>(sum, alpha); /* tmp1 = sum*alpha */
  TmpType tmp2 = impl::mul<TmpType>(r_v, beta);  /* tmp2 = r*beta */
  tmp1 += tmp2;                                  /* tmp1 = tmp1+tmp2 */
  *r = impl::to<T>(tmp1);                        /* r = tmp1 */

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
}

//-----------------

template<typename T,
         typename X,
         typename Y,
         typename IdxType = int>
requires std::signed_integral<IdxType>
constexpr void dot_x(blas_conj_type conj,
                     IdxType n,
                     T alpha,
                     const X *x,
                     IdxType incx,
                     T beta,
                     const Y *y,
                     IdxType incy,
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
 * conj   (input) enum blas_conj_type
 *        When x and y are complex vectors, specifies whether vector
 *        components x[i] are used unconjugated or conjugated.
 *
 * n      (input) IdxType
 *        The length of vectors x and y.
 *
 * alpha  (input) T
 *
 * x      (input) const X*
 *        Array of length n.
 *
 * incx   (input) IdxType
 *        The stride used to access components x[i].
 *
 * beta   (input) T
 *
 * y      (input) const Y*
 *        Array of length n.
 *
 * incy   (input) IdxType
 *        The stride used to access components y[i].
 *
 * r      (input/output) T*
 *
 * prec   (input) enum blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 */
{
//static const char routine_name[] = "XBLAS::dot_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::dot<T, X, Y, impl::internal_precision_t<T, blas_prec_single>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  case blas_prec_double:
    XBLAS::dot<T, X, Y, impl::internal_precision_t<T, blas_prec_double>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  case blas_prec_indigenous:
    XBLAS::dot<T, X, Y, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  case blas_prec_extra:
    XBLAS::dot<T, X, Y, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(conj, n, alpha, x, incx, beta, y, incy, r);
    break;
  }
}

//-----------------
} // namespace XBLAS
//-----------------

#endif // XBLAS_DOT_HPP
