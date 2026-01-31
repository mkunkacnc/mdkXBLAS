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
requires (sizeof(X) <= sizeof(T) &&
          sizeof(Y) <= sizeof(T) &&
          sizeof(TmpType) >= sizeof(T) &&
          std::signed_integral<IdxType>)
constexpr void dot(blas_conj_type /*conj*/,
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

  IdxType i, ix = 0, iy = 0;
  T *r_i = r;
  const X *x_i = x;
  const Y *y_i = y;
  T alpha_i = alpha;
  T beta_i = beta;
  X x_ii;
  Y y_ii;
  T r_v;
  TmpType prod;
  TmpType sum;
  TmpType tmp1;
  TmpType tmp2;

  /* Test the input parameters. */
  if (n < 0)
    BLAS_error(routine_name, -2, n, NULL);
  else if (incx == 0)
    BLAS_error(routine_name, -5, incx, NULL);
  else if (incy == 0)
    BLAS_error(routine_name, -8, incy, NULL);

  /* Immediate return. */
  if ((beta_i == T(1)) && (n == 0 || (alpha_i == T(0))))
    return;

  r_v = r_i[0];
  sum = T(0);

  if (incx < 0)
    ix = (-n + 1) * incx;
  if (incy < 0)
    iy = (-n + 1) * incy;

  for (i = 0; i < n; ++i) {
    x_ii = x_i[ix];
    y_ii = y_i[iy];

    prod = impl::mul<TmpType>(x_ii, y_ii); /* prod = x[i]*y[i] */
    sum = sum + prod;                      /* sum = sum+prod */
    ix += incx;
    iy += incy;
  } /* endfor */

  tmp1 = sum * alpha_i;   /* tmp1 = sum*alpha */
  tmp2 = r_v * beta_i;    /* tmp2 = r*beta */
  tmp1 = tmp1 + tmp2;     /* tmp1 = tmp1+tmp2 */
  *r = impl::to<T>(tmp1); /* r = tmp1 */
}

//-----------------
} //namespace XBLAS
//-----------------

#endif // XBLAS_DOT_HPP
