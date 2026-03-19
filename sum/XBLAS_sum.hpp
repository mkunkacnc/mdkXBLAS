#ifndef XBLAS_SUM_HPP
#define XBLAS_SUM_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void sum(N n,
                   const T *x,
                   N incx,
                   T *sum)
/*
 * Purpose
 * =======
 *
 * This routine computes the summation:
 *
 *     sum <- SUM_{i=0, n-1} x[i].
 *
 * Arguments
 * =========
 *
 * n     (input) N
 *       The length of vector x.
 *
 * x     (input) const T*
 *       Array of length n.
 *
 * incx  (input) N
 *       The stride used to access components x[i].
 *
 * sum   (output) T*
 *
 */
{
  static const char routine_name[] = "XBLAS::sum";

  FPU_FIX_DECL;

  /* Test the input parameters. */
  if (n < 0)
    BLAS_error(routine_name, -1, n, nullptr);
  if (incx == 0)
    BLAS_error(routine_name, -3, incx, nullptr);

  /* Immediate return. */
  if (n <= 0) {
    *sum = impl::zero_v<T>;
    return;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  TmpType tmp = impl::zero_v<TmpType>;

  IdxType xi;
  if (incx < 0)
    xi = -(n - 1) * incx;
  else
    xi = 0;

  for (IdxType i = 0; i < n; i++, xi += incx) {
    tmp += x[xi];
  }
  *sum = impl::to<T>(tmp);

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::sum */

//-----------------

template<typename T,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
requires (impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void sum_x(N n,
                     const T *x,
                     N incx,
                     T *sum,
                     blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine computes the summation:
 *
 *     sum <- SUM_{i=0, n-1} x[i].
 *
 * Arguments
 * =========
 *
 * n     (input) N
 *       The length of vector x.
 *
 * x     (input) const T*
 *       Array of length n.
 *
 * incx  (input) N
 *       The stride used to access components x[i].
 *
 * sum   (output) T*
 *
 * prec  (input) blas_prec_type
 *       Specifies the internal precision to be used.
 *       = blas_prec_single: single precision.
 *       = blas_prec_double: double precision.
 *       = blas_prec_extra : anything at least 1.5 times as accurate
 *                           than double, and wider than 80-bits.
 *                           We use double-double in our implementation.
 *
 */
{
//static const char routine_name[] = "XBLAS::sum_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::sum<T, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(n, x, incx, sum);
    break;
  case blas_prec_double:
    XBLAS::sum<T, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(n, x, incx, sum);
    break;
  case blas_prec_indigenous:
    XBLAS::sum<T, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(n, x, incx, sum);
    break;
  case blas_prec_extra:
    XBLAS::sum<T, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(n, x, incx, sum);
    break;
  }
} /* end XBLAS::sum_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_SUM_HPP
