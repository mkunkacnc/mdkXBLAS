#ifndef XBLAS_HPMV_HPP
#define XBLAS_HPMV_HPP

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
         int need_alpha,
         int need_beta,
         int need_column_major,
         typename TmpType,
         typename PrdType,
         typename T,
         typename A,
         typename X,
         typename N,
         typename IdxType>
constexpr void hpmv_impl(N n,
                         T alpha,
                         const A *ap,
                         const X *x,
                         N incx,
                         T beta,
                         T *y,
                         N incy,
                         IdxType x_start,
                         IdxType y_start)
{
  IdxType y_index = y_start;
  IdxType ap_start = 0;
  for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
    IdxType x_index = x_start;
    IdxType ap_index = ap_start;
    PrdType rowsum = impl::zero_v<PrdType>;

    for (IdxType step = 0; step < matrix_row; step++) {
      A matval = impl::Conj_h<do_conj1>::func(ap[ap_index]);
      PrdType rowtmp = impl::mul<PrdType>(matval, x[x_index]);
      rowsum += rowtmp;
      if constexpr (need_column_major) {
        ap_index += 1;
      } else {
        ap_index += (n - step - 1);
      }
      x_index += incx;
    }

    /* need to do diagonal element without referencing the imaginary part */
    auto matval_r = impl::Real::func(ap[ap_index]);
    PrdType rowtmp = impl::mul<PrdType>(matval_r, x[x_index]);
    rowsum += rowtmp;
    if constexpr (need_column_major) {
      ap_index += (matrix_row + 1);
    } else {
      ap_index += 1;
    }
    x_index += incx;

    for (IdxType step = matrix_row + 1; step < n; step++) {
      A matval = impl::Conj_h<do_conj2>::func(ap[ap_index]);
      PrdType rowtmp = impl::mul<PrdType>(matval, x[x_index]);
      rowsum += rowtmp;
      if constexpr (need_column_major) {
        ap_index += (step + 1);
      } else {
        ap_index += 1;
      }
      x_index += incx;
    }

    if constexpr (need_alpha == 1) {
      if constexpr (need_beta == 0) {
        y[y_index] = impl::to<T>(rowsum);
      } else {
        TmpType tmp1 = rowsum;
        TmpType tmp2 = impl::mul<TmpType>(y[y_index], beta);
        tmp1 += tmp2;
        y[y_index] = impl::to<T>(tmp1);
      }
    } else {
      TmpType tmp1 = impl::mul<TmpType>(rowsum, alpha);
      TmpType tmp2 = impl::mul<TmpType>(beta, y[y_index]);
      tmp2 += tmp1;
      y[y_index] = impl::to<T>(tmp2);
    }

    y_index += incy;
    if constexpr (need_column_major) {
      ap_start += (matrix_row + 1);
    } else {
      ap_start += 1;
    }
  }
} /* end XBLAS::hpmv */

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
constexpr void hpmv(blas_order_type order,
                    blas_uplo_type uplo,
                    N n,
                    T alpha,
                    const A *ap,
                    const X *x,
                    N incx,
                    T beta,
                    T *y,
                    N incy)
/*
 * Purpose
 * =======
 *
 * Computes y = alpha * ap * x + beta * y, where ap is a hermitian
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Order of ap; row or column major
 *
 * uplo   (input) blas_uplo_type
 *        Whether ap is upper or lower
 *
 * n      (input) N
 *        Dimension of ap and the length of vector x
 *
 * alpha  (input) T
 *
 * ap     (input) const A*
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
 */
{
  static const char *routine_name = "XBLAS::hpmv";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  /* Check for error conditions. */
  if (order != blas_colmajor && order != blas_rowmajor)
    BLAS_error(routine_name, -1, order, nullptr);
  if (uplo != blas_upper && uplo != blas_lower)
    BLAS_error(routine_name, -2, uplo, nullptr);
  if (n < 0)
    BLAS_error(routine_name, -3, n, nullptr);
  if (incx == 0)
    BLAS_error(routine_name, -7, incx, nullptr);
  if (incy == 0) {
    BLAS_error(routine_name, -10, incy, nullptr);
  }

  if (n == 0)
    return;
  if (alpha == T(0) && beta == T(1))
    return;

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  IdxType x_start;
  if (incx < 0)
    x_start = (-n + 1) * incx;
  else
    x_start = 0;

  IdxType y_start;
  if (incy < 0)
    y_start = (-n + 1) * incy;
  else
    y_start = 0;

  blas_order_type order_i;
  if (uplo == blas_lower)
    order_i = (order == blas_rowmajor) ? blas_colmajor : blas_rowmajor;
  else
    order_i = order;

  if (alpha == T(0)) {
    IdxType y_index = y_start;
    for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
      TmpType tmp2 = impl::mul<TmpType>(beta, y[y_index]);
      y[y_index] = impl::to<T>(tmp2);
      y_index += incy;
    }
  } else {
    if (order_i == blas_rowmajor) {
      if (alpha == T(1)) {
        if (beta == T(0)) {
          if (uplo == blas_upper) {
            impl::hpmv_impl<1, 0, 1, 0, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } else { /* if uplo == ... */
            impl::hpmv_impl<0, 1, 1, 0, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } /* end if uplo == blas_upper ... */
        } else {
          if (uplo == blas_upper) {
            impl::hpmv_impl<1, 0, 1, -1, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } else {                /* if uplo == ... */
            impl::hpmv_impl<0, 1, 1, -1, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } /* end if uplo == blas_upper ... */
        }
      } else {
        if (beta == T(0)) {
          if (uplo == blas_upper) {
            impl::hpmv_impl<1, 0, -1, 0, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } else { /* if uplo == ... */
            impl::hpmv_impl<0, 1, -1, 0, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } /* end if uplo == blas_upper ... */
        } else {
          if (uplo == blas_upper) {
            impl::hpmv_impl<1, 0, -1, -1, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } else { /* if uplo == ... */
            impl::hpmv_impl<0, 1, -1, -1, 0, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } /* end if uplo == blas_upper ... */
        }
      }
    } else {
      if (alpha == T(1)) {
        if (beta == T(0)) {
          if (uplo == blas_upper) {
            impl::hpmv_impl<1, 0, 1, 0, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } else { /* if uplo == ... */
            impl::hpmv_impl<0, 1, 1, 0, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } /* end if uplo == blas_upper ... */
        } else {
          if (uplo == blas_upper) {
            impl::hpmv_impl<1, 0, 1, -1, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } else { /* if uplo == ... */
            impl::hpmv_impl<0, 1, 1, -1, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } /* end if uplo == blas_upper ... */
        }
      } else {
        if (beta == T(0)) {
          if (uplo == blas_upper) {
            impl::hpmv_impl<1, 0, -1, 0, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } else { /* if uplo == ... */
            impl::hpmv_impl<0, 1, -1, 0, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } /* end if uplo == blas_upper ... */
        } else {
          if (uplo == blas_upper) {
            impl::hpmv_impl<1, 0, -1, -1, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } else { /* if uplo == ... */
            impl::hpmv_impl<0, 1, -1, -1, 1, TmpType, PrdType>(n, alpha, ap, x, incx, beta, y, incy, x_start, y_start);
          } /* end if uplo == blas_upper ... */
        }
      }
    }
  } /* end alpha != 0 */

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::hpmv */

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
constexpr void hpmv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      N n,
                      T alpha,
                      const A *ap,
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
 * Computes y = alpha * ap * x + beta * y, where ap is a hermitian
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Order of ap; row or column major
 *
 * uplo   (input) blas_uplo_type
 *        Whether ap is upper or lower
 *
 * n      (input) N
 *        Dimension of ap and the length of vector x
 *
 * alpha  (input) T
 *
 * ap     (input) const A*
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
 */
{
  static const char *routine_name = "XBLAS::hpmv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::hpmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::hpmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::hpmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::hpmv<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, n, alpha, ap, x, incx, beta, y, incy);
    break;
  default:
    BLAS_error(routine_name, -11, prec, nullptr);
    break;
  }
} /* end XBLAS::hpmv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_HPMV_HPP
