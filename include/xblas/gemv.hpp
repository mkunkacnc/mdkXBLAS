#ifndef XBLAS_GEMV_HPP
#define XBLAS_GEMV_HPP

#include "blas_enum.h"
#include "impl.hpp"

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

template<int do_conj,
         int need_alpha,
         int need_beta,
         typename TmpType,
         typename PrdType,
         typename T,
         typename A,
         typename X,
         typename N,
         typename IdxType>
constexpr void gemv_impl(T alpha,
                         const A *a,
                         const X *x,
                         N incx,
                         T beta,
                         T *y,
                         N incy,
                         IdxType kx,
                         IdxType ky,
                         IdxType lenx,
                         IdxType leny,
                         IdxType incai,
                         IdxType incaij)
{
  IdxType ai = 0;
  IdxType iy = ky;
  for (IdxType i = 0; i < leny; i++) {
    PrdType sum = impl::zero_v<PrdType>;
    IdxType aij = ai;
    IdxType jx = kx;
    for (IdxType j = 0; j < lenx; j++) {
      A a_elem = impl::Conj_h<do_conj>::func(a[aij]);
      PrdType prod = impl::mul<PrdType>(a_elem, x[jx]);
      sum += prod;
      aij += incaij;
      jx += incx;
    }

    if constexpr (need_beta == 0) {
      if constexpr (need_alpha == 1) {
        y[iy] = impl::to<T>(sum);
      } else {
        TmpType tmp1 = impl::mul<TmpType>(sum, alpha);
        y[iy] = impl::to<T>(tmp1);
      }
    } else {
      TmpType tmp1 = impl::mul<TmpType>(sum, alpha);
      TmpType tmp2 = impl::mul<TmpType>(y[iy], beta);
      tmp1 += tmp2;
      y[iy] = impl::to<T>(tmp1);
    }

    ai += incai;
    iy += incy;
  }
} /* end XBLAS::impl::gemv_impl */

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
constexpr void gemv(blas_order_type order,
                    blas_trans_type trans,
                    N m,
                    N n,
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
 * Computes y = alpha * A * x + beta * y, where A is a general matrix.
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
 * y      (input/output) T*
 *
 * incy   (input) N
 *        The stride for vector y.
 *
 */
{
  static const char *routine_name = "XBLAS::gemv";

  using PrdType = impl::get_inner_type_t<A, X, TmpType>;

  FPU_FIX_DECL;

  /* all error calls */
  if (m < 0)
    BLAS_error(routine_name, -3, m, nullptr);
  else if (n <= 0)
    BLAS_error(routine_name, -4, n, nullptr);
  else if (incx == 0)
    BLAS_error(routine_name, -9, incx, nullptr);
  else if (incy == 0)
    BLAS_error(routine_name, -12, incy, nullptr);

  if ((order == blas_colmajor && lda < m) ||
      (order == blas_rowmajor && lda < n)) {
    BLAS_error(routine_name, -7, lda, nullptr);
  }

  IdxType lenx, leny;
  IdxType incai, incaij;
  if ((order == blas_rowmajor) && (trans == blas_no_trans)) {
    lenx = n;
    leny = m;
    incai = lda;
    incaij = 1;
  } else if ((order == blas_rowmajor) && (trans != blas_no_trans)) {
    lenx = m;
    leny = n;
    incai = 1;
    incaij = lda;
  } else if ((order == blas_colmajor) && (trans == blas_no_trans)) {
    lenx = n;
    leny = m;
    incai = 1;
    incaij = lda;
  } else { /* colmajor and blas_trans */
    lenx = m;
    leny = n;
    incai = lda;
    incaij = 1;
  }

  IdxType kx;
  if (incx > 0)
    kx = 0;
  else
    kx = (1 - lenx) * incx;

  IdxType ky;
  if (incy > 0)
    ky = 0;
  else
    ky = (1 - leny) * incy;

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  /* No extra-precision needed for alpha = 0 */
  if (alpha == T(0)) {
    if (beta == T(0)) {
      IdxType iy = ky;
      for (IdxType i = 0; i < leny; i++) {
        y[iy] = T(0);
        iy += incy;
      }
    } else if (beta != T(0)) {
      IdxType iy = ky;
      for (IdxType i = 0; i < leny; i++) {
        TmpType tmp1 = impl::mul<TmpType>(y[iy], beta);
        y[iy] = impl::to<T>(tmp1);
        iy += incy;
      }
    }
  } else {
    if constexpr (impl::is_complex_v<A>) {
      if (trans == blas_conj_trans) {
        /* if beta = 0, we can save m multiplies: y = alpha*A*x */
        if (beta == T(0)) {
          /* save m more multiplies if alpha = 1 */
          if (alpha == T(1)) {
            impl::gemv_impl<1,  1, 0, TmpType, PrdType>(alpha, a, x, incx, beta, y, incy,
                                                        kx, ky, lenx, leny, incai, incaij);
          } else {
            impl::gemv_impl<1, -1, 0, TmpType, PrdType>(alpha, a, x, incx, beta, y, incy,
                                                        kx, ky, lenx, leny, incai, incaij);
          }
        } else {
          /* the most general form, y = alpha*A*x + beta*y */
          impl::gemv_impl<1, -1, -1, TmpType, PrdType>(alpha, a, x, incx, beta, y, incy,
                                                       kx, ky, lenx, leny, incai, incaij);
        }
      } else {
        /* if beta = 0, we can save m multiplies: y = alpha*A*x */
        if (beta == T(0)) {
          /* save m more multiplies if alpha = 1 */
          if (alpha == T(1)) {
            impl::gemv_impl<0,  1, 0, TmpType, PrdType>(alpha, a, x, incx, beta, y, incy,
                                                        kx, ky, lenx, leny, incai, incaij);
          } else {
            impl::gemv_impl<0, -1, 0, TmpType, PrdType>(alpha, a, x, incx, beta, y, incy,
                                                        kx, ky, lenx, leny, incai, incaij);
          }
        } else {
          /* the most general form, y = alpha*A*x + beta*y */
          impl::gemv_impl<0, -1, -1, TmpType, PrdType>(alpha, a, x, incx, beta, y, incy,
                                                       kx, ky, lenx, leny, incai, incaij);
        }
      }
    } else {
      /* if beta = 0, we can save m multiplies: y = alpha*A*x */
      if (beta == T(0)) {
        /* save m more multiplies if alpha = 1 */
        if (alpha == T(1)) {
          impl::gemv_impl<0,  1, 0, TmpType, PrdType>(alpha, a, x, incx, beta, y, incy,
                                                      kx, ky, lenx, leny, incai, incaij);
        } else {
          impl::gemv_impl<0, -1, 0, TmpType, PrdType>(alpha, a, x, incx, beta, y, incy,
                                                      kx, ky, lenx, leny, incai, incaij);
        }
      } else {
        /* the most general form, y = alpha*A*x + beta*y */
        impl::gemv_impl<0, -1, -1, TmpType, PrdType>(alpha, a, x, incx, beta, y, incy,
                                                     kx, ky, lenx, leny, incai, incaij);
      }
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::gemv */

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
constexpr void gemv_x(blas_order_type order,
                      blas_trans_type trans,
                      N m,
                      N n,
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
 * Computes y = alpha * A * x + beta * y, where A is a general matrix.
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
  static const char *routine_name = "XBLAS::gemv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::gemv<T, A, X, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_double:
    XBLAS::gemv<T, A, X, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_indigenous:
    XBLAS::gemv<T, A, X, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  case blas_prec_extra:
    XBLAS::gemv<T, A, X, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    break;
  default:
    BLAS_error(routine_name, -13, prec, nullptr);
    break;
  }
} /* end XBLAS::gemv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_GEMV_HPP
