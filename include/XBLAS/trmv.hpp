#ifndef XBLAS_TRMV_HPP
#define XBLAS_TRMV_HPP

#include "blas_enum.h"
#include "impl.hpp"

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

template<int do_conj,
         int need_unit_diag,
         typename TmpType,
         typename T,
         typename A,
         typename N,
         typename IdxType>
constexpr void trmv_impl(N n,
                         T alpha,
                         const A *t,
                         T *x,
                         IdxType xj0,
                         IdxType inc_x,
                         IdxType inc_ti,
                         IdxType inc_tij)
{
  IdxType ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
  IdxType tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
  for (IdxType i = 0; i < n; i++) {
    TmpType sum = impl::zero_v<TmpType>;

    IdxType xj = xj0;
    IdxType tij = ti + tij0;
    for (IdxType j = i; j < n - 1; j++) {
      A t_elem = impl::Conj_h<do_conj>::func(t[tij]);
      TmpType prod = impl::mul<TmpType>(x[xj], t_elem);
      sum += prod;
      xj += inc_x;
      tij += inc_tij;
    }

    if constexpr (need_unit_diag) {
      sum = impl::add<TmpType>(sum, x[xj]);
    } else {
      A t_elem = impl::Conj_h<do_conj>::func(t[tij]);
      TmpType prod = impl::mul<TmpType>(x[xj], t_elem);
      sum += prod;
    }

    if (alpha == T(1)) {
      x[xj] = impl::to<T>(sum);
    } else {
      TmpType tmp = impl::mul<TmpType>(sum, alpha);
      x[xj] = impl::to<T>(tmp);
    }

    ti += inc_ti;
  }
}

//-----------------
} // namespace impl
//-----------------

template<typename T,
         typename A,
         typename N,
         typename TmpType = T,
         typename IdxType = impl::internal_index_type_t<N>>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void trmv(blas_order_type order,
                    blas_uplo_type uplo,
                    blas_trans_type trans,
                    blas_diag_type diag,
                    N n,
                    T alpha,
                    const A *t,
                    N ldt,
                    T *x,
                    N incx)
/*
 * Purpose
 * =======
 *
 * Computes x <-- alpha * T * x, where T is a triangular matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        column major, row major
 *
 * uplo   (input) blas_uplo_type
 *        upper, lower
 *
 * trans  (input) blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) blas_diag_type
 *        unit, non unit
 *
 * n      (input) N
 *        the dimension of T
 *
 * alpha  (input) T
 *
 * t      (input) const A*
 *        Triangular matrix
 *
 * ldt    (input) N
 *        Leading dimension of T
 *
 * x      (input/output) T*
 *        Array of length n.
 *
 * incx   (input) N
 *        The stride used to access components x[i].
 *
 */
{
  static const char *routine_name = "XBLAS::trmv";

  FPU_FIX_DECL;

  /* all error calls */
  if (order != blas_rowmajor && order != blas_colmajor) {
    BLAS_error(routine_name, -1, order, nullptr);
  } else if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, nullptr);
  } else if (trans != blas_trans && trans != blas_no_trans && trans != blas_conj_trans) {
    BLAS_error(routine_name, -3, trans, nullptr);
  } else if (diag != blas_non_unit_diag && diag != blas_unit_diag) {
    BLAS_error(routine_name, -4, diag, nullptr);
  } else if (n < 0) {
    BLAS_error(routine_name, -5, n, nullptr);
  } else if (ldt < n) {
    BLAS_error(routine_name, -8, ldt, nullptr);
  } else if (incx == 0) {
    BLAS_error(routine_name, -10, incx, nullptr);
  }

  IdxType inc_x, inc_ti, inc_tij;
  if (trans == blas_no_trans) {
    if (uplo == blas_upper) {
      inc_x = -incx;
      if (order == blas_rowmajor) {
        inc_ti = ldt;
        inc_tij = -1;
      } else {
        inc_ti = 1;
        inc_tij = -ldt;
      }
    } else {
      inc_x = incx;
      if (order == blas_rowmajor) {
        inc_ti = -ldt;
        inc_tij = 1;
      } else {
        inc_ti = -1;
        inc_tij = ldt;
      }
    }
  } else {
    if (uplo == blas_upper) {
      inc_x = incx;
      if (order == blas_rowmajor) {
        inc_ti = -1;
        inc_tij = ldt;
      } else {
        inc_ti = -ldt;
        inc_tij = 1;
      }
    } else {
      inc_x = -incx;
      if (order == blas_rowmajor) {
        inc_ti = 1;
        inc_tij = -ldt;
      } else {
        inc_ti = ldt;
        inc_tij = -1;
      }
    }
  }

  IdxType xj0 = (inc_x > 0 ? 0 : -(n - 1) * inc_x);

  if (alpha == T(0)) {
    IdxType xj = xj0;
    for (IdxType j = 0; j < n; j++) {
      x[xj] = T(0);
      xj += inc_x;
    }
    return;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if constexpr (impl::is_complex_v<A> || impl::is_complex_v<T>) {
    if (diag == blas_unit_diag) {
      if (trans == blas_conj_trans) {
        impl::trmv_impl<1, 1, TmpType>(n, alpha, t, x, xj0, inc_x, inc_ti, inc_tij);
      } else {
        impl::trmv_impl<0, 1, TmpType>(n, alpha, t, x, xj0, inc_x, inc_ti, inc_tij);
      }
    } else {
      if (trans == blas_conj_trans) {
        impl::trmv_impl<1, 0, TmpType>(n, alpha, t, x, xj0, inc_x, inc_ti, inc_tij);
      } else {
        impl::trmv_impl<0, 0, TmpType>(n, alpha, t, x, xj0, inc_x, inc_ti, inc_tij);
      }
    }
  } else {
    if (diag == blas_unit_diag) {
      impl::trmv_impl<0, 1, TmpType>(n, alpha, t, x, xj0, inc_x, inc_ti, inc_tij);
    } else {
      impl::trmv_impl<0, 0, TmpType>(n, alpha, t, x, xj0, inc_x, inc_ti, inc_tij);
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::trmv */

//-----------------

template<typename T,
         typename A,
         typename N,
         typename TmpType = T,
         typename IdxType = impl::internal_index_type_t<N>>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<N> &&
          std::signed_integral<IdxType> &&
          sizeof(N) <= sizeof(IdxType))
constexpr void trmv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      blas_trans_type trans,
                      blas_diag_type diag,
                      N n,
                      T alpha,
                      const A *t,
                      N ldt,
                      T *x,
                      N incx,
                      blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes x <-- alpha * T * x, where T is a triangular matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        column major, row major
 *
 * uplo   (input) blas_uplo_type
 *        upper, lower
 *
 * trans  (input) blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) blas_diag_type
 *        unit, non unit
 *
 * n      (input) N
 *        the dimension of T
 *
 * alpha  (input) T
 *
 * t      (input) const A*
 *        Triangular matrix
 *
 * ldt    (input) N
 *        Leading dimension of T
 *
 * x      (input/output) T*
 *        Array of length n.
 *
 * incx   (input) N
 *        The stride used to access components x[i].
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
  static const char *routine_name = "XBLAS::trmv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::trmv<T, A, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_double:
    XBLAS::trmv<T, A, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_indigenous:
    XBLAS::trmv<T, A, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_extra:
    XBLAS::trmv<T, A, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  default:
    BLAS_error(routine_name, -11, prec, nullptr);
    break;
  }
} /* end XBLAS::trmv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_TRMV_HPP
