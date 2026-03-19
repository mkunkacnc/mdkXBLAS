#ifndef XBLAS_TRMV_HPP
#define XBLAS_TRMV_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename A,
         typename N,
         typename TmpType = T,
         typename IdxType = N>
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
  static const char routine_name[] = "XBLAS::trmv";

  FPU_FIX_DECL;

  IdxType i, j;
  IdxType xj, xj0;
  IdxType ti, tij, tij0;

  IdxType inc_ti, inc_tij;
  IdxType inc_x;

  const A *t_i = t;
  T *x_i = x;
  T alpha_i = alpha;

  A t_elem;
  T x_elem;
  TmpType prod;
  TmpType sum;
  TmpType tmp;

  /* all error calls */
  if ((order != blas_rowmajor && order != blas_colmajor) ||
      (uplo != blas_upper && uplo != blas_lower) ||
      (trans != blas_trans &&
       trans != blas_no_trans &&
       trans != blas_conj_trans) ||
      (diag != blas_non_unit_diag && diag != blas_unit_diag) ||
      (ldt < n) || (incx == 0)) {
    BLAS_error(routine_name, 0, 0, nullptr);
  } else if (n <= 0) {
    BLAS_error(routine_name, -4, n, nullptr);
  } else if (incx == 0) {
    BLAS_error(routine_name, -9, incx, nullptr);
  }

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

  xj0 = (inc_x > 0 ? 0 : -(n - 1) * inc_x);
  if (alpha_i == T(0)) {
    xj = xj0;
    for (j = 0; j < n; j++) {
      x_i[xj] = T(0);
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
        ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
        tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
        for (i = 0; i < n; i++) {
          sum = impl::zero_v<TmpType>;

          xj = xj0;
          tij = ti + tij0;
          for (j = i; j < (n - 1); j++) {
            t_elem = impl::Conj::func(t_i[tij]);
            x_elem = x_i[xj];
            prod = impl::mul<TmpType>(x_elem, t_elem);
            sum = sum + prod;
            xj += inc_x;
            tij += inc_tij;
          }

          x_elem = x_i[xj];
          sum = impl::add<TmpType>(sum, x_elem);

          if (alpha_i == T(1)) {
            x_i[xj] = impl::to<T>(sum);
          } else {
            tmp = impl::mul<TmpType>(sum, alpha_i);
            x_i[xj] = impl::to<T>(tmp);
          }

          ti += inc_ti;
        }
      } else {
        ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
        tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
        for (i = 0; i < n; i++) {
          sum = impl::zero_v<TmpType>;

          xj = xj0;
          tij = ti + tij0;
          for (j = i; j < (n - 1); j++) {
            t_elem = t_i[tij];
            x_elem = x_i[xj];
            prod = impl::mul<TmpType>(x_elem, t_elem);
            sum = sum + prod;
            xj += inc_x;
            tij += inc_tij;
          }

          x_elem = x_i[xj];
          sum = impl::add<TmpType>(sum, x_elem);

          if (alpha_i == T(1)) {
            x_i[xj] = impl::to<T>(sum);
          } else {
            tmp = impl::mul<TmpType>(sum, alpha_i);
            x_i[xj] = impl::to<T>(tmp);
          }

          ti += inc_ti;
        }
      }
    } else {
      if (trans == blas_conj_trans) {
        ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
        tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
        for (i = 0; i < n; i++) {
          sum = impl::zero_v<TmpType>;

          xj = xj0;
          tij = ti + tij0;
          for (j = i; j < n; j++) {
            t_elem = impl::Conj::func(t_i[tij]);
            x_elem = x_i[xj];
            prod = impl::mul<TmpType>(x_elem, t_elem);
            sum = sum + prod;
            xj += inc_x;
            tij += inc_tij;
          }

          if (alpha_i == T(1)) {
            x_i[xj - inc_x] = impl::to<T>(sum);
          } else {
            tmp = impl::mul<TmpType>(sum, alpha_i);
            x_i[xj - inc_x] = impl::to<T>(tmp);
          }

          ti += inc_ti;
        }
      } else {
        ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
        tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
        for (i = 0; i < n; i++) {
          sum = impl::zero_v<TmpType>;

          xj = xj0;
          tij = ti + tij0;
          for (j = i; j < n; j++) {
            t_elem = t_i[tij];
            x_elem = x_i[xj];
            prod = impl::mul<TmpType>(x_elem, t_elem);
            sum = sum + prod;
            xj += inc_x;
            tij += inc_tij;
          }

          if (alpha_i == T(1)) {
            x_i[xj - inc_x] = impl::to<T>(sum);
          } else {
            tmp = impl::mul<TmpType>(sum, alpha_i);
            x_i[xj - inc_x] = impl::to<T>(tmp);
          }

          ti += inc_ti;
        }
      }
    }
  } else {
    if (diag == blas_unit_diag) {
      ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
      tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
      for (i = 0; i < n; i++) {
        sum = impl::zero_v<TmpType>;

        xj = xj0;
        tij = ti + tij0;
        for (j = i; j < (n - 1); j++) {
          t_elem = t_i[tij];
          x_elem = x_i[xj];
          prod = impl::mul<TmpType>(x_elem, t_elem);
          sum = sum + prod;
          xj += inc_x;
          tij += inc_tij;
        }

        x_elem = x_i[xj];
        sum = sum + x_elem;

        if (alpha_i == T(1)) {
          x_i[xj] = impl::to<T>(sum);
        } else {
          tmp = impl::mul<TmpType>(sum, alpha_i);
          x_i[xj] = impl::to<T>(tmp);
        }

        ti += inc_ti;
      }
    } else {
      ti = (inc_ti > 0 ? 0 : -(n - 1) * inc_ti);
      tij0 = (inc_tij > 0 ? 0 : -(n - 1) * inc_tij);
      for (i = 0; i < n; i++) {
        sum = impl::zero_v<TmpType>;

        xj = xj0;
        tij = ti + tij0;
        for (j = i; j < n; j++) {
          t_elem = t_i[tij];
          x_elem = x_i[xj];
          prod = impl::mul<TmpType>(x_elem, t_elem);
          sum = sum + prod;
          xj += inc_x;
          tij += inc_tij;
        }

        if (alpha_i == T(1)) {
          x_i[xj - inc_x] = impl::to<T>(sum);
        } else {
          tmp = impl::mul<TmpType>(sum, alpha_i);
          x_i[xj - inc_x] = impl::to<T>(tmp);
        }

        ti += inc_ti;
      }
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
         typename IdxType = N>
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
//static const char routine_name[] = "XBLAS::trmv_x";
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
  }
} /* end XBLAS::trmv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_TRMV_HPP
