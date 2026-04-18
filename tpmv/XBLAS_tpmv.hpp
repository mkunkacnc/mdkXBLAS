#ifndef XBLAS_TPMV_HPP
#define XBLAS_TPMV_HPP

#include "blas_enum.h"
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
constexpr void tpmv(blas_order_type order,
                    blas_uplo_type uplo,
                    blas_trans_type trans,
                    blas_diag_type diag,
                    N n,
                    T alpha,
                    const A *tp,
                    T *x,
                    N incx)
/*
 * Purpose
 * =======
 *
 * Computes x = alpha * tp * x, x = alpha * tp_transpose * x,
 * or x = alpha * tp_conjugate_transpose where tp is a triangular
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Order of tp; row or column major
 *
 * uplo   (input) blas_uplo_type
 *        Whether tp is upper or lower
 *
 * trans  (input) blas_trans_type
 *
 * diag   (input) blas_diag_type
 *        Whether the diagonal entries of tp are 1
 *
 * n      (input) N
 *        Dimension of tp and the length of vector x
 *
 * alpha  (input) T
 *
 * tp     (input) const A*
 *
 * x      (input/output) T*
 *
 * incx   (input) N
 *        The stride for vector x.
 *
 */
{
  static const char *routine_name = "XBLAS::tpmv";

  FPU_FIX_DECL;

  /* Check for error conditions. */
  if (order != blas_colmajor && order != blas_rowmajor) {
    BLAS_error(routine_name, -1, order, nullptr);
  } else if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, nullptr);
  } else if (trans != blas_trans && trans != blas_no_trans && trans != blas_conj_trans) {
    BLAS_error(routine_name, -3, trans, nullptr);
  } else if (diag != blas_non_unit_diag && diag != blas_unit_diag) {
    BLAS_error(routine_name, -4, diag, nullptr);
  } else if (n < 0) {
    BLAS_error(routine_name, -5, n, nullptr);
  } else if (incx == 0) {
    BLAS_error(routine_name, -9, incx, nullptr);
  }

  if (n == 0) {
    return;
  }

  IdxType x_start;
  if (incx < 0)
    x_start = (-n + 1) * incx;
  else
    x_start = 0;

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if ((uplo == blas_upper && trans == blas_no_trans && order == blas_rowmajor) ||
      (uplo == blas_lower && trans != blas_no_trans && order == blas_colmajor)) {

    IdxType tp_index = 0;
    for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
      IdxType x_index = x_start + incx * matrix_row;
      IdxType x_index2 = x_index;
      IdxType col_index = matrix_row;
      TmpType rowsum = impl::zero_v<TmpType>;

      while (col_index < n) {
        TmpType rowtmp;
        if ((diag == blas_unit_diag) && (col_index == matrix_row)) {
          rowtmp = impl::to<TmpType>(x[x_index]);
        } else {
          rowtmp = impl::mul<TmpType>(tp[tp_index], x[x_index]);
        }
        rowsum += rowtmp;
        x_index += incx;
        tp_index++;
        col_index++;
      }
      TmpType result = impl::mul<TmpType>(rowsum, alpha);
      x[x_index2] = impl::to<T>(result);
    }
  } else if ((uplo == blas_upper && trans == blas_no_trans && order == blas_colmajor) ||
             (uplo == blas_lower && trans != blas_no_trans && order == blas_rowmajor)) {

    IdxType tp_start = ((n - 1) * n) / 2;
    IdxType x_index2 = x_start;
    for (IdxType matrix_row = 0; matrix_row < n; matrix_row++) {
      IdxType inctp2 = n - 1;
      IdxType x_index = x_start + incx * (n - 1);
      IdxType tp_index = tp_start + matrix_row;
      IdxType col_index = (n - 1) - matrix_row;
      TmpType rowsum = impl::zero_v<TmpType>;

      while (col_index >= 0) {
        TmpType rowtmp;
        if ((diag == blas_unit_diag) && (col_index == 0)) {
          rowtmp = impl::to<TmpType>(x[x_index]);
        } else {
          rowtmp = impl::mul<TmpType>(tp[tp_index], x[x_index]);
        }
        rowsum += rowtmp;
        x_index -= incx;
        tp_index -= inctp2;
        inctp2--;
        col_index--;
      }
      TmpType result = impl::mul<TmpType>(rowsum, alpha);
      x[x_index2] = impl::to<T>(result);
      x_index2 += incx;
    }
  } else if ((uplo == blas_lower && trans == blas_no_trans && order == blas_rowmajor) ||
             (uplo == blas_upper && trans != blas_no_trans && order == blas_colmajor)) {

    IdxType tp_start = (n - 1) + ((n - 1) * n) / 2;
    IdxType tp_index = tp_start;
    IdxType x_index = x_start + (n - 1) * incx;
    for (IdxType matrix_row = n - 1; matrix_row >= 0; matrix_row--) {
      IdxType x_index2 = x_index;
      TmpType rowsum = impl::zero_v<TmpType>;

      for (IdxType step = 0; step <= matrix_row; step++) {
        TmpType rowtmp;
        if ((diag == blas_unit_diag) && (step == 0)) {
          rowtmp = impl::to<TmpType>(x[x_index2]);
        } else {
          rowtmp = impl::mul<TmpType>(tp[tp_index], x[x_index2]);
        }
        rowsum += rowtmp;
        x_index2 -= incx;
        tp_index -= 1;
      }
      TmpType result = impl::mul<TmpType>(rowsum, alpha);
      x[x_index] = impl::to<T>(result);
      x_index -= incx;
    }
  } else {
    IdxType x_index = x_start + (n - 1) * incx;
    for (IdxType matrix_row = n - 1; matrix_row >= 0; matrix_row--) {
      IdxType tp_index = matrix_row;
      IdxType x_index2 = x_start;
      TmpType rowsum = impl::zero_v<TmpType>;

      IdxType stride = n;
      for (IdxType step = 0; step <= matrix_row; step++) {
        TmpType rowtmp;
        if ((diag == blas_unit_diag) && (step == matrix_row)) {
          rowtmp = impl::to<TmpType>(x[x_index2]);
        } else {
          rowtmp = impl::mul<TmpType>(tp[tp_index], x[x_index2]);
        }
        rowsum += rowtmp;
        stride--;
        tp_index += stride;
        x_index2 += incx;
      }
      TmpType result = impl::mul<TmpType>(rowsum, alpha);
      x[x_index] = impl::to<T>(result);
      x_index -= incx;
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::tpmv */

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
constexpr void tpmv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      blas_trans_type trans,
                      blas_diag_type diag,
                      N n,
                      T alpha,
                      const A *tp,
                      T *x,
                      N incx,
                      blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * Computes x = alpha * tp * x, x = alpha * tp_transpose * x,
 * or x = alpha * tp_conjugate_transpose where tp is a triangular
 * packed matrix.
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        Order of tp; row or column major
 *
 * uplo   (input) blas_uplo_type
 *        Whether tp is upper or lower
 *
 * trans  (input) blas_trans_type
 *
 * diag   (input) blas_diag_type
 *        Whether the diagonal entries of tp are 1
 *
 * n      (input) N
 *        Dimension of tp and the length of vector x
 *
 * alpha  (input) T
 *
 * tp     (input) const A*
 *
 * x      (input/output) T*
 *
 * incx   (input) N
 *        The stride for vector x.
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
  static const char *routine_name = "XBLAS::tpmv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::tpmv<T, A, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, trans, diag, n, alpha, tp, x, incx);
    break;
  case blas_prec_double:
    XBLAS::tpmv<T, A, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, trans, diag, n, alpha, tp, x, incx);
    break;
  case blas_prec_indigenous:
    XBLAS::tpmv<T, A, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, trans, diag, n, alpha, tp, x, incx);
    break;
  case blas_prec_extra:
    XBLAS::tpmv<T, A, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, trans, diag, n, alpha, tp, x, incx);
    break;
  default:
    BLAS_error(routine_name, -10, prec, nullptr);
    break;
  }
} /* end XBLAS::tpmv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_TPMV_HPP
