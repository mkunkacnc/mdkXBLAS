#ifndef XBLAS_TPMV_HPP
#define XBLAS_TPMV_HPP

#include "blas_extended_private.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

template<typename T,
         typename A,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void tpmv(blas_order_type order,
                    blas_uplo_type uplo,
                    blas_trans_type trans,
                    blas_diag_type diag,
                    IdxType n,
                    T alpha,
                    const A *tp,
                    T *x,
                    IdxType incx)
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
 * n      (input) IdxType
 *        Dimension of tp and the length of vector x
 *
 * alpha  (input) T
 *
 * tp     (input) const A*
 *
 * x      (input/output) T*
 *
 * incx   (input) IdxType
 *        The stride for vector x.
 *
 */
{
  static const char routine_name[] = "XBLAS::tpmv";

  FPU_FIX_DECL;

  IdxType matrix_row, step, tp_index, tp_start, x_index, x_start;
  IdxType inctp, x_index2, stride, col_index, inctp2;

  T alpha_i = alpha;

  const A *tp_i = tp;
  T *x_i = x;
  TmpType rowsum;
  TmpType rowtmp;
  TmpType result;
  A matval;
  T vecval;
  A one = A(1);

  inctp = 1;

  if (incx < 0)
    x_start = (-n + 1) * incx;
  else
    x_start = 0;

  if (n < 1) {
    return;
  }

  /* Check for error conditions. */
  if (order != blas_colmajor && order != blas_rowmajor) {
    BLAS_error(routine_name, -1, order, nullptr);
  }
  if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, nullptr);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -9, incx, nullptr);
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if constexpr (impl::is_complex_v<A>) { // Any differences between complex and real? This may be unnecessary.
    if ((uplo == blas_upper &&
         trans == blas_no_trans && order == blas_rowmajor) ||
        (uplo == blas_lower &&
         trans != blas_no_trans && order == blas_colmajor)) {
      tp_start = 0;
      tp_index = tp_start;
      for (matrix_row = 0; matrix_row < n; matrix_row++) {
        x_index = x_start + incx * matrix_row;
        x_index2 = x_index;
        col_index = matrix_row;
        rowsum = impl::zero_v<TmpType>;
        rowtmp = impl::zero_v<TmpType>;
        result = impl::zero_v<TmpType>;

        while (col_index < n) {
          vecval = x_i[x_index];
          if ((diag == blas_unit_diag) && (col_index == matrix_row)) {
            rowtmp = impl::mul<TmpType>(vecval, one); // why multiply by one?
          } else {
            matval = tp_i[tp_index];
            rowtmp = impl::mul<TmpType>(matval, vecval);
          }
          rowsum = rowsum + rowtmp;
          x_index += incx;
          tp_index += inctp;
          col_index++;
        }
        result = impl::mul<TmpType>(rowsum, alpha_i);
        x_i[x_index2] = impl::to<T>(result);
      }
    } else if ((uplo == blas_upper &&
                trans == blas_no_trans && order == blas_colmajor) ||
               (uplo == blas_lower &&
                trans != blas_no_trans && order == blas_rowmajor)) {
      tp_start = ((n - 1) * n) / 2;
      inctp2 = n - 1;
      x_index2 = x_start;
      for (matrix_row = 0; matrix_row < n; matrix_row++, inctp2 = n - 1) {
        x_index = x_start + incx * (n - 1);
        tp_index = (tp_start + matrix_row) * inctp;
        col_index = (n - 1) - matrix_row;
        rowsum = impl::zero_v<TmpType>;
        rowtmp = impl::zero_v<TmpType>;
        result = impl::zero_v<TmpType>;

        while (col_index >= 0) {
          vecval = x_i[x_index];
          if ((diag == blas_unit_diag) && (col_index == 0)) {
            rowtmp = impl::mul<TmpType>(vecval, one);
          } else {
            matval = tp_i[tp_index];
            rowtmp = impl::mul<TmpType>(matval, vecval);
          }
          rowsum = rowsum + rowtmp;
          x_index -= incx;
          tp_index -= inctp2 * inctp;
          inctp2--;
          col_index--;
        }
        result = impl::mul<TmpType>(rowsum, alpha_i);
        x_i[x_index2] = impl::to<T>(result);
        x_index2 += incx;
      }
    } else if ((uplo == blas_lower &&
                trans == blas_no_trans && order == blas_rowmajor) ||
               (uplo == blas_upper &&
                trans != blas_no_trans && order == blas_colmajor)) {
      tp_start = (n - 1) + ((n - 1) * n) / 2;
      tp_index = tp_start * inctp;
      x_index = x_start + (n - 1) * incx;

      for (matrix_row = n - 1; matrix_row >= 0; matrix_row--) {
        x_index2 = x_index;
        rowsum = impl::zero_v<TmpType>;
        rowtmp = impl::zero_v<TmpType>;
        result = impl::zero_v<TmpType>;

        for (step = 0; step <= matrix_row; step++) {
          vecval = x_i[x_index2];
          if ((diag == blas_unit_diag) && (step == 0)) {
            rowtmp = impl::mul<TmpType>(vecval, one);
          } else {
            matval = tp_i[tp_index];
            rowtmp = impl::mul<TmpType>(matval, vecval);
          }
          rowsum = rowsum + rowtmp;
          x_index2 -= incx;
          tp_index -= inctp;
        }
        result = impl::mul<TmpType>(rowsum, alpha_i);
        x_i[x_index] = impl::to<T>(result);
        x_index -= incx;
      }
    } else {
      tp_start = 0;
      x_index = x_start + (n - 1) * incx;
      for (matrix_row = n - 1; matrix_row >= 0; matrix_row--) {
        tp_index = matrix_row * inctp;
        x_index2 = x_start;
        rowsum = impl::zero_v<TmpType>;
        rowtmp = impl::zero_v<TmpType>;
        result = impl::zero_v<TmpType>;

        stride = n;
        for (step = 0; step <= matrix_row; step++) {
          vecval = x_i[x_index2];
          if ((diag == blas_unit_diag) && (step == matrix_row)) {
            rowtmp = impl::mul<TmpType>(vecval, one);
          } else {
            matval = tp_i[tp_index];
            rowtmp = impl::mul<TmpType>(matval, vecval);
          }
          rowsum = rowsum + rowtmp;
          stride--;
          tp_index += stride * inctp;
          x_index2 += incx;
        }
        result = impl::mul<TmpType>(rowsum, alpha_i);
        x_i[x_index] = impl::to<T>(result);
        x_index -= incx;
      }
    }
  } else {
    if ((uplo == blas_upper &&
         trans == blas_no_trans && order == blas_rowmajor) ||
        (uplo == blas_lower &&
         trans != blas_no_trans && order == blas_colmajor)) {
      tp_start = 0;
      tp_index = tp_start;
      for (matrix_row = 0; matrix_row < n; matrix_row++) {
        x_index = x_start + incx * matrix_row;
        x_index2 = x_index;
        col_index = matrix_row;
        rowsum = impl::zero_v<TmpType>;
        rowtmp = impl::zero_v<TmpType>;
        result = impl::zero_v<TmpType>;
        while (col_index < n) {
          vecval = x_i[x_index];
          if ((diag == blas_unit_diag) && (col_index == matrix_row)) {
            rowtmp = impl::mul<TmpType>(vecval, one);
          } else {
            matval = tp_i[tp_index];
            rowtmp = impl::mul<TmpType>(matval, vecval);
          }
          rowsum = rowsum + rowtmp;
          x_index += incx;
          tp_index += inctp;
          col_index++;
        }
        result = impl::mul<TmpType>(rowsum, alpha_i);
        x_i[x_index2] = impl::to<T>(result);
      }
    } else if ((uplo == blas_upper &&
                trans == blas_no_trans && order == blas_colmajor) ||
               (uplo == blas_lower &&
                trans != blas_no_trans && order == blas_rowmajor)) {
      tp_start = ((n - 1) * n) / 2;
      inctp2 = n - 1;
      x_index2 = x_start;
      for (matrix_row = 0; matrix_row < n; matrix_row++, inctp2 = n - 1) {
        x_index = x_start + incx * (n - 1);
        tp_index = (tp_start + matrix_row) * inctp;
        col_index = (n - 1) - matrix_row;
        rowsum = impl::zero_v<TmpType>;
        rowtmp = impl::zero_v<TmpType>;
        result = impl::zero_v<TmpType>;
        while (col_index >= 0) {
          vecval = x_i[x_index];
          if ((diag == blas_unit_diag) && (col_index == 0)) {
            rowtmp = impl::mul<TmpType>(vecval, one);
          } else {
            matval = tp_i[tp_index];
            rowtmp = impl::mul<TmpType>(matval, vecval);
          }
          rowsum = rowsum + rowtmp;
          x_index -= incx;
          tp_index -= inctp2 * inctp;
          inctp2--;
          col_index--;
        }
        result = impl::mul<TmpType>(rowsum, alpha_i);
        x_i[x_index2] = impl::to<T>(result);
        x_index2 += incx;
      }
    } else if ((uplo == blas_lower &&
                trans == blas_no_trans && order == blas_rowmajor) ||
               (uplo == blas_upper &&
                trans != blas_no_trans && order == blas_colmajor)) {
      tp_start = (n - 1) + ((n - 1) * n) / 2;
      tp_index = tp_start * inctp;
      x_index = x_start + (n - 1) * incx;

      for (matrix_row = n - 1; matrix_row >= 0; matrix_row--) {
        x_index2 = x_index;
        rowsum = impl::zero_v<TmpType>;
        rowtmp = impl::zero_v<TmpType>;
        result = impl::zero_v<TmpType>;
        for (step = 0; step <= matrix_row; step++) {
          vecval = x_i[x_index2];
          if ((diag == blas_unit_diag) && (step == 0)) {
            rowtmp = impl::mul<TmpType>(vecval, one);
          } else {
            matval = tp_i[tp_index];
            rowtmp = impl::mul<TmpType>(matval, vecval);
          }
          rowsum = rowsum + rowtmp;
          x_index2 -= incx;
          tp_index -= inctp;
        }
        result = impl::mul<TmpType>(rowsum, alpha_i);
        x_i[x_index] = impl::to<T>(result);
        x_index -= incx;
      }
    } else {
      tp_start = 0;
      x_index = x_start + (n - 1) * incx;
      for (matrix_row = n - 1; matrix_row >= 0; matrix_row--) {
        tp_index = matrix_row * inctp;
        x_index2 = x_start;
        rowsum = impl::zero_v<TmpType>;
        rowtmp = impl::zero_v<TmpType>;
        result = impl::zero_v<TmpType>;
        stride = n;
        for (step = 0; step <= matrix_row; step++) {
          vecval = x_i[x_index2];
          if ((diag == blas_unit_diag) && (step == matrix_row)) {
            rowtmp = impl::mul<TmpType>(vecval, one);
          } else {
            matval = tp_i[tp_index];
            rowtmp = impl::mul<TmpType>(matval, vecval);
          }
          rowsum = rowsum + rowtmp;
          stride--;
          tp_index += stride * inctp;
          x_index2 += incx;
        }
        result = impl::mul<TmpType>(rowsum, alpha_i);
        x_i[x_index] = impl::to<T>(result);
        x_index -= incx;
      }
    }
  }
  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::tpmv */

//-----------------

template<typename T,
         typename A,
         typename TmpType = T,
         typename IdxType = int>
requires (impl::size_le_v<A, T> &&
          impl::size_le_v<T, TmpType> &&
          std::signed_integral<IdxType>)
constexpr void tpmv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      blas_trans_type trans,
                      blas_diag_type diag,
                      IdxType n,
                      T alpha,
                      const A *tp,
                      T *x,
                      IdxType incx,
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
 * n      (input) IdxType
 *        Dimension of tp and the length of vector x
 *
 * alpha  (input) T
 *
 * tp     (input) const A*
 *
 * x      (input/output) T*
 *
 * incx   (input) IdxType
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
//static const char routine_name[] = "XBLAS::tpmv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::tpmv<T, A, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, trans, diag, n, alpha, tp, x, incx);
    break;
  case blas_prec_double:
    XBLAS::tpmv<T, A, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, trans, diag, n, alpha, tp, x, incx);
    break;
  case blas_prec_indigenous:
    XBLAS::tpmv<T, A, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, trans, diag, n, alpha, tp, x, incx);
    break;
  case blas_prec_extra:
    XBLAS::tpmv<T, A, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, trans, diag, n, alpha, tp, x, incx);
    break;
  }
} /* end XBLAS::tpmv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_TPMV_HPP
