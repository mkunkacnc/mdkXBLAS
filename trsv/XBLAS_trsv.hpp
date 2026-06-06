#ifndef XBLAS_TRSV_HPP
#define XBLAS_TRSV_HPP

#include "blas_enum.h"
#include "common/XBLAS_impl.hpp"
#include <new>

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

template<int do_conj,
         int do_incr,
         int do_j_i,
         typename TmpType,
         typename IdxType,
         typename T,
         typename A,
         typename X,
         typename N>
constexpr void trsv_impl(blas_diag_type diag,
                         N n,
                         T alpha,
                         const A *t,
                         N ldt,
                         X *x,
                         IdxType start_x,
                         N incx)
{
  if constexpr (do_incr) {
    IdxType jx = start_x;
    for (IdxType j = 0; j < n; j++) {
      /* compute Xj = Xj - SUM Aij(or Aji) * Xi
         i=j+1 to n-1           */
      TmpType temp3 = impl::to<TmpType>(x[jx]);
      TmpType temp1 = impl::mul<TmpType>(temp3, alpha);

      IdxType ix = start_x;
      for (IdxType i = 0; i < j; i++) {
        A T_element;
        if constexpr (do_j_i) {
            T_element = impl::Conj_h<do_conj>::func(t[j + i * ldt]);
        } else {
            T_element = impl::Conj_h<do_conj>::func(t[i + j * ldt]);
        }
        TmpType temp3 = impl::to<TmpType>(x[ix]);
        TmpType temp2 = impl::mul<TmpType>(temp3, T_element);
        temp1 = temp1 - temp2;
        ix += incx;
      } /* for i<j */

      /* if the diagonal entry is not equal to one, then divide Xj by
         the entry */
      if (diag == blas_non_unit_diag) {
        A T_element = impl::Conj_h<do_conj>::func(t[j + j * ldt]);
        temp1 = impl::div(temp1, T_element);
      }
      /* if (diag == blas_non_unit_diag) */
      x[jx] = impl::to<X>(temp1);
      jx += incx;
    }
  } else {
    IdxType jx = start_x + (n - 1) * incx;
    for (IdxType j = n - 1; j >= 0; j--) {
      /* compute Xj = alpha*Xj - SUM Aij(or Aji) * Xi
         i=j+1 to n-1           */
      TmpType temp3 = impl::to<TmpType>(x[jx]);
      TmpType temp1 = impl::mul<TmpType>(temp3, alpha);

      IdxType ix = start_x + (n - 1) * incx;
      for (IdxType i = n - 1; i >= j + 1; i--) {
        A T_element;
        if constexpr (do_j_i) {
            T_element = impl::Conj_h<do_conj>::func(t[j + i * ldt]);
        } else {
            T_element = impl::Conj_h<do_conj>::func(t[i + j * ldt]);
        }
        TmpType temp3 = impl::to<TmpType>(x[ix]);
        TmpType temp2 = impl::mul<TmpType>(temp3, T_element);
        temp1 = temp1 - temp2;
        ix -= incx;
      } /* for j<n */

      /* if the diagonal entry is not equal to one, then divide Xj by
         the entry */
      if (diag == blas_non_unit_diag) {
        A T_element = impl::Conj_h<do_conj>::func(t[j + j * ldt]);
        temp1 = impl::div(temp1, T_element);
      }
      /* if (diag == blas_non_unit_diag) */
      x[jx] = impl::to<X>(temp1);
      jx -= incx;
    }
  }
}

//-----------------
} // namespace impl
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
constexpr void trsv(blas_order_type order,
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
 * This routine solve :
 *
 *     x <- alpha * inverse(T) * x
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
 * x      (input/output) T*
 *        Array of length n.
 *
 * incx   (input) N
 *        The stride used to access components x[i].
 *
 */
{
  static const char *routine_name = "XBLAS::trsv";

  FPU_FIX_DECL;

  if (order != blas_rowmajor && order != blas_colmajor) {
    BLAS_error(routine_name, -1, order, nullptr);
  }

  if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, nullptr);
  }

  if (trans != blas_trans && trans != blas_no_trans && trans != blas_conj_trans) {
    BLAS_error(routine_name, -3, trans, nullptr);
  }

  if (diag != blas_non_unit_diag && diag != blas_unit_diag) {
    BLAS_error(routine_name, -4, diag, nullptr);
  }

  if (n < 0) {
    BLAS_error(routine_name, -5, n, nullptr);
  }

  if (ldt < n) {
    BLAS_error(routine_name, -8, ldt, nullptr);
  }

  if (incx == 0) {
    BLAS_error(routine_name, -10, incx, nullptr);
  }

  if (n == 0)
    return;

  /* configuring the vector starting idx */
  IdxType start_x;
  if (incx <= 0) {
    start_x = -(n - 1) * incx;
  } else {
    start_x = 0;
  }

  /* if alpha is zero, then return x as a zero vector */
  if (alpha == T(0)) {
    IdxType ix = start_x;
    for (IdxType i = 0; i < n; i++) {
      x[ix] = T(0);
      ix += incx;
    }
    return;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if constexpr (impl::is_complex_v<A>) {
    if constexpr (sizeof(TmpType) > sizeof(T)) {
      /* allocate space for intx */
      TmpType *intx = new(std::nothrow) TmpType[n];

      if (n > 0 && intx == nullptr) {
        BLAS_error(routine_name, -5, n, "allocation failed.\n");
      }

      /* copy x to intx */
      IdxType ix = start_x;
      IdxType jx = 0;
      for (IdxType i = 0; i < n; i++) {
        TmpType temp1 = impl::to<TmpType>(x[ix]);
        intx[jx] = temp1;
        ix += incx;
        ++jx;
      }

      if ((order == blas_rowmajor && trans == blas_no_trans && uplo == blas_upper) ||
          (order == blas_colmajor && trans != blas_no_trans && uplo == blas_lower)) {
        if (trans == blas_conj_trans) {
          impl::trsv_impl<1, 0, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);
        } else {
          impl::trsv_impl<0, 0, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);
        }
      } else if ((order == blas_rowmajor && trans == blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor && trans != blas_no_trans && uplo == blas_upper)) {
        if (trans == blas_conj_trans) {
          impl::trsv_impl<1, 1, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);
        } else {
          impl::trsv_impl<0, 1, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);
        }
      } else if ((order == blas_rowmajor && trans != blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor && trans == blas_no_trans && uplo == blas_upper)) {
        if (trans == blas_conj_trans) {
          impl::trsv_impl<1, 0, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);
        } else {
          impl::trsv_impl<0, 0, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);
        }
      } else if ((order == blas_rowmajor && trans != blas_no_trans && uplo == blas_upper) ||
                 (order == blas_colmajor && trans == blas_no_trans && uplo == blas_lower)) {
        if (trans == blas_conj_trans) {
          impl::trsv_impl<1, 1, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);
        } else {
          impl::trsv_impl<0, 1, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);
        }
      }

      /* copy the final results from intx to x */
      ix = start_x;
      jx = 0;
      for (IdxType i = 0; i < n; i++) {
        TmpType temp1 = intx[jx];
        x[ix] = impl::to<T>(temp1);
        ix += incx;
        ++jx;
      }

      delete[] intx;

    } else {
      if ((order == blas_rowmajor && trans == blas_no_trans && uplo == blas_upper) ||
          (order == blas_colmajor && trans != blas_no_trans && uplo == blas_lower)) {
        if (trans == blas_conj_trans) {
          impl::trsv_impl<1, 0, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);
        } else {
          impl::trsv_impl<0, 0, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);
        }
      } else if ((order == blas_rowmajor && trans == blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor && trans != blas_no_trans && uplo == blas_upper)) {
        if (trans == blas_conj_trans) {
          impl::trsv_impl<1, 1, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);
        } else {
          impl::trsv_impl<0, 1, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);
        }
      } else if ((order == blas_rowmajor && trans != blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor && trans == blas_no_trans && uplo == blas_upper)) {
        if (trans == blas_conj_trans) {
          impl::trsv_impl<1, 0, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);
        } else {
          impl::trsv_impl<0, 0, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);
        }
      } else if ((order == blas_rowmajor && trans != blas_no_trans && uplo == blas_upper) ||
                 (order == blas_colmajor && trans == blas_no_trans && uplo == blas_lower)) {
        if (trans == blas_conj_trans) {
          impl::trsv_impl<1, 1, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);
        } else {
          impl::trsv_impl<0, 1, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);
        }
      }
    }
  } else {
    if constexpr (sizeof(TmpType) > sizeof(T)) {
      /* allocate space for intx */
      TmpType *intx = new(std::nothrow) TmpType[n];
      if (n > 0 && intx == nullptr) {
        BLAS_error(routine_name, 0, 0, "allocation failed.\n");
      }

      /* copy x to intx */
      IdxType ix = start_x;
      IdxType jx = 0;
      for (IdxType i = 0; i < n; i++) {
        TmpType temp1 = impl::to<TmpType>(x[ix]);
        intx[jx] = temp1;
        ix += incx;
        ++jx;
      }

      if ((order == blas_rowmajor && trans == blas_no_trans && uplo == blas_upper) ||
          (order == blas_colmajor && trans != blas_no_trans && uplo == blas_lower)) {

        impl::trsv_impl<0, 0, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);

      } else if ((order == blas_rowmajor && trans == blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor && trans != blas_no_trans && uplo == blas_upper)) {

        impl::trsv_impl<0, 1, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);

      } else if ((order == blas_rowmajor && trans != blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor && trans == blas_no_trans && uplo == blas_upper)) {

        impl::trsv_impl<0, 0, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);

      } else if ((order == blas_rowmajor && trans != blas_no_trans && uplo == blas_upper) ||
                 (order == blas_colmajor && trans == blas_no_trans && uplo == blas_lower)) {

        impl::trsv_impl<0, 1, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, intx, 0, 1);
      }

      /* copy the final results from intx to x */
      ix = start_x;
      jx = 0;
      for (IdxType i = 0; i < n; i++) {
        TmpType temp1 = intx[jx];
        x[ix] = impl::to<T>(temp1);
        ix += incx;
        ++jx;
      }

      delete[] intx;

    } else {
      if ((order == blas_rowmajor && trans == blas_no_trans && uplo == blas_upper) ||
          (order == blas_colmajor && trans != blas_no_trans && uplo == blas_lower)) {

        impl::trsv_impl<0, 0, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);

      } else if ((order == blas_rowmajor && trans == blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor && trans != blas_no_trans && uplo == blas_upper)) {

        impl::trsv_impl<0, 1, 0, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);

      } else if ((order == blas_rowmajor && trans != blas_no_trans && uplo == blas_lower) ||
                 (order == blas_colmajor && trans == blas_no_trans && uplo == blas_upper)) {

        impl::trsv_impl<0, 0, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);

      } else if ((order == blas_rowmajor && trans != blas_no_trans && uplo == blas_upper) ||
                 (order == blas_colmajor && trans == blas_no_trans && uplo == blas_lower)) {

        impl::trsv_impl<0, 1, 1, TmpType, IdxType>(diag, n, alpha, t, ldt, x, start_x, incx);
      }
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::trsv */

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
constexpr void trsv_x(blas_order_type order,
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
 * This routine solve :
 *
 *     x <- alpha * inverse(T) * x
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
  static const char *routine_name = "XBLAS::trsv_x";
  switch (prec) {
  case blas_prec_single:
    XBLAS::trsv<T, A, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_double:
    XBLAS::trsv<T, A, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_indigenous:
    XBLAS::trsv<T, A, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  case blas_prec_extra:
    XBLAS::trsv<T, A, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, trans, diag, n, alpha, t, ldt, x, incx);
    break;
  default:
    BLAS_error(routine_name, -11, prec, nullptr);
    break;
  }
} /* end XBLAS::trsv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_TRSV_HPP
