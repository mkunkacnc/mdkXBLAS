#ifndef XBLAS_TBSV_HPP
#define XBLAS_TBSV_HPP

#include "blas_enum.h"
#include "common/XBLAS_impl.hpp"

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

template<int do_conj,
         typename TmpType,
         typename T,
         typename A,
         typename N,
         typename IdxType>
constexpr void tbsv_impl_x(blas_diag_type diag,
                           N n,
                           N k,
                           T alpha,
                           const A *t,
                           T *x,
                           N incx,
                           TmpType *x_internal,
                           IdxType start_xi,
                           IdxType dot_start,
                           IdxType dot_inc,
                           IdxType dot_start_inc1,
                           IdxType dot_start_inc2)
{
  TmpType temp1;        /* temporary variable for calculations */
  TmpType temp2;        /* temporary variable for calculations */
  TmpType temp3;        /* temporary variable for calculations */
  T x_elem;
  A T_element;        /* temporary variable for an element of matrix T */

  IdxType x_inti = 0;
  const IdxType inc_x_inti = 1;
  IdxType k_compare = k;        /*used for comparisons with x_inti */

  /*loop 1 */
  IdxType xi = start_xi;
  /* x_inti already initialized to 0 */
  IdxType j = 0;
  for (; j < k; j++) {
    /* each time through loop, xi lands on next x to compute. */
    x_elem = x[xi];
    /* preform the multiplication -
       in this implementation we do not separate the alpha = 1 case */
    temp1 = impl::mul<TmpType>(x_elem, alpha);

    IdxType Tij = dot_start;
    dot_start += dot_start_inc1;

    /*start loop buffer over in loop 1 */
    x_inti = 0;
    for (IdxType i = j; i > 0; i--) {
      T_element = impl::Conj_h<do_conj>::func(t[Tij]);
      temp3 = x_internal[x_inti];
      temp2 = impl::mul<TmpType>(temp3, T_element);
      temp1 = temp1 - temp2;
      x_inti += inc_x_inti;
      Tij += dot_inc;
    } /* for across row */

    /* if the diagonal entry is not equal to one, then divide Xj by
       the entry */
    if (diag == blas_non_unit_diag) {
      T_element = impl::Conj_h<do_conj>::func(t[Tij]);
      temp1 = impl::div(temp1, T_element);
    }

    /* if (diag == blas_non_unit_diag) */
    /* place internal precision result in internal buffer */
    x_internal[x_inti] = temp1;

    /* place result x in same place as got x this loop */
    x[xi] = impl::to<T>(temp1);
    xi += incx;
  } /* for j<k */
  /*end loop 1 */

  /* loop2 ***************************** */
  x_inti = 0;
  /*loop 2 continue without changing j to start */
  for (; j < n; j++) {
    /* each time through loop, xi lands on next x to compute. */
    x_elem = x[xi];
    temp1 = impl::mul<TmpType>(x_elem, alpha);

    IdxType Tij = dot_start;
    dot_start += dot_start_inc2;

    IdxType i = k;
    for (; i > 0 && (x_inti < k_compare); i--) {
      T_element = impl::Conj_h<do_conj>::func(t[Tij]);
      temp3 = x_internal[x_inti];
      temp2 = impl::mul<TmpType>(temp3, T_element);
      temp1 = temp1 - temp2;
      x_inti += inc_x_inti;
      Tij += dot_inc;
    } /* for across row */
    /*reset index to internal storage loop buffer. */
    x_inti = 0;
    for (; i > 0; i--) {
      T_element = impl::Conj_h<do_conj>::func(t[Tij]);
      temp3 = x_internal[x_inti];
      temp2 = impl::mul<TmpType>(temp3, T_element);
      temp1 = temp1 - temp2;
      x_inti += inc_x_inti;
      Tij += dot_inc;
    } /* for across row */

    /* if the diagonal entry is not equal to one, then divide by
       the entry */
    if (diag == blas_non_unit_diag) {
      T_element = impl::Conj_h<do_conj>::func(t[Tij]);
      temp1 = impl::div(temp1, T_element);
    }

    /* if (diag == blas_non_unit_diag) */
    /* place internal precision result in internal buffer */
    x_internal[x_inti] = temp1;
    x_inti += inc_x_inti;
    if (x_inti >= k_compare)
      x_inti = 0;

    /* place result x in same place as got x this loop */
    x[xi] = impl::to<T>(temp1);
    xi += incx;
  } /* for j<n */
}

template<int do_conj,
         typename TmpType,
         typename T,
         typename A,
         typename N,
         typename IdxType>
constexpr void tbsv_impl(blas_diag_type diag,
                         N n,
                         N k,
                         T alpha,
                         const A *t,
                         T *x,
                         N incx,
                         IdxType start_xi,
                         IdxType dot_start,
                         IdxType dot_inc,
                         IdxType dot_start_inc1,
                         IdxType dot_start_inc2)
{
  TmpType temp1;        /* temporary variable for calculations */
  TmpType temp2;        /* temporary variable for calculations */
  T x_elem;
  A T_element;

  /*loop 1 */
  IdxType xi = start_xi;
  IdxType j = 0;
  for (; j < k; j++) {
    /* each time through loop, xi lands on next x to compute. */
    x_elem = x[xi];
    /* preform the multiplication -
       in this implementation we do not separate the alpha = 1 case */
    temp1 = impl::mul<TmpType>(x_elem, alpha);

    xi = start_xi;
    IdxType Tij = dot_start;
    dot_start += dot_start_inc1;

    for (IdxType i = j; i > 0; i--) {
      T_element = impl::Conj_h<do_conj>::func(t[Tij]);
      x_elem = x[xi];
      temp2 = impl::mul<TmpType>(x_elem, T_element);
      temp1 = temp1 - temp2;
      xi += incx;
      Tij += dot_inc;
    } /* for across row */

    /* if the diagonal entry is not equal to one, then divide Xj by
       the entry */
    if (diag == blas_non_unit_diag) {
      T_element = impl::Conj_h<do_conj>::func(t[Tij]);
      temp1 = impl::div(temp1, T_element);
    } /* if (diag == blas_non_unit_diag) */

    x[xi] = impl::to<T>(temp1);
    xi += incx;
  } /* for j<k */
  /*end loop 1 */

  /*loop 2 continue without changing j to start */
  for (; j < n; j++) {
    /* each time through loop, xi lands on next x to compute. */
    x_elem = x[xi];
    temp1 = impl::mul<TmpType>(x_elem, alpha);

    xi = start_xi;
    start_xi += incx;
    IdxType Tij = dot_start;
    dot_start += dot_start_inc2;

    for (IdxType i = k; i > 0; i--) {
      T_element = impl::Conj_h<do_conj>::func(t[Tij]);
      x_elem = x[xi];
      temp2 = impl::mul<TmpType>(x_elem, T_element);
      temp1 = temp1 - temp2;
      xi += incx;
      Tij += dot_inc;
    } /* for across row */

    /* if the diagonal entry is not equal to one, then divide by
       the entry */
    if (diag == blas_non_unit_diag) {
      T_element = impl::Conj_h<do_conj>::func(t[Tij]);
      temp1 = impl::div(temp1, T_element);
    } /* if (diag == blas_non_unit_diag) */

    x[xi] = impl::to<T>(temp1);
    xi += incx;
  } /* for j<n */
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
constexpr void tbsv(blas_order_type order,
                    blas_uplo_type uplo,
                    blas_trans_type trans,
                    blas_diag_type diag,
                    N n,
                    N k,
                    T alpha,
                    const A *t,
                    N ldt,
                    T *x,
                    N incx)
/*
 * Purpose
 * =======
 *
 * This routine solves :
 *
 *     x <- alpha * inverse(t) * x
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        column major, row major (blas_rowmajor, blas_colmajor)
 *
 * uplo   (input) blas_uplo_type
 *        upper, lower (blas_upper, blas_lower)
 *
 * trans  (input) blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) blas_diag_type
 *        unit, non unit (blas_unit_diag, blas_non_unit_diag)
 *
 * n      (input) N
 *        the dimension of t
 *
 * k      (input) N
 *        the number of subdiagonals/superdiagonals of t
 *
 * alpha  (input) T
 *
 * t      (input) const A*
 *        Triangular Banded matrix
 *
 * x      (input/output) T*
 *        Array of length n.
 *
 * incx   (input) N
 *        The stride used to access components x[i].
 *
 */
{
  /* Routine name */
  static const char *routine_name = "XBLAS::tbsv";

  FPU_FIX_DECL;

  if (order != blas_rowmajor && order != blas_colmajor) {
    BLAS_error(routine_name, -1, order, nullptr);
  }
  if (uplo != blas_upper && uplo != blas_lower) {
    BLAS_error(routine_name, -2, uplo, nullptr);
  }
  if ((trans != blas_trans) && (trans != blas_no_trans) &&
      (trans != static_cast<blas_trans_type>(blas_conj)) && (trans != blas_conj_trans)) {
    BLAS_error(routine_name, -2, uplo, nullptr);
  }
  if (diag != blas_non_unit_diag && diag != blas_unit_diag) {
    BLAS_error(routine_name, -4, diag, nullptr);
  }
  if (n < 0) {
    BLAS_error(routine_name, -5, n, nullptr);
  }
  if (k >= n) {
    BLAS_error(routine_name, -6, k, nullptr);
  }
  if ((ldt < 1) || (ldt <= k)) {
    BLAS_error(routine_name, -9, ldt, nullptr);
  }
  if (incx == 0) {
    BLAS_error(routine_name, -11, incx, nullptr);
  }

  if (n == 0)
    return;

  /* configuring the vector starting idx */
  IdxType start_xi;
  if (incx < 0) {
    start_xi = (1 - n) * incx;
  } else {
    start_xi = 0;
  }

  /* if alpha is zero, then return x as a zero vector */
  if (alpha == T(0)) {
    IdxType xi = start_xi;
    for (IdxType i = 0; i < n; i++) {
      x[xi] = T(0);
      xi += incx;
    }
    return;
  }

  /* check to see if k=0.  if so, we can optimize somewhat */
  if (k == 0) {
    if ((alpha == T(1)) && (diag == blas_unit_diag)) {
      /* nothing to do */
      return;
    } else {
      /* just run the loops as is. */
    }
  }

  /* get index variables prepared */
  IdxType dot_start;
  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (order == blas_rowmajor)) {
    dot_start = k;
  } else {
    dot_start = 0;
  }

  IdxType dot_start_inc1, dot_start_inc2, dot_inc;
  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (order == blas_rowmajor)) {
    dot_inc = 1;
    dot_start_inc1 = ldt - 1;
    dot_start_inc2 = ldt;
  } else {
    dot_inc = ldt - 1;
    dot_start_inc1 = 1;
    dot_start_inc2 = ldt;
  }

  if (((trans == blas_trans) || (trans == blas_conj_trans)) ^
      (uplo == blas_lower)) {
    /*start at the first element of x */
    /* substitution will proceed forwards (forward substitution) */
  } else {
    /*start at the last element of x */
    /* substitution will proceed backwards (back substitution) */
    dot_inc = -dot_inc;
    dot_start_inc1 = -dot_start_inc1;
    dot_start_inc2 = -dot_start_inc2;
    dot_start = ldt * (n - 1) + k - dot_start;
    /*order of the following 2 statements matters! */
    start_xi = start_xi + (n - 1) * incx;
    incx = -incx;
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_START;
  }

  if constexpr (impl::is_complex_v<A>) {
    if constexpr (sizeof(TmpType) > sizeof(T)) {

      TmpType *x_internal = new(std::nothrow) TmpType[k];
      if (k > 0 && x_internal == nullptr) {
        BLAS_error(routine_name, -6, k, "allocation failed.\n");
      }

      if ((trans == static_cast<blas_trans_type>(blas_conj)) || (trans == blas_conj_trans)) {
        /* conjugated */
        impl::tbsv_impl_x<1>(diag, n, k, alpha, t, x, incx,
                             x_internal, start_xi, dot_start, dot_inc, dot_start_inc1, dot_start_inc2);
      } else {
        /* not conjugated */
        impl::tbsv_impl_x<0>(diag, n, k, alpha, t, x, incx,
                             x_internal, start_xi, dot_start, dot_inc, dot_start_inc1, dot_start_inc2);
      }

      delete[] x_internal;

    } else {
      if ((trans == static_cast<blas_trans_type>(blas_conj)) || (trans == blas_conj_trans)) {
        /* conjugated */
        impl::tbsv_impl<1, TmpType>(diag, n, k, alpha, t, x, incx,
                                    start_xi, dot_start, dot_inc, dot_start_inc1, dot_start_inc2);
      } else {
        /* not conjugated */
        impl::tbsv_impl<0, TmpType>(diag, n, k, alpha, t, x, incx,
                                    start_xi, dot_start, dot_inc, dot_start_inc1, dot_start_inc2);
      }
    }
  } else {
    if constexpr (sizeof(TmpType) > sizeof(T)) {

      TmpType *x_internal = new(std::nothrow) TmpType[k];
      if (k > 0 && x_internal == nullptr) {
        BLAS_error(routine_name, -6, k, "allocation failed.\n");
      }

      impl::tbsv_impl_x<0>(diag, n, k, alpha, t, x, incx,
                           x_internal, start_xi, dot_start, dot_inc, dot_start_inc1, dot_start_inc2);

      delete[] x_internal;

    } else {
      impl::tbsv_impl<0, TmpType>(diag, n, k, alpha, t, x, incx,
                                  start_xi, dot_start, dot_inc, dot_start_inc1, dot_start_inc2);
    }
  }

  if constexpr (impl::uses_double_double_v<TmpType>) {
    FPU_FIX_STOP;
  }
} /* end XBLAS::tbsv */

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
constexpr void tbsv_x(blas_order_type order,
                      blas_uplo_type uplo,
                      blas_trans_type trans,
                      blas_diag_type diag,
                      N n,
                      N k,
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
 * This routine solves :
 *
 *     x <- alpha * inverse(t) * x
 *
 * Arguments
 * =========
 *
 * order  (input) blas_order_type
 *        column major, row major (blas_rowmajor, blas_colmajor)
 *
 * uplo   (input) blas_uplo_type
 *        upper, lower (blas_upper, blas_lower)
 *
 * trans  (input) blas_trans_type
 *        no trans, trans, conj trans
 *
 * diag   (input) blas_diag_type
 *        unit, non unit (blas_unit_diag, blas_non_unit_diag)
 *
 * n      (input) N
 *        the dimension of t
 *
 * k      (input) N
 *        the number of subdiagonals/superdiagonals of t
 *
 * alpha  (input) T
 *
 * t      (input) const A*
 *        Triangular Banded matrix
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
  static const char *routine_name = "XBLAS::tbsv_x";
  if (k == 0) {
    if constexpr (std::is_same_v<impl::inner_type_t<T>, float>) {
      /* must set prec to output. Ignore user input of prec */
      prec = blas_prec_single;
    } else if constexpr (std::is_same_v<impl::inner_type_t<T>, double>) {
      /* must set prec to output. Ignore user input of prec */
      prec = blas_prec_double;
    }
  }

  switch (prec) {
  case blas_prec_single:
    XBLAS::tbsv<T, A, N, impl::internal_precision_t<T, blas_prec_single>, IdxType>(order, uplo, trans, diag, n, k, alpha, t, ldt, x, incx);
    break;
  case blas_prec_double:
    XBLAS::tbsv<T, A, N, impl::internal_precision_t<T, blas_prec_double>, IdxType>(order, uplo, trans, diag, n, k, alpha, t, ldt, x, incx);
    break;
  case blas_prec_indigenous:
    XBLAS::tbsv<T, A, N, impl::internal_precision_t<T, blas_prec_indigenous>, IdxType>(order, uplo, trans, diag, n, k, alpha, t, ldt, x, incx);
    break;
  case blas_prec_extra:
    XBLAS::tbsv<T, A, N, impl::internal_precision_t<T, blas_prec_extra>, IdxType>(order, uplo, trans, diag, n, k, alpha, t, ldt, x, incx);
    break;
  default:
    BLAS_error(routine_name, -12, prec, nullptr);
    break;
  }
} /* end XBLAS::tbsv_x */

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_TBSV_HPP
