#ifndef XBLAS_FPINFO_HPP
#define XBLAS_FPINFO_HPP

#include "blas_enum.h"
#include "double_double.hpp"
#include <limits>

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

template<typename T>
constexpr int fpinfo(enum blas_cmach_type cmach)
{
  int rmach = 0;

  if (cmach == blas_base) {
    rmach = std::numeric_limits<T>::radix; // FLT_RADIX, 2
  } else if (cmach == blas_t) {
    rmach = std::numeric_limits<T>::digits; // FLT_DIG, 24
  } else if (cmach == blas_rnd) {
    rmach = std::numeric_limits<T>::round_style == std::round_to_nearest; // FLT_ROUNDS == 1
  } else if (cmach == blas_ieee) {
    rmach = std::numeric_limits<T>::is_iec559; // 1
  } else if (cmach == blas_emin) {
    rmach = std::numeric_limits<T>::min_exponent; // -126;
  } else if (cmach == blas_emax) {
    rmach = std::numeric_limits<T>::max_exponent; // 127;
  }

  return rmach;
}

template<>
inline
constexpr int fpinfo<double_double>(enum blas_cmach_type cmach)
{
  int rmach = 0;

  if (cmach == blas_base) {
    rmach = std::numeric_limits<double>::radix;
  } else if (cmach == blas_t) {
    rmach = 2*std::numeric_limits<double>::digits - 1;
  } else if (cmach == blas_rnd) {
    rmach = 0; // false
  } else if (cmach == blas_ieee) {
    rmach = 0; // false
  } else if (cmach == blas_emin) {
    rmach = std::numeric_limits<double>::min_exponent;
  } else if (cmach == blas_emax) {
    rmach = std::numeric_limits<double>::max_exponent;
  }

  return rmach;
}

//-----------------
} // namespace impl
//-----------------

inline
constexpr int fpinfo_x(enum blas_cmach_type cmach, enum blas_prec_type prec)
/*
 * Purpose
 * =======
 *
 * This routine queries for machine-specific floating point characteristics.
 *
 * Arguments
 * =========
 *
 * cmach  (input) enum blas_cmach_type
 *        Specifies the floating point parameter to be returned.
 *        = blas_base: base of the machine
 *        = blas_t   : number of "correct" (BASE) digits in the mantissa
 *        = blas_rnd : 1 when "proper rounding" occurs in addition,
 *                     0 otherwise
 *        = blas_ieee: 1 when rounding in addition occurs in "IEEE style",
 *                     0 otherwise
 *        = blas_emin: minimum exponent before (gradual) underflow
 *        = blas_emax: maximum exponent before overflow
 *
 * prec   (input) enum blas_prec_type
 *        Specifies the internal precision to be used.
 *        = blas_prec_single: single precision.
 *        = blas_prec_double: double precision.
 *        = blas_prec_extra : anything at least 1.5 times as accurate
 *                            than double, and wider than 80-bits.
 *                            We use double-double in our implementation.
 *
 */
{
  int rmach = 0;

  switch (prec) {
  case blas_prec_single:
    rmach = impl::fpinfo<float>(cmach);
    break;

  case blas_prec_indigenous:
  case blas_prec_double:
    rmach = impl::fpinfo<double>(cmach);
    break;

  case blas_prec_extra:
    rmach = impl::fpinfo<double_double>(cmach);
    break;
  }

  return rmach;
}

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_FPINFO_HPP
