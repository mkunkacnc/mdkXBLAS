#include "blas_extended.h"
#include "fpinfo.hpp"

int BLAS_fpinfo_x(enum blas_cmach_type cmach, enum blas_prec_type prec)
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
  return XBLAS::fpinfo_x(cmach, prec);
}
