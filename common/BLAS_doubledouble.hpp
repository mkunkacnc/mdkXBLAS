#ifndef XBLAS_DOUBLEDOUBLE_HPP
#define XBLAS_DOUBLEDOUBLE_HPP

#include "blas_extended_private.h"

class DoubleDouble
{
public:
    DoubleDouble() {}
    explicit DoubleDouble(double x);

    static DoubleDouble mul(float a, float b);
    static DoubleDouble mul(double a, double b);

private:
    static constexpr double split = SPLIT;

public:
    double head;
    double tail;
};

inline
DoubleDouble::DoubleDouble(double x)
:head(x),
 tail(0)
{}

inline
DoubleDouble DoubleDouble::mul(double a, double b)
{
  /* Compute double-double = double * double. */
  double a1, a2, b1, b2, con;

#define SPLIT_VAR(a)                \
  con = a * DoubleDouble::split;    \
  a##1 = con - a;                   \
  a##1 = con - a##1;                \
  a##2 = a - a##1;

  SPLIT_VAR(a)
  SPLIT_VAR(b)

#undef SPLIT_VAR

  DoubleDouble res(a * b);
  res.tail = (((a1 * b1 - res.head) + a1 * b2) + a2 * b1) + a2 * b2;
  return res;
}

inline
DoubleDouble DoubleDouble::mul(float a, float b)
{
  /* Compute double-double = float * float. */
  return DoubleDouble(static_cast<double>(a) * b);
}

#endif // XBLAS_DOUBLEDOUBLE_HPP
