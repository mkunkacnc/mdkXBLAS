#ifndef XBLAS_DOUBLEDOUBLE_HPP
#define XBLAS_DOUBLEDOUBLE_HPP

#include "blas_extended_private.h"

class DoubleDouble
{
public:
    DoubleDouble() {}
    explicit DoubleDouble(double x);
    DoubleDouble(double h, double t);

    static DoubleDouble mul(float a, float b);
    static DoubleDouble mul(double a, double b);

public:
    static constexpr double split = SPLIT;

    double head;
    double tail;
};

DoubleDouble operator *(const DoubleDouble& a, double b);

//-----------------

inline
DoubleDouble::DoubleDouble(double x)
:head(x),
 tail(0)
{}

inline
DoubleDouble::DoubleDouble(double h, double t)
:head(h),
 tail(t)
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

  DoubleDouble c(a * b);
  c.tail = (((a1 * b1 - c.head) + a1 * b2) + a2 * b1) + a2 * b2;
  return c;
}

inline
DoubleDouble DoubleDouble::mul(float a, float b)
{
  /* Compute double-double = float * float. */
  return DoubleDouble(static_cast<double>(a) * b);
}

inline
DoubleDouble operator *(const DoubleDouble& a, double b)
{
  /* Compute double-double = double-double * double. */
  double a11, a21, b1, b2, c11, c21, c2, con, t1, t2;

  con = a.head * DoubleDouble::split;
  a11 = con - a.head;
  a11 = con - a11;
  a21 = a.head - a11;

  con = b * DoubleDouble::split;
  b1 = con - b;
  b1 = con - b1;
  b2 = b - b1;

  c11 = a.head * b;
  c21 = (((a11 * b1 - c11) + a11 * b2) + a21 * b1) + a21 * b2;

  c2 = a.tail * b;
  t1 = c11 + c2;
  t2 = (c2 - (t1 - c11)) + c21;

  DoubleDouble c(t1 + t2);
  c.tail = t2 - (c.head - t1);
  return c;
}

#endif // XBLAS_DOUBLEDOUBLE_HPP
