#ifndef XBLAS_DOUBLEDOUBLE_HPP
#define XBLAS_DOUBLEDOUBLE_HPP

#include "blas_extended_private.h"

class DoubleDouble
{
public:
    DoubleDouble() {}
    explicit DoubleDouble(double x);
    DoubleDouble(double h, double t);

    static DoubleDouble add(double a, double b);

    static DoubleDouble mul(float a, float b);
    static DoubleDouble mul(double a, double b);

    DoubleDouble& operator +=(const DoubleDouble& rhs);

public:
    static constexpr double split = SPLIT;

    double head;
    double tail;
};

DoubleDouble operator *(const DoubleDouble& a, double b);
DoubleDouble operator +(const DoubleDouble& a, const DoubleDouble& b);

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
DoubleDouble DoubleDouble::add(double a, double b)
{
  /* Compute double-double = double + double. */
  double e, t1, t2;

  /* Knuth trick. */
  t1 = a + b;
  e = t1 - a;
  t2 = ((b - e) + (a - (t1 - e)));

  /* The result is t1 + t2, after normalization. */
  DoubleDouble c(t1 + t2);
  c.tail = t2 - (c.head - t1);
  return c;
}

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
DoubleDouble& DoubleDouble::operator +=(const DoubleDouble& rhs)
{
  DoubleDouble lhs(*this);
  *this = lhs + rhs;
  return *this;
}

//-----------------

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

inline
DoubleDouble operator +(const DoubleDouble& a, const DoubleDouble& b)
{
  /* Compute double-double = double-double + double-double. */
  double bv;
  double s1, s2, t1, t2;

  /* Add two hi words. */
  s1 = a.head + b.head;
  bv = s1 - a.head;
  s2 = ((b.head - bv) + (a.head - (s1 - bv)));

  /* Add two lo words. */
  t1 = a.tail + b.tail;
  bv = t1 - a.tail;
  t2 = ((b.tail - bv) + (a.tail - (t1 - bv)));

  s2 += t1;

  /* Renormalize (s1, s2)  to  (t1, s2) */
  t1 = s1 + s2;
  s2 = s2 - (t1 - s1);

  t2 += s2;

  /* Renormalize (t1, t2)  */
  DoubleDouble c(t1 + t2);
  c.tail = t2 - (c.head - t1);
  return c;
}


#endif // XBLAS_DOUBLEDOUBLE_HPP
