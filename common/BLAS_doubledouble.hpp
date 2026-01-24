#ifndef XBLAS_DOUBLE_DOUBLE_HPP
#define XBLAS_DOUBLE_DOUBLE_HPP

#include "blas_extended_private.h"

//---------------
namespace XBLAS {
//---------------

class double_double
{
public:
    double_double() {}
    explicit double_double(double x);
    double_double(double h, double t);

    static double_double add(double a, double b);

    static double_double mul(float a, float b);
    static double_double mul(double a, double b);

    double_double& operator +=(const double_double& rhs);

public:
    static constexpr double split = SPLIT;

    double head;
    double tail;
};

double_double operator *(const double_double& a, float b);
double_double operator *(const double_double& a, double b);
double_double operator +(const double_double& a, const double_double& b);

//-----------------

inline
double_double::double_double(double x)
:head(x),
 tail(0)
{}

inline
double_double::double_double(double h, double t)
:head(h),
 tail(t)
{}

inline
double_double double_double::add(double a, double b)
{
  /* Compute double-double = double + double. */
  double e, t1, t2;

  /* Knuth trick. */
  t1 = a + b;
  e = t1 - a;
  t2 = ((b - e) + (a - (t1 - e)));

  /* The result is t1 + t2, after normalization. */
  double_double c(t1 + t2);
  c.tail = t2 - (c.head - t1);
  return c;
}

inline
double_double double_double::mul(double a, double b)
{
  /* Compute double-double = double * double. */
  double a1, a2, b1, b2, con;

#define SPLIT_VAR(a)                \
  con = a * double_double::split;    \
  a##1 = con - a;                   \
  a##1 = con - a##1;                \
  a##2 = a - a##1;

  SPLIT_VAR(a)
  SPLIT_VAR(b)

#undef SPLIT_VAR

  double_double c(a * b);
  c.tail = (((a1 * b1 - c.head) + a1 * b2) + a2 * b1) + a2 * b2;
  return c;
}

inline
double_double double_double::mul(float a, float b)
{
  /* Compute double-double = float * float. */
  return double_double(static_cast<double>(a) * b);
}

inline
double_double& double_double::operator +=(const double_double& rhs)
{
  double_double lhs(*this);
  *this = lhs + rhs;
  return *this;
}

//-----------------

inline
double_double operator *(const double_double& a, float b)
{
  return a * static_cast<double>(b);
}

inline
double_double operator *(const double_double& a, double b)
{
  /* Compute double-double = double-double * double. */
  double a11, a21, b1, b2, c11, c21, c2, con, t1, t2;

  con = a.head * double_double::split;
  a11 = con - a.head;
  a11 = con - a11;
  a21 = a.head - a11;

  con = b * double_double::split;
  b1 = con - b;
  b1 = con - b1;
  b2 = b - b1;

  c11 = a.head * b;
  c21 = (((a11 * b1 - c11) + a11 * b2) + a21 * b1) + a21 * b2;

  c2 = a.tail * b;
  t1 = c11 + c2;
  t2 = (c2 - (t1 - c11)) + c21;

  double_double c(t1 + t2);
  c.tail = t2 - (c.head - t1);
  return c;
}

inline
double_double operator +(const double_double& a, const double_double& b)
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
  double_double c(t1 + t2);
  c.tail = t2 - (c.head - t1);
  return c;
}

//-----------------
} //namespace XBLAS
//-----------------

#endif // XBLAS_DOUBLE_DOUBLE_HPP
