#ifndef XBLAS_DOUBLE_DOUBLE_HPP
#define XBLAS_DOUBLE_DOUBLE_HPP

#include "blas_extended_private.h"

#include <complex>
#include <type_traits>

//---------------
namespace XBLAS {
//---------------

class double_double
{
  friend constexpr double_double operator +(const double_double& a, const double_double& b);
  friend constexpr double_double operator +(const double_double& a, double b);
  friend constexpr double_double operator +(const double_double& a, float b);

  friend constexpr double_double operator -(const double_double& a, const double_double& b);

  friend constexpr double_double operator *(const double_double& a, float b);
  friend constexpr double_double operator *(const double_double& a, double b);

  template<typename T>
  requires std::floating_point<T>
  friend constexpr std::complex<double_double> operator *(const std::complex<double_double>& a, const std::complex<T>& b);

  template<typename T>
  requires std::floating_point<T>
  friend constexpr std::complex<double_double> operator *(const std::complex<T>& a, const std::complex<double_double>& b);

  template<typename T>
  requires std::floating_point<T>
  friend constexpr std::complex<double_double> operator *(const double_double& a, const std::complex<T>& b);

  template<typename T>
  requires std::floating_point<T>
  friend constexpr std::complex<double_double> operator *(const std::complex<double_double>& a, T b);

  friend constexpr double_double operator /(const double_double& a, const double_double& b);
  friend constexpr double_double operator /(const double_double& a, double b);
  friend constexpr double_double operator /(const double_double& a, float b);

public:
  static constexpr double split = SPLIT;

  constexpr double_double() {}
  constexpr explicit double_double(double x);
  constexpr double_double(double h, double t);

  static constexpr double_double add(double a, double b);
  static constexpr double add(double_double a, double_double b);

  static constexpr double_double mul(float a, float b);
  static constexpr double_double mul(float a, double b);
  static constexpr double_double mul(double a, float b);
  static constexpr double_double mul(double a, double b);
  static constexpr std::complex<double_double> mul(std::complex<float> a, std::complex<float> b);
  static constexpr std::complex<double_double> mul(std::complex<float> a, std::complex<double> b);
  static constexpr std::complex<double_double> mul(std::complex<double> a, std::complex<float> b);
  static constexpr std::complex<double_double> mul(std::complex<double> a, std::complex<double> b);

  template<typename T, typename U>
  requires (std::floating_point<T> && std::floating_point<U>)
  static constexpr std::complex<double_double> mul(std::complex<T> a, U b);

  template<typename T, typename U>
  requires (std::floating_point<T> && std::floating_point<U>)
  static constexpr std::complex<double_double> mul(T a, std::complex<U> b);

  template<typename T>
  requires std::floating_point<T>
  static constexpr std::complex<double_double> mul(std::complex<double_double> a, std::complex<T> b) { return a * b; }

  template<typename T>
  requires std::floating_point<T>
  static constexpr std::complex<double_double> mul(std::complex<T> a, std::complex<double_double> b) { return a * b; }

  template<typename T>
  requires std::floating_point<T>
  static constexpr std::complex<double_double> mul(double_double a, std::complex<T> b) { return a * b; }

  template<typename T>
  requires std::floating_point<T>
  static constexpr std::complex<double_double> mul(std::complex<double_double> a, T b) { return a * b; }

  constexpr double_double& operator +=(const double_double& rhs);
  constexpr double_double& operator +=(double rhs);
  constexpr double_double& operator +=(float rhs);

  constexpr double_double& operator -=(const double_double& rhs);

  constexpr double to_double() const { return head; }

private:
  double head;
  double tail;
};

constexpr double_double operator +(const double_double& a, const double_double& b);
constexpr double_double operator +(const double_double& a, double b);
constexpr double_double operator +(double a, const double_double& b);
constexpr double_double operator +(const double_double& a, float b);

constexpr double_double operator -(const double_double& a, const double_double& b);

constexpr double_double operator *(const double_double& a, float b);
constexpr double_double operator *(const double_double& a, double b);

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> operator *(const std::complex<double_double>& a, const std::complex<T>& b);

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> operator *(const std::complex<T>& a, const std::complex<double_double>& b);

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> operator *(const double_double& a, const std::complex<T>& b);

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> operator *(const std::complex<double_double>& a, T b);

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double>& operator *=(std::complex<double_double>& a, T b);

constexpr double_double operator /(const double_double& a, const double_double& b);
constexpr double_double operator /(const double_double& a, double b);
constexpr double_double operator /(const double_double& a, float b);

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> operator /(const std::complex<double_double>& a, T b);

constexpr std::complex<double_double>& operator /=(std::complex<double_double>& a, double b);

//-----------------

inline
constexpr double_double::double_double(double x)
:head(x),
 tail(0)
{}

inline
constexpr double_double::double_double(double h, double t)
:head(h),
 tail(t)
{}

inline
constexpr double_double double_double::add(double a, double b)
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
constexpr double double_double::add(double_double a, double_double b)
{
  /* Compute double = double-double + double-double. */
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

  /* Renormalize (t1, t2) */
  return t1 + t2;
}

inline
constexpr double_double double_double::mul(float a, float b)
{
  /* Compute double-double = float * float. */
  return double_double(static_cast<double>(a) * b);
}

inline
constexpr double_double double_double::mul(float a, double b)
{
  /* Compute double-double = float * double. */
  return double_double::mul(static_cast<double>(a), b);
}

inline
constexpr double_double double_double::mul(double a, float b)
{
  /* Compute double-double = double * float. */
  return double_double::mul(b, a);
}

inline
constexpr double_double double_double::mul(double a, double b)
{
  /* Compute double-double = double * double. */
  double a1, a2, b1, b2, con;

#define SPLIT_VAR(a)                \
  con = a * double_double::split;   \
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
constexpr std::complex<double_double> double_double::mul(std::complex<float> a, std::complex<float> b)
{
  /* Compute complex-extra = complex-float * complex-float. */
  /* Real part */
  double d1 = static_cast<double>( std::real(a)) * std::real(b);
  double d2 = static_cast<double>(-std::imag(a)) * std::imag(b);
  double_double cr = double_double::add(d1, d2); /* ar*br - ai*bi */
  /* Imaginary part */
  d1 = static_cast<double>(std::real(a)) * std::imag(b);
  d2 = static_cast<double>(std::imag(a)) * std::real(b);
  double_double ci = double_double::add(d1, d2); /* ar*bi + ai*br */
  return std::complex<double_double>(cr, ci);
}

inline
constexpr std::complex<double_double> double_double::mul(std::complex<float> a, std::complex<double> b)
{
  return double_double::mul(static_cast<std::complex<double>>(a), b);
}

inline
constexpr std::complex<double_double> double_double::mul(std::complex<double> a, std::complex<float> b)
{
  return double_double::mul(a, static_cast<std::complex<double>>(b));
}

inline
constexpr std::complex<double_double> double_double::mul(std::complex<double> a, std::complex<double> b)
{
  /* Compute complex-extra = complex-double * complex-double. */
  /* Real part */
  double_double t1 = double_double::mul( std::real(a), std::real(b));
  double_double t2 = double_double::mul(-std::imag(a), std::imag(b));
  double_double cr = t1 + t2; /* ar*br - ai*bi */
  /* Imaginary part */
  t1 = double_double::mul(std::imag(a), std::real(b));
  t2 = double_double::mul(std::real(a), std::imag(b));
  double_double ci = t1 + t2; /* ar*bi + ai*br */
  return std::complex<double_double>(cr, ci);
}

template<typename T, typename U>
requires (std::floating_point<T> && std::floating_point<U>)
constexpr std::complex<double_double> double_double::mul(std::complex<T> a, U b)
{
  // complex<double_double> mul(complex<T>, U), T, U is floating point
  return std::complex<double_double>(double_double::mul(std::real(a), b), double_double::mul(std::imag(a), b));
}

template<typename T, typename U>
requires (std::floating_point<T> && std::floating_point<U>)
constexpr std::complex<double_double> double_double::mul(T a, std::complex<U> b)
{
  // complex<double_double> mul(T, complex<U>), T, U is floating point
  return double_double::mul(b, a);
}

inline
constexpr double_double& double_double::operator +=(const double_double& rhs)
{
  double_double lhs(*this);
  *this = lhs + rhs;
  return *this;
}

inline
constexpr double_double& double_double::operator +=(double rhs)
{
  double_double lhs(*this);
  *this = lhs + rhs;
  return *this;
}

inline
constexpr double_double& double_double::operator +=(float rhs)
{
  double_double lhs(*this);
  *this = lhs + rhs;
  return *this;
}

inline
constexpr double_double& double_double::operator -=(const double_double& rhs)
{
  double_double lhs(*this);
  *this = lhs - rhs;
  return *this;
}

//-----------------

inline
constexpr double_double operator +(const double_double& a, const double_double& b)
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

inline
constexpr double_double operator +(const double_double& a, double b)
{
  /* Compute double-double = double-double + double. */
  double e, t1, t2;

  /* Knuth trick. */
  t1 = a.head + b;
  e = t1 - a.head;
  t2 = ((b - e) + (a.head - (t1 - e))) + a.tail;

  /* The result is t1 + t2, after normalization. */
  double_double c(t1 + t2);
  c.tail = t2 - (c.head - t1);
  return c;
}

inline
constexpr double_double operator +(double a, const double_double& b)
{
  return b + a;
}

inline
constexpr double_double operator +(const double_double& a, float b)
{
  return a + static_cast<double>(b);
}

inline
constexpr double_double operator -(const double_double& a, const double_double& b)
{
  double_double rhs(b);
  rhs.head = -rhs.head;
  rhs.tail = -rhs.tail;
  return a + rhs;
}

inline
constexpr double_double operator *(const double_double& a, float b)
{
  return a * static_cast<double>(b);
}

inline
constexpr double_double operator *(const double_double& a, double b)
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

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> operator *(const std::complex<double_double>& a, const std::complex<T>& b)
{
  // complex<double_double> mul(complex<double_double>, complex<T>)
  /* Real part */
  auto d1 = std::real(a) * std::real(b);
  auto d2 = std::imag(a) * -std::imag(b);
  double_double cr = d1 + d2; /* ar*br - ai*bi */
  /* Imaginary part */
  d1 = std::real(a) * std::imag(b);
  d2 = std::imag(a) * std::real(b);
  double_double ci = d1 + d2; /* ar*bi + ai*br */
  return std::complex<double_double>(cr, ci);
}

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> operator *(const std::complex<T>& a, const std::complex<double_double>& b)
{
  // complex<double_double> mul(complex<T>, complex<double_double>)
  return b * a;
}

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> operator *(const double_double& a, const std::complex<T>& b)
{
  // complex<double_double> mul(double_double, complex<T>)
  return std::complex<double_double>(a * std::real(b), a * std::imag(b));
}

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> operator *(const std::complex<double_double>& a, T b)
{
  // complex<double_double> mul(complex<double_double>, T)
  return std::complex<double_double>(std::real(a) * b, std::imag(a) * b);
}

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double>& operator *=(std::complex<double_double>& a, T b)
{
  a = a * b;
  return a;
}

inline
constexpr double_double operator /(const double_double& a, const double_double& b)
{
  double q1, q2, q3;
  double a1, a2, b1, b2;
  double p1, p2, c;
  double s1, s2, v;
  double t1, t2;
  double r1, r2;
  double cona, conb;

  q1 = a.head / b.head;        /*  approximate quotient */

  /*  Compute  q1 * b  */
  cona = q1 * double_double::split;
  conb = b.head * double_double::split;
  a1 = cona - (cona - q1);
  b1 = conb - (conb - b.head);
  a2 = q1 - a1;
  b2 = b.head - b1;

  /*  (p1, p2) is the product of high order terms. */
  p1 = q1 * b.head;
  p2 = (((a1 * b1 - p1) + a1 * b2) + a2 * b1) + a2 * b2;

  /*  Compute the low-order term */
  c = q1 * b.tail;

  /*  Compute  (s1, s2) = (p1, p2) + c */
  s1 = p1 + c;
  v = s1 - p1;
  s2 = ((c - v) + (p1 - (s1 - v))) + p2;

  /*  Renormalize. */
  p1 = s1 + s2;
  p2 = s2 - (p1 - s1);

  /*  Compute  a - (p1, p2) */
  s1 = a.head - p1;
  v = s1 - a.head;
  s2 = (a.head - (s1 - v)) - (p1 + v);

  t1 = a.tail - p2;
  v = t1 - a.tail;
  t2 = (a.tail - (t1 - v)) - (p2 + v);

  s2 += t1;
  t1 = s1 + s2;
  s2 = s2 - (t1 - s1);

  t2 += s2;
  r1 = t1 + t2;
  r2 = t2 - (r1 - t1);

  /*  Compute the next quotient. */
  q2 = r1 / b.head;

  /*  Compute residual   r1 - q2 * b */
  cona = q2 * double_double::split;
  a1 = cona - (cona - q2);
  a2 = q2 - a1;

  /*  (p1, p2) is the product of high order terms. */
  p1 = q2 * b.head;
  p2 = (((a1 * b1 - p1) + a1 * b2) + a2 * b1) + a2 * b2;

  /*  Compute the low-order term */
  c = q2 * b.tail;

  /*  Compute  (s1, s2) = (p1, p2) + c */
  s1 = p1 + c;
  v = s1 - p1;
  s2 = ((c - v) + (p1 - (s1 - v))) + p2;

  /*  Renormalize. */
  p1 = s1 + s2;
  p2 = s2 - (p1 - s1);

  /*  Compute  (r1, r2) - (p1, p2)    */
  s1 = r1 - p1;
  v = s1 - r1;
  s2 = (r1 - (s1 - v)) - (p1 + v);

  t1 = r2 - p2;
  v = t1 - r2;
  t2 = (r2 - (t1 - v)) - (p2 + v);

  s2 += t1;
  t1 = s1 + s2;
  s2 = s2 - (t1 - s1);

  t2 += s2;
  s1 = t1 + t2;

  /*  Compute the last correction. */
  q3 = s1 / b.head;

  /* Renormalize q1, q2, q3. */
  s1 = q2 + q3;
  s2 = q3 - (s1 - q2);

  double_double quotient(q1 + s1);
  t1 = s1 - (quotient.head - q1);
  quotient.tail = s2 + t1;
  return quotient;
}

inline
constexpr double_double operator /(const double_double& a, double b)
{
  /* Compute double-double = double-double / double,
     using a Newton iteration scheme. */
  double b1, b2, con, e, t1, t2, t11, t21, t12, t22;

  /* Compute a DP approximation to the quotient. */
  t1 = a.head / b;

  /* Split t1 and b into two parts with at most 26 bits each,
     using the Dekker-Veltkamp method. */
  con = t1 * double_double::split;
  t11 = con - (con - t1);
  t21 = t1 - t11;
  con = b * double_double::split;
  b1 = con - (con - b);
  b2 = b - b1;

  /* Compute t1 * b using Dekker method. */
  t12 = t1 * b;
  t22 = (((t11 * b1 - t12) + t11 * b2) + t21 * b1) + t21 * b2;

  /* Compute dda - (t12, t22) using Knuth trick. */
  t11 = a.head - t12;
  e = t11 - a.head;
  t21 = ((-t12 - e) + (a.head - (t11 - e))) + a.tail - t22;

  /* Compute high-order word of (t11, t21) and divide by b. */
  t2 = (t11 + t21) / b;

  /* The result is t1 + t2, after normalization. */
  double_double c(t1 + t2);
  c.tail = t2 - (c.head - t1);
  return c;
}

inline
constexpr double_double operator /(const double_double& a, float b)
{
  return a / static_cast<double>(b);
}

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> operator /(const std::complex<double_double>& a, T b)
{
  return std::complex<double_double>(std::real(a) / b, std::imag(a) / b);
}

inline
constexpr std::complex<double_double>& operator /=(std::complex<double_double>& a, double b)
{
  a = a / b;
  return a;
}

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_DOUBLE_DOUBLE_HPP
