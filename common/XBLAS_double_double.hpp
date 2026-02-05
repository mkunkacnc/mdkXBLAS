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
public:
    constexpr double_double() {}
    constexpr explicit double_double(double x);
    constexpr double_double(double h, double t);

    static constexpr double_double add(double a, double b);

    static constexpr double_double mul(float a, float b);
    static constexpr double_double mul(float a, double b);
    static constexpr double_double mul(double a, float b);
    static constexpr double_double mul(double a, double b);
    static constexpr std::complex<double_double> mul(std::complex<float> a, std::complex<float> b);
    static constexpr std::complex<double_double> mul(std::complex<float> a, std::complex<double> b);
    static constexpr std::complex<double_double> mul(std::complex<double> a, std::complex<float> b);
    static constexpr std::complex<double_double> mul(std::complex<double> a, std::complex<double> b);

    template<typename T>
    requires std::floating_point<T>
    static constexpr std::complex<double_double> mul(std::complex<T> a, T b);

    template<typename T>
    requires std::floating_point<T>
    static constexpr std::complex<double_double> mul(T a, std::complex<T> b);

    template<typename T>
    requires std::floating_point<T>
    static constexpr std::complex<double_double> mul(std::complex<double_double> a, std::complex<T> b) { return a * b; }

    template<typename T>
    requires std::floating_point<T>
    static constexpr std::complex<double_double> mul(std::complex<T> a, std::complex<double_double> b) { return a * b; }

    template<typename T>
    requires std::floating_point<T>
    static constexpr std::complex<double_double> mul(double_double a, std::complex<T> b) { return a * b; }

    constexpr double_double& operator +=(const double_double& rhs);

public:
    static constexpr double split = SPLIT;

    double head;
    double tail;
};

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

constexpr double_double operator +(const double_double& a, const double_double& b);

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
constexpr std::complex<double_double> double_double::mul(std::complex<float> a, std::complex<float> b)
{
  /* Compute complex-extra = complex-float * complex-float. */
  /* Real part */
  double d1 = static_cast<double>(std::real(a)) * std::real(b);
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

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> double_double::mul(std::complex<T> a, T b)
{
  // complex<double_double> mul(complex<T>, T), T is floating point
  return std::complex<double_double>(double_double::mul(std::real(a), b), double_double::mul(std::imag(a), b));
}

template<typename T>
requires std::floating_point<T>
constexpr std::complex<double_double> double_double::mul(T a, std::complex<T> b)
{
  // complex<double_double> mul(T, complex<T>), T is floating point
  return double_double::mul(b, a);
}

inline
constexpr double_double& double_double::operator +=(const double_double& rhs)
{
  double_double lhs(*this);
  *this = lhs + rhs;
  return *this;
}

//-----------------

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

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_DOUBLE_DOUBLE_HPP
