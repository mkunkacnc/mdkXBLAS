#ifndef XBLAS_IMPL_HPP
#define XBLAS_IMPL_HPP

#include "common/XBLAS_double_double.hpp"

#include <type_traits>
#include <complex>

//---------------
namespace XBLAS {
//---------------

template<typename T>
concept has_value_type = requires(T t)  {
  typename T::value_type;
};

//-------------------------------------

template<typename C,
         typename A,
         typename B>
requires (!std::is_same_v<C, A> || !std::is_same_v<C, B>)
inline C mul(A a, B b)
{
  return static_cast<C>(a) * b;
}

template<typename C>
inline C mul(C a, C b)
{
  return a * b;
}

template<typename C,
         typename A,
         typename B>
requires (std::floating_point<B> &&
          has_value_type<C> &&
          !std::is_same_v<typename C::value_type, B> &&
          !std::is_same_v<typename C::value_type, double_double>)
inline C mul(std::complex<A> a, B b)
{
  return mul<C>(a, static_cast<C::value_type>(b));
}

template<typename C,
         typename A,
         typename B>
requires (has_value_type<C> && !std::is_same_v<typename C::value_type, B>)
inline C mul(std::complex<A> a, std::complex<B> b)
{
  return mul<C>(a, static_cast<C>(b));
}

template<>
inline double_double mul(double a, double b)
{
  return double_double::mul(a, b);
}

template<>
inline double_double mul(double a, float b)
{
  return double_double::mul(a, static_cast<double>(b));
}

template<>
inline double_double mul(float a, float b)
{
  return double_double::mul(a, b);
}

template<>
inline std::complex<double_double> mul(std::complex<float> a, std::complex<float> b)
{
  /* Real part */
  double d1 = static_cast<double>(real(a)) * real(b);
  double d2 = static_cast<double>(-imag(a)) * imag(b);
  double_double cr = double_double::add(d1, d2); /* ar*br - ai*bi */
  /* imaginary part */
  d1 = static_cast<double>(real(a)) * imag(b);
  d2 = static_cast<double>(imag(a)) * real(b);
  double_double ci = double_double::add(d1, d2); /* ar*bi + ai*br */
  return std::complex<double_double>(cr, ci);
}

template<>
inline std::complex<double_double> mul(std::complex<double> a, std::complex<double> b)
{
  /* Compute complex-extra = complex-double * complex-double. */
  /* Real part */
  double_double t1 = double_double::mul( real(a), real(b));
  double_double t2 = double_double::mul(-imag(a), imag(b));
  double_double cr = t1 + t2; /* ar*br - ai*bi */
  /* Imaginary part */
  t1 = double_double::mul(imag(a), real(b));
  t2 = double_double::mul(real(a), imag(b));
  double_double ci = t1 + t2; /* ar*bi + ai*br */
  return std::complex<double_double>(cr, ci);
}

template<>
inline std::complex<double_double> mul(std::complex<double> a, std::complex<float> b)
{
  return mul<std::complex<double_double>>(a, static_cast<std::complex<double>>(b));
}

template<>
inline std::complex<double_double> mul(std::complex<float> a, float b)
{
  return std::complex<double_double>(double_double::mul(real(a), b), double_double::mul(imag(a), b));
}

template<>
inline std::complex<double_double> mul(std::complex<double> a, double b)
{
  return std::complex<double_double>(double_double::mul(real(a), b), double_double::mul(imag(a), b));
}

//-------------------------------------

template<typename To,
         typename From>
inline To to(From from)
{
  return static_cast<To>(from);
}

template<typename To>
inline To to(double_double from)
{
  return static_cast<To>(from.head);
}

template<typename To>
requires has_value_type<To>
inline To to(std::complex<double_double> from)
{
  return To(to<typename To::value_type>(real(from)), to<typename To::value_type>(imag(from)));
}

//-----------------
} //namespace XBLAS
//-----------------

#endif // XBLAS_IMPL_HPP
