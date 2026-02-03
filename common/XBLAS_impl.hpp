#ifndef XBLAS_IMPL_HPP
#define XBLAS_IMPL_HPP

#include "common/XBLAS_double_double.hpp"

#include <type_traits>
#include <complex>

//---------------
namespace XBLAS {
//---------------

//--------------
namespace impl {
//--------------

//---------------------------
// IS_COMPLEX

template<typename T>
struct is_complex { static constexpr auto value = false; };

template<typename T>
struct is_complex<std::complex<T>> { static constexpr auto value = true; };

template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

//---------------------------
// INNER_TYPE
// T or T::value_type

template<typename T>
struct inner_type { using type = T; };

template<typename T>
requires is_complex_v<T>
struct inner_type<T> { using type = T::value_type; };

template<typename T>
using inner_type_t = inner_type<T>::type;

//---------------------------
// SIZE_LE

template<typename X, typename Y>
struct size_le { static constexpr auto value = sizeof(inner_type_t<X>) <= sizeof(inner_type_t<Y>); };

template<typename X, typename Y>
inline constexpr bool size_le_v = size_le<X, Y>::value;

//---------------------------
// GET_INNER_TYPE
// T or T::value_type, depending on types X and Y

template<typename X,
         typename Y,
         typename T>
struct get_inner_type { using type = T; };

// if X and Y are both real, we want the Z in T = complex<Z>
template<typename X,
         typename Y,
         typename T>
requires (std::floating_point<X> && std::floating_point<Y>)
struct get_inner_type<X, Y, T> { using type = inner_type_t<T>; };

template<typename X,
         typename Y,
         typename T>
using get_inner_type_t = get_inner_type<X, Y, T>::type;

//-------------------------------------

template<typename T, int prec>
struct internal_precision { using type = T; };

template<>
struct internal_precision<float, blas_prec_double> { using type = double; };
template<>
struct internal_precision<float, blas_prec_indigenous> { using type = double; };
template<>
struct internal_precision<float, blas_prec_extra> { using type = double_double; };

template<>
struct internal_precision<double, blas_prec_extra> { using type = double_double; };

template<>
struct internal_precision<std::complex<float>, blas_prec_double> { using type = std::complex<double>; };
template<>
struct internal_precision<std::complex<float>, blas_prec_indigenous> { using type = std::complex<double>; };
template<>
struct internal_precision<std::complex<float>, blas_prec_extra> { using type = std::complex<double_double>; };

template<>
struct internal_precision<std::complex<double>, blas_prec_extra> { using type = std::complex<double_double>; };

template<typename T, int prec>
using internal_precision_t = typename internal_precision<T, prec>::type;

//-------------------------------------

template<typename C,
         typename A,
         typename B>
constexpr inline C mul(A a, B b)
{
  if constexpr (std::is_same_v<C, A> &&
                std::is_same_v<C, B>) {
    return a * b;

  } else if constexpr (std::is_same_v<C, double_double> &&
                       std::floating_point<A> &&
                       std::floating_point<B>) {
    return double_double::mul(a, b);

  } else if constexpr (std::is_same_v<C, std::complex<double_double>>) {
    if constexpr (std::is_same_v<A, std::complex<float>> &&
                  std::is_same_v<B, std::complex<float>>) {
      /* Compute complex-extra = complex-float * complex-float. */
      /* Real part */
      double d1 = static_cast<double>(real(a)) * real(b);
      double d2 = static_cast<double>(-imag(a)) * imag(b);
      double_double cr = double_double::add(d1, d2); /* ar*br - ai*bi */
      /* imaginary part */
      d1 = static_cast<double>(real(a)) * imag(b);
      d2 = static_cast<double>(imag(a)) * real(b);
      double_double ci = double_double::add(d1, d2); /* ar*bi + ai*br */
      return std::complex<double_double>(cr, ci);

    } else if constexpr (std::is_same_v<A, std::complex<float>> &&
                         std::is_same_v<B, std::complex<double>>) {
      return mul<std::complex<double_double>>(static_cast<std::complex<double>>(a), b);

    } else if constexpr (std::is_same_v<A, std::complex<double>> &&
                         std::is_same_v<B, std::complex<float>>) {
      return mul<std::complex<double_double>>(a, static_cast<std::complex<double>>(b));

    } else if constexpr (std::is_same_v<A, std::complex<double>> &&
                         std::is_same_v<B, std::complex<double>>) {
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

    // complex<double_double> mul(complex<T>, T), T is floating point
    } else if constexpr (is_complex_v<A> && std::floating_point<inner_type_t<A>> &&
                         std::is_same_v<inner_type_t<A>, B>) {
      return std::complex<double_double>(double_double::mul(real(a), b), double_double::mul(imag(a), b));

    // complex<double_double> mul(T, complex<T>), T is floating point
    } else if constexpr (is_complex_v<B> && std::floating_point<inner_type_t<B>> &&
                         std::is_same_v<inner_type_t<B>, A>) {
      return mul<std::complex<double_double>>(b, a);

    } else if constexpr (std::is_same_v<C, A> && is_complex_v<B>) {
      /* Real part */
      auto d1 = real(a) * real(b);
      auto d2 = imag(a) * -imag(b);
      double_double cr = d1 + d2; /* ar*br - ai*bi */
      /* imaginary part */
      d1 = real(a) * imag(b);
      d2 = imag(a) * real(b);
      double_double ci = d1 + d2; /* ar*bi + ai*br */
      return std::complex<double_double>(cr, ci);

    } else if constexpr (is_complex_v<A> && std::is_same_v<C, B>) {
      return mul<std::complex<double_double>>(b, a);

    } else if constexpr (std::is_same_v<A, double_double> &&
                         is_complex_v<B> && std::floating_point<inner_type_t<B>>) {
      return std::complex<double_double>(a * real(b), a * imag(b));

    } else {
      static_assert(false, "Missing else if in XBLAS::impl::mul");
    }

  // All complex<double_doubles> should have been handled above.
  } else if constexpr (is_complex_v<C> &&
                       is_complex_v<A> &&
                       std::floating_point<B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    static_assert(!std::is_same_v<C, double_double>);
    return static_cast<C>(a) * static_cast<inner_type_t<C>>(b);

  } else if constexpr (is_complex_v<C> &&
                       std::floating_point<A> &&
                       is_complex_v<B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    static_assert(!std::is_same_v<C, double_double>);
    return static_cast<inner_type_t<C>>(a) * static_cast<C>(b);

  } else if constexpr (is_complex_v<C> &&
                       is_complex_v<A> &&
                       is_complex_v<B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    static_assert(!std::is_same_v<C, double_double>);
    return static_cast<C>(a) * static_cast<C>(b);

  // At this point C, A, B should all be real (possibly double_double), not complex.
  } else if constexpr (std::is_same_v<C, A> || std::is_same_v<C, B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    return a * b;

  } else {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    static_assert(!std::is_same_v<C, double_double>);
    return static_cast<C>(a) * b;
  }
}

//-------------------------------------

template<typename To,
         typename From>
constexpr inline To to(From from)
{
  return static_cast<To>(from);
}

template<typename To>
constexpr inline To to(double_double from)
{
  return static_cast<To>(from.head);
}

template<typename To>
requires is_complex_v<To>
constexpr inline To to(std::complex<double_double> from)
{
  return To(to<typename To::value_type>(real(from)), to<typename To::value_type>(imag(from)));
}

//---------------------------
// CONJ
// use nothing (real) or conj (complex)

struct Conj
{
    template<typename T> // for real types
    static constexpr auto func(T x) { return x; }

    template<typename T> // for complex types
    static constexpr auto func(const std::complex<T>& z) { return conj(z); }
};

//---------------------------
// ZERO
// return 0 with the correct type

template<typename T>
struct zero { static constexpr auto value = T(0); };

template<typename T>
struct zero<std::complex<T>> { static constexpr auto value = std::complex<T>(T(0), T(0)); };

template<typename T>
inline constexpr auto zero_v = zero<T>::value;

//-----------------
} // namespace impl
//-----------------

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_IMPL_HPP
