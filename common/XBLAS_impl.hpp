#ifndef XBLAS_IMPL_HPP
#define XBLAS_IMPL_HPP

#include "common/XBLAS_double_double.hpp"

#include <complex>
#include <type_traits>

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
// USES_DOUBLE_DOUBLE

template<typename T>
struct uses_double_double { static constexpr auto value = std::is_same_v<inner_type_t<T>, double_double>; };

template<typename T>
inline constexpr bool uses_double_double_v = uses_double_double<T>::value;

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
// INTERNAL_PRECISION

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
// MUL

template<typename C,
         typename A,
         typename B>
constexpr inline C mul(A a, B b)
{
  // C mul(C, C)
  if constexpr (std::is_same_v<C, A> &&
                std::is_same_v<C, B>) {
    return a * b;

  // double_double mul(A, B), A, B floating-point
  } else if constexpr (std::is_same_v<C, double_double> &&
                       std::floating_point<A> &&
                       std::floating_point<B>) {
    return double_double::mul(a, b);

  // complex<double_double> mul(A, B), A, B real, complex, double_double, etc.
  } else if constexpr (std::is_same_v<C, std::complex<double_double>>) {
    return double_double::mul(a, b);

  // All complex<double_double> should have been handled above.
  // complex<T> mul(complex<U>, B)
  } else if constexpr (is_complex_v<C> &&
                       is_complex_v<A> &&
                       std::floating_point<B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    return static_cast<C>(a) * static_cast<inner_type_t<C>>(b);

  // complex<T> mul(A, complex<U>)
  } else if constexpr (is_complex_v<C> &&
                       std::floating_point<A> &&
                       is_complex_v<B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    return static_cast<inner_type_t<C>>(a) * static_cast<C>(b);

  // complex<T> mul(complex<U>, complex<V>)
  } else if constexpr (is_complex_v<C> &&
                       is_complex_v<A> &&
                       is_complex_v<B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    return static_cast<C>(a) * static_cast<C>(b);

  // At this point C, A, B should all be real (possibly double_double), not complex.
  // C mul(C, B) or C mul(A, C)
  } else if constexpr (std::is_same_v<C, A> || std::is_same_v<C, B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    return a * b;

  // C mul(A, B)
  } else {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    static_assert(!std::is_same_v<C, double_double>);
    return static_cast<C>(a) * b;
  }
}

//-------------------------------------
// TO

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
  return To(to<inner_type_t<To>>(std::real(from)), to<inner_type_t<To>>(std::imag(from)));
}

//-------------------------------------
// ADD

template<typename C,
         typename A,
         typename B>
constexpr inline C add(A a, B b)
{
  if constexpr (std::floating_point<C> &&
                std::is_same_v<A, double_double> &&
                std::is_same_v<B, double_double>) {
    return double_double::add(a, b);

  } else if constexpr (is_complex_v<C> &&
                       std::floating_point<inner_type_t<C>> &&
                       std::is_same_v<A, std::complex<double_double>> &&
                       std::is_same_v<B, std::complex<double_double>>) {
    return C(double_double::add(std::real(a), std::real(b)), double_double::add(std::imag(a), std::imag(b)));

  } else {
    return to<C>(a + b);
  }
}

//---------------------------
// CONJ
// use nothing (real) or conj (complex)

struct Conj
{
    template<typename T> // for real types
    static constexpr auto func(T x) { return x; }

    template<typename T> // for complex types
    static constexpr auto func(const std::complex<T>& z) { return std::conj(z); }
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
