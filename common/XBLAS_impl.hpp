#ifndef XBLAS_IMPL_HPP
#define XBLAS_IMPL_HPP

// Need at least C++23
static_assert(__cplusplus >= 202302L, "Need at least C++23");
// Both of these may be defined starting with C++23
//static_assert(__STDCPP_FLOAT16_T__, "Need float16");
//static_assert(__STDCPP_FLOAT128_T__, "Need float128");

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
  // complex<T> mul(A, B)
  } else if constexpr (is_complex_v<C> &&
                       std::floating_point<A> &&
                       std::floating_point<B>) {
    static_assert(!std::is_same_v<C, std::complex<double_double>>);
    return static_cast<inner_type_t<C>>(a) * b;

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
requires std::floating_point<inner_type_t<To>>
constexpr inline To to(double_double from)
{
  return static_cast<To>(from.to_double());
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

  } else if constexpr (is_complex_v<C> &&
                       is_complex_v<A> &&
                       is_complex_v<B>) {
    return C(std::real(a) + std::real(b), std::imag(a) + std::imag(b));

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

template<int is_hermitian> // declaration, no implementation
struct Conj_h;

template<>       // implement full specialization
struct Conj_h<1> // hermitian, use the conj function appropriately
{
    template<typename T>
    static constexpr auto func(T x) { return Conj::func(x); }
};

template<>       // implement full specialization
struct Conj_h<0> // symmetric, make no changes.
{
    template<typename T>
    static constexpr auto func(T x) { return x; }
};

//---------------------------
// REAL
// use nothing (real) or real (complex)

struct Real
{
    template<typename T> // for real types
    static constexpr auto func(T x) { return x; }

    template<typename T> // for complex types
    static constexpr auto func(const std::complex<T>& z) { return std::real(z); }
};

template<int is_hermitian> // declaration, no implementation
struct Real_h;

template<>       // implement full specialization
struct Real_h<1> // hermitian, use the real function appropriately
{
    template<typename T>
    static constexpr auto func(T x) { return Real::func(x); }
};

template<>       // implement full specialization
struct Real_h<0> // symmetric, make no changes.
{
    template<typename T>
    static constexpr auto func(T x) { return x; }
};

//---------------------------
// IMAG
// use zero (real) or imag (complex)

struct Imag
{
    template<typename T> // for real types
    static constexpr auto func(T /*x*/) { return T{0}; }

    template<typename T> // for complex types
    static constexpr auto func(const std::complex<T>& z) { return std::imag(z); }
};

template<int is_hermitian> // declaration, no implementation
struct Imag_h;

template<>       // implement full specialization
struct Imag_h<1> // hermitian, use the conj function appropriately
{
    template<typename T>
    static constexpr auto func(T x) { return Imag::func(x); }
};

template<>       // implement full specialization
struct Imag_h<0> // symmetric, make no changes.
{
    template<typename T>
    static constexpr auto func(T x) { return x; }
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

//---------------------------
// DIV

template<typename A,
         typename B>
constexpr A div(A a, B b)
{
  if constexpr (std::is_same_v<A, std::complex<double>> &&
                std::is_same_v<B, std::complex<double>>) {
    double S = 1.0, eps, ov, un;
    double abs_a, abs_b, abs_c, abs_d, ab, cd;
    double r;
    double t;
    double q[2];

    eps = std::pow(2.0, -24.0);
    un = std::pow(2.0, -126.0);
    ov = std::pow(2.0, 128.0) * (1 - eps);
    abs_a = std::fabs(std::real(a));
    abs_b = std::fabs(std::imag(a));
    abs_c = std::fabs(static_cast<double>(std::real(b)));
    abs_d = std::fabs(static_cast<double>(std::imag(b)));
    ab = std::max(abs_a, abs_b);
    cd = std::max(abs_c, abs_d);

    /* Scaling */
    if (ab > ov / 16) {        /* scale down a, b */
      a /= 16;
      S = S * 16;
    }
    if (cd > ov / 16) {        /* scale down c, d */
      b /= 16;
      S = S / 16;
    }
    if (ab < un / eps * 2) {        /* scale up a, b */
      t = 2.0 / (eps * eps);
      a *= t;
      S = S / t;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      t = 2.0 / (eps * eps);
      b *= t;
      S = S * t;
    }

    /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
    if (abs_c > abs_d) {
      r = std::imag(b) / std::real(b);
      t = 1 / (std::real(b) + std::imag(b) * r);
      q[0] = (std::real(a) + std::imag(a) * r) * t;
      q[1] = (std::imag(a) - std::real(a) * r) * t;
    } else {
      r = std::real(b) / std::imag(b);
      t = 1 / (std::imag(b) + std::real(b) * r);
      q[0] = ( std::imag(a) + std::real(a) * r) * t;
      q[1] = (-std::real(a) + std::imag(a) * r) * t;
    }
    /* Scale back */
    return A(q[0] * S, q[1] * S);

  } else if constexpr (std::is_same_v<A, std::complex<double>> &&
                       std::is_same_v<B, std::complex<float>>) {
    double S = 1.0, eps, ov, un, eps1, ov1, un1;
    double abs_a, abs_b, abs_c, abs_d, ab, cd;
    double r;
    double t;
    double q[2];

    eps = std::pow(2.0, -24.0);
    un = std::pow(2.0, -126.0);
    ov = std::pow(2.0, 128.0) * (1 - eps);
    eps1 = std::pow(2.0, -53.0);
    un1 = std::pow(2.0, -1022.0);
    ov1 = 1.79769313486231571e+308;
    /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0; */
    abs_a = std::fabs(std::real(a));
    abs_b = std::fabs(std::imag(a));
    abs_c = std::fabs(static_cast<double>(std::real(b)));
    abs_d = std::fabs(static_cast<double>(std::imag(b)));
    ab = std::max(abs_a, abs_b);
    cd = std::max(abs_c, abs_d);

    /* Scaling */
    if (ab > ov1 / 16) {        /* scale down a, b */
      a /= 16;
      S = S * 16;
    }
    if (cd > ov / 16) {        /* scale down c, d */
      b /= 16;
      S = S / 16;
    }
    if (ab < un1 / eps1 * 2) {        /* scale up a, b */
      t = 2.0 / (eps1 * eps1);
      a *= t;
      S = S / t;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      t = 2.0 / (eps * eps);
      b *= t;
      S = S * t;
    }

    /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
    if (abs_c > abs_d) {
      r = std::imag(b) / std::real(b);
      t = 1 / (std::real(b) + std::imag(b) * r);
      q[0] = (std::real(a) + std::imag(a) * r) * t;
      q[1] = (std::imag(a) - std::real(a) * r) * t;
    } else {
      r = std::real(b) / std::imag(b);
      t = 1 / (std::imag(b) + std::real(b) * r);
      q[0] = ( std::imag(a) + std::real(a) * r) * t;
      q[1] = (-std::real(a) + std::imag(a) * r) * t;
    }
    /* Scale back */
    return A(q[0] * S, q[1] * S);

  } else if constexpr (std::is_same_v<A, std::complex<float>> &&
                       std::is_same_v<B, std::complex<float>>) {
    double S = 1.0, eps, ov, un;
    double abs_a, abs_b, abs_c, abs_d, ab, cd;
    double r;
    double t;
    double q[2];

    eps = std::pow(2.0, -24.0);
    un = std::pow(2.0, -126.0);
    ov = std::pow(2.0, 128.0) * (1 - eps);
    abs_a = std::fabs(static_cast<double>(std::real(a)));
    abs_b = std::fabs(static_cast<double>(std::imag(a)));
    abs_c = std::fabs(static_cast<double>(std::real(b)));
    abs_d = std::fabs(static_cast<double>(std::imag(b)));
    ab = std::max(abs_a, abs_b);
    cd = std::max(abs_c, abs_d);

    /* Scaling */
    if (ab > ov / 16) {        /* scale down a, b */
      a /= 16;
      S = S * 16;
    }
    if (cd > ov / 16) {        /* scale down c, d */
      b /= 16;
      S = S / 16;
    }
    if (ab < un / eps * 2) {        /* scale up a, b */
      t = 2.0 / (eps * eps);
      a *= t;
      S = S / t;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      t = 2.0 / (eps * eps);
      b *= t;
      S = S * t;
    }

    /* Now un/eps*2 <= (a, b, c, d) >= ov/16 */
    if (abs_c > abs_d) {
      r = std::imag(b) / std::real(b);
      t = 1 / (std::real(b) + std::imag(b) * r);
      q[0] = (std::real(a) + std::imag(a) * r) * t;
      q[1] = (std::imag(a) - std::real(a) * r) * t;
    } else {
      r = std::real(b) / std::imag(b);
      t = 1 / (std::imag(b) + std::real(b) * r);
      q[0] = ( std::imag(a) + std::real(a) * r) * t;
      q[1] = (-std::real(a) + std::imag(a) * r) * t;
    }
    /* Scale back */
    return A(q[0] * S, q[1] * S);

  } else if constexpr (std::is_same_v<A, std::complex<double_double>> &&
                       std::is_same_v<B, std::complex<double>>) {
    double S = 1.0, eps, ov, un, eps1, ov1, un1;
    double abs_a, abs_b, abs_c, abs_d, ab, cd;
    double s;
    double r;
    double_double t, q[2];

    eps = std::pow(2.0, -53.0);        /* double precision */
    un = std::pow(2.0, -1022.0);
    ov = 1.79769313486231571e+308;
    /* = (pow(2.0, 1023.0) * (1 - eps)) * 2.0 */
    eps1 = std::pow(2.0, -104.0);        /* extra precision */
    un1 = std::pow(2.0, -1022.0);
    ov1 = 1.79769313486231571e+308;
    /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
    abs_a = std::fabs(to<double>(std::real(a)));
    abs_b = std::fabs(to<double>(std::imag(a)));
    abs_c = std::fabs(static_cast<double>(std::real(b)));
    abs_d = std::fabs(static_cast<double>(std::imag(b)));
    ab = std::max(abs_a, abs_b);
    cd = std::max(abs_c, abs_d);

    /* Scaling */
    if (ab > ov1 / 16) {        /* scale down a, b */
      a = a/16.0;
      S = S * 16;
    }
    if (cd > ov / 16) {        /* scale down c, d */
      b /= 16;
      S = S / 16;
    }
    if (ab < un1 / eps1 * 2) {        /* scale up a, b */
      s = 2.0 / (eps1 * eps1);
      a = a * s;
      S = S / s;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      s = 2.0 / (eps * eps);
      b *= s;
      S = S * s;
    }

    /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
    if (abs_c > abs_d) {
      r = std::imag(b) / std::real(b);
      t = std::real(b) + double_double::mul(std::imag(b), r);
      q[0] = (std::real(a) + std::imag(a) * r) / t;
      q[1] = (std::imag(a) - std::real(a) * r) / t;
    } else {
      r = std::real(b) / std::imag(b);
      t = std::imag(b) + double_double::mul(std::real(b), r);
      q[0] = (std::imag(a) + std::real(a) * r) / t;
      q[1] = (std::imag(a) * r - std::real(a)) / t;
    }

    /* Scale back */
    if (S == 1.0) {
      return A(q[0], q[1]);
    } else {
      return A(q[0] * S, q[1] * S);
    }

  } else if constexpr (std::is_same_v<A, std::complex<double_double>> &&
                       std::is_same_v<B, std::complex<float>>) {
    double S = 1.0, eps, ov, un, eps1, ov1, un1;
    double abs_a, abs_b, abs_c, abs_d, ab, cd;
    double s;
    double r;
    double_double t, q[2];

    eps = std::pow(2.0, -24.0);        /* single precision */
    un = std::pow(2.0, -126.0);
    ov = std::pow(2.0, 128.0) * (1 - eps);
    eps1 = std::pow(2.0, -104.0);        /* extra precision */
    un1 = std::pow(2.0, -1022.0);
    ov1 = 1.79769313486231571e+308;
    /* = (pow(2.0, 1023.0) * (1 - eps1)) * 2.0 */
    abs_a = std::fabs(to<double>(std::real(a)));
    abs_b = std::fabs(to<double>(std::imag(a)));
    abs_c = std::fabs(static_cast<double>(std::real(b)));
    abs_d = std::fabs(static_cast<double>(std::imag(b)));
    ab = std::max(abs_a, abs_b);
    cd = std::max(abs_c, abs_d);

    /* Scaling */
    if (ab > ov1 / 16) {        /* scale down a, b */
      a = a/16.0;
      S = S * 16;
    }
    if (cd > ov / 16) {        /* scale down c, d */
      b /= 16;
      S = S / 16;
    }
    if (ab < un1 / eps1 * 2) {        /* scale up a, b */
      s = 2.0 / (eps1 * eps1);
      a = a * s;
      S = S / s;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      s = 2.0 / (eps * eps);
      b *= s;
      S = S * s;
    }

    /* Now un1/eps1*2 <= (a,b) >= ov1/16, un/eps*2 <= (c,d) >= ov/16 */
    if (abs_c > abs_d) {
      r = std::imag(b) / std::real(b);
      t = std::real(b) + double_double::mul(std::imag(b), r);
      q[0] = (std::real(a) + std::imag(a) * r) / t;
      q[1] = (std::imag(a) - std::real(a) * r) / t;
    } else {
      r = std::real(b) / std::imag(b);
      t = std::imag(b) + double_double::mul(std::real(b), r);
      q[0] = (std::imag(a) + std::real(a) * r) / t;
      q[1] = (std::imag(a) * r - std::real(a)) / t;
    }

    /* Scale back */
    if (S == 1.0) {
      return A(q[0], q[1]);
    } else {
      return A(q[0] * S, q[1] * S);
    }

  } else if constexpr (std::is_same_v<inner_type_t<A>, double_double>) {
    return a / b;

  } else if constexpr (impl::is_complex_v<A>) {
    static_assert(!impl::is_complex_v<B>);
    return a / static_cast<impl::inner_type_t<A>>(b);

  } else {
    return a / static_cast<A>(b);
  }
}

//-----------------
} // namespace impl
//-----------------

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_IMPL_HPP
