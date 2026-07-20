#ifndef XBLAS_IMPL_HPP
#define XBLAS_IMPL_HPP

// Need at least C++23
static_assert(__cplusplus >= 202302L, "Need at least C++23");

#ifdef XBLAS_USE_FLOAT128
static_assert(__STDCPP_FLOAT128_T__, "Need std::float128_t");

#include <stdfloat>

using XBLAS_X_t = std::float128_t;

// Dummy implementation
#define FPU_FIX_DECL  static_cast<void>(0)
#define FPU_FIX_START static_cast<void>(0)
#define FPU_FIX_STOP  static_cast<void>(0)

// Dummy implementation
namespace XBLAS {
struct double_double
{
  template<typename A, typename B>
  static double add(A a, B b) { return 0; }

  template<typename A, typename B>
  static double mul(A a, B b) { return 0; }

  double to_double() const { return 0; }
};
} // namespace XBLAS

#else
#include "double_double.hpp"

using XBLAS_X_t = XBLAS::double_double;
#endif

#ifdef XBLAS_USE_INT64_INTERNAL_INDEX_TYPE
#include <cstdint>
#endif

#include <cmath>
#include <complex>
#include <limits>
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
// TO_TYPE

template<typename From>
struct to_type { using type = From; };

template<typename From>
requires std::is_same_v<From, double_double>
struct to_type<From> { using type = double; };

template<typename From>
using to_type_t = to_type<From>::type;

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

//-------------------------------------
// INTERNAL_PRECISION

template<typename T, int prec>
struct internal_precision { using type = T; };

template<>
struct internal_precision<float, blas_prec_double> { using type = double; };
template<>
struct internal_precision<float, blas_prec_indigenous> { using type = double; };
template<>
struct internal_precision<float, blas_prec_extra> { using type = XBLAS_X_t; };

template<>
struct internal_precision<double, blas_prec_extra> { using type = XBLAS_X_t; };

template<>
struct internal_precision<std::complex<float>, blas_prec_double> { using type = std::complex<double>; };
template<>
struct internal_precision<std::complex<float>, blas_prec_indigenous> { using type = std::complex<double>; };
template<>
struct internal_precision<std::complex<float>, blas_prec_extra> { using type = std::complex<XBLAS_X_t>; };

template<>
struct internal_precision<std::complex<double>, blas_prec_extra> { using type = std::complex<XBLAS_X_t>; };

template<typename T, int prec>
using internal_precision_t = typename internal_precision<T, prec>::type;

//-------------------------------------
// INTERNAL_INDEX_TYPE

template<typename N>
struct internal_index_type {
#ifdef XBLAS_USE_INT64_INTERNAL_INDEX_TYPE
   using type = std::int64_t;
#else
   using type = N;
#endif
};

template<typename N>
using internal_index_type_t = typename internal_index_type<N>::type;

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
    return C(to<inner_type_t<C>>(std::real(a) + std::real(b)), to<inner_type_t<C>>(std::imag(a) + std::imag(b)));

  //} else if TODO case of sizeof C > A or B ? Is that ever the case or is sizeof C <= A and B?

  } else {
    return to<C>(a + b);
  }
}

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

//---------------------------
// DIV

template<typename A,
         typename B>
constexpr A do_complex_div(A a, B b)
{
  /*
    Textbook complex division:
    a + ib   (a + ib)(c - id)   ac + bd     bc - ad
    ------ = ---------------- = ------- + i -------
    c + id   (c + id)(c - id)   cc + dd     cc + dd

    If any of the numbers are large, there is a great risk of
    overflow during the intermediate calculations even if the
    final quotient is representable as a floating point number.

    Smith's method:
    Suppose |c| > |d|:
    ac + bd     bc - ad   a + b x (d/c)     b - a x (d/c)
    ------- + i ------- = ------------- + i ------------- should be better behaved.
    cc + dd     cc + dd   c + d x (d/c)     c + d x (d/c)

    else:
    ac + bd     bc - ad   a x (c/d) + b     b x (c/d) - a
    ------- + i ------- = ------------- + i ------------- should be better behaved.
    cc + dd     cc + dd   c x (c/d) + d     c x (c/d) + d

    There is still the case of d/c, c/d underflowing to zero.

    Baudin's enhancement to smith method:
    Suppose d/c == 0:
    a + b x (d/c)     b - a x (d/c)   a + d x (b/c)     b - d x (a/c)
    ------------- + i ------------- = ------------- + i -------------
    c + d x (d/c)     c + d x (d/c)         c                 c

    Suppose c/d == 0:
    a x (c/d) + b     b x (c/d) - a   c x (a/d) + b     c x (b/d) - a
    ------------- + i ------------- = ------------- + i -------------
    c x (c/d) + d     c x (c/d) + d         d                 d

    Things can still overflow or underflow if the numbers are
    close enough to max or min floating-point values.

    1. Suppose max(|a|,|b|) is large:
    a + b x (d/c)     b - a x (d/c)        (a/16) + (b/16) x (d/c)          (b/16) - (a/16) x (d/c)
    ------------- + i ------------- = 16 x ----------------------- + 16 x i -----------------------
    c + d x (d/c)     c + d x (d/c)             c + d x (d/c)                    c + d x (d/c)
    is less likely to overflow if a and/or b are large.

    2. Suppose max(|c|,|d|) is large:
    a + b x (d/c)     b - a x (d/c)    1        a + b x (d/c)         1          b - a x (d/c)
    ------------- + i ------------- = -- x ----------------------- + -- x i -----------------------
    c + d x (d/c)     c + d x (d/c)   16   (c/16) + (d/16) x (d/c)   16     (c/16) + (d/16) x (d/c)
    is less likely to overflow if c and/or d are large.

    3. Suppose max(|a|,|b|) is small:
    a + b x (d/c)     b - a x (d/c)   1   (a x f) + (b x f) x (d/c)   1     (b x f) - (a x f) x (d/c)
    ------------- + i ------------- = - x ------------------------- + - x i -------------------------
    c + d x (d/c)     c + d x (d/c)   f         c + d x (d/c)         f           c + d x (d/c)
    for some large factor f, is less likely to underflow if a and/or b are small.

    4. Suppose max(|c|,|d|) is small:
    a + b x (d/c)     b - a x (d/c)             a + b x (d/c)                     b - a x (d/c)
    ------------- + i ------------- = f x ------------------------- + f x i -------------------------
    c + d x (d/c)     c + d x (d/c)       (c x f) + (d x f) x (d/c)         (c x f) + (d x f) x (d/c)
    is less likely to overflow if c and/or d are large.
  */

  using A_t = impl::inner_type_t<A>;
  using B_t = impl::inner_type_t<B>;
  static_assert(sizeof(A_t) >= sizeof(B_t)); // We aren't losing precision.

  using A_T = to_type_t<A_t>;
  constexpr A_T ov_a = std::numeric_limits<A_T>::max();
  constexpr B_t ov_b = std::numeric_limits<B_t>::max();

  const A_T abs_a = std::fabs(to<A_T>(std::real(a)));
  const A_T abs_b = std::fabs(to<A_T>(std::imag(a)));
  const B_t abs_c = std::fabs(std::real(b));
  const B_t abs_d = std::fabs(std::imag(b));
  const A_T ab = std::max(abs_a, abs_b);
  const B_t cd = std::max(abs_c, abs_d);

  /* Scaling */
  A_T S = 1.0;

  constexpr A_T AO = A_T(16);
  if (ab > ov_a / AO) {
    // scale down a, b
    a /= AO;
    S = S * AO;
  }

  if (cd > ov_b / 16) {
    // scale down c, d
    b /= 16;
    S = S / 16;
  }

  A_t q[2];

  if (abs_c > abs_d) {
    const B_t r = std::imag(b) / std::real(b);
    const B_t t = std::real(b) + std::imag(b) * r;
    q[0] = (std::real(a) + std::imag(a) * r) / t;
    q[1] = (std::imag(a) - std::real(a) * r) / t;
  } else {
    const B_t r = std::real(b) / std::imag(b);
    const B_t t = std::imag(b) + std::real(b) * r;
    q[0] = (std::imag(a) + std::real(a) * r) / t;
    q[1] = (std::imag(a) * r - std::real(a)) / t;
  }

  return A(q[0] * S, q[1] * S);
}

template<typename A,
         typename B>
constexpr A div(A a, B b)
{
  if constexpr (std::is_same_v<A, std::complex<double>> &&
                std::is_same_v<B, std::complex<double>>) {
    constexpr double eps = std::pow(2.0, -std::numeric_limits<float>::digits);
    constexpr double un = std::pow(2.0, std::numeric_limits<float>::min_exponent - 1);

    //static_assert(std::numeric_limits<double>::max() == 0); // 1.7976931348623157e+308
    //static_assert(std::numeric_limits<double>::max()/2 == 0); // 8.9884656743115785e+307
    //static_assert(std::pow(2.0, std::numeric_limits<double>::max_exponent - 1) == 0); // 8.9884656743115795e+307

    const double abs_a = std::fabs(std::real(a));
    const double abs_b = std::fabs(std::imag(a));
    const double abs_c = std::fabs(std::real(b));
    const double abs_d = std::fabs(std::imag(b));
    const double ab = std::max(abs_a, abs_b);
    const double cd = std::max(abs_c, abs_d);

    /* Scaling */
    double S = 1.0;

    if (ab < un / eps * 2) {        /* scale up a, b */
      const double t = 2.0 / (eps * eps);
      a *= t;
      S = S / t;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      const double t = 2.0 / (eps * eps);
      b *= t;
      S = S * t;
    }

    return impl::do_complex_div(a, b) * S;

  } else if constexpr (std::is_same_v<A, std::complex<double>> &&
                       std::is_same_v<B, std::complex<float>>) {
    constexpr double eps = std::pow(2.0, -std::numeric_limits<float>::digits);
    constexpr double un = std::pow(2.0, std::numeric_limits<float>::min_exponent - 1);
    constexpr double eps1 = std::pow(2.0, -std::numeric_limits<double>::digits);
    constexpr double un1 = std::pow(2.0, std::numeric_limits<double>::min_exponent - 1);

    const double abs_a = std::fabs(std::real(a));
    const double abs_b = std::fabs(std::imag(a));
    const double abs_c = std::fabs(static_cast<double>(std::real(b)));
    const double abs_d = std::fabs(static_cast<double>(std::imag(b)));
    const double ab = std::max(abs_a, abs_b);
    const double cd = std::max(abs_c, abs_d);

    /* Scaling */
    double S = 1.0;

    if (ab < un1 / eps1 * 2) {        /* scale up a, b */
      const double t = 2.0 / (eps1 * eps1);
      a *= t;
      S = S / t;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      const double t = 2.0 / (eps * eps);
      b *= t;
      S = S * t;
    }

    return impl::do_complex_div(a, b) * S;

  } else if constexpr (std::is_same_v<A, std::complex<float>> &&
                       std::is_same_v<B, std::complex<float>>) {
    constexpr double eps = std::pow(2.0, -std::numeric_limits<float>::digits);
    constexpr double un = std::pow(2.0, std::numeric_limits<float>::min_exponent - 1);

    const double abs_a = std::fabs(static_cast<double>(std::real(a)));
    const double abs_b = std::fabs(static_cast<double>(std::imag(a)));
    const double abs_c = std::fabs(static_cast<double>(std::real(b)));
    const double abs_d = std::fabs(static_cast<double>(std::imag(b)));
    const double ab = std::max(abs_a, abs_b);
    const double cd = std::max(abs_c, abs_d);

    /* Scaling */
    double S = 1.0;

    if (ab < un / eps * 2) {        /* scale up a, b */
      const double t = 2.0 / (eps * eps);
      a *= t;
      S = S / t;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      const double t = 2.0 / (eps * eps);
      b *= t;
      S = S * t;
    }

    return impl::do_complex_div(a, b) * static_cast<float>(S);

  } else if constexpr (std::is_same_v<A, std::complex<XBLAS_X_t>> &&
                       std::is_same_v<B, std::complex<double>>) {
    constexpr double eps = std::pow(2.0, -std::numeric_limits<double>::digits);        /* double precision */
    constexpr double un = std::pow(2.0, std::numeric_limits<double>::min_exponent - 1);
    constexpr double eps1 = std::pow(2.0, -2*(std::numeric_limits<double>::digits - 1));        /* extra precision */
    constexpr double un1 = std::pow(2.0, std::numeric_limits<double>::min_exponent - 1);

    const double abs_a = std::fabs(to<double>(std::real(a)));
    const double abs_b = std::fabs(to<double>(std::imag(a)));
    const double abs_c = std::fabs(std::real(b));
    const double abs_d = std::fabs(std::imag(b));
    const double ab = std::max(abs_a, abs_b);
    const double cd = std::max(abs_c, abs_d);

    /* Scaling */
    double S = 1.0;

    if (ab < un1 / eps1 * 2) {        /* scale up a, b */
      const double s = 2.0 / (eps1 * eps1);
#ifdef XBLAS_USE_FLOAT128
      a = a * static_cast<inner_type_t<A>>(s);
#else
      a = a * s;
#endif
      S = S / s;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      const double s = 2.0 / (eps * eps);
      b *= s;
      S = S * s;
    }

    return impl::do_complex_div(a, b) * S;

  } else if constexpr (std::is_same_v<A, std::complex<XBLAS_X_t>> &&
                       std::is_same_v<B, std::complex<float>>) {
    constexpr double eps = std::pow(2.0, -std::numeric_limits<float>::digits);        /* single precision */
    constexpr double un = std::pow(2.0, std::numeric_limits<float>::min_exponent - 1);
    constexpr double eps1 = std::pow(2.0, -2*(std::numeric_limits<double>::digits - 1)); // -104       /* extra precision */
    constexpr double un1 = std::pow(2.0, std::numeric_limits<double>::min_exponent - 1);

    const double abs_a = std::fabs(to<double>(std::real(a)));
    const double abs_b = std::fabs(to<double>(std::imag(a)));
    const double abs_c = std::fabs(static_cast<double>(std::real(b)));
    const double abs_d = std::fabs(static_cast<double>(std::imag(b)));
    const double ab = std::max(abs_a, abs_b);
    const double cd = std::max(abs_c, abs_d);

    /* Scaling */
    double S = 1.0;

    if (ab < un1 / eps1 * 2) {        /* scale up a, b */
      const double s = 2.0 / (eps1 * eps1);
#ifdef XBLAS_USE_FLOAT128
      a = a * static_cast<inner_type_t<A>>(s);
#else
      a = a * s;
#endif
      S = S / s;
    }
    if (cd < un / eps * 2) {        /* scale up c, d */
      const double s = 2.0 / (eps * eps);
      b *= s;
      S = S * s;
    }

    return impl::do_complex_div(a, b) * S;

  } else if constexpr (std::is_same_v<inner_type_t<A>, XBLAS_X_t>) {
#ifdef XBLAS_USE_FLOAT128
    return a / static_cast<inner_type_t<A>>(b);
#else
    return a / b;
#endif
  } else if constexpr (is_complex_v<A>) {
    static_assert(!is_complex_v<B>);
    return a / static_cast<inner_type_t<A>>(b);

  } else {
    return a / static_cast<A>(b);
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
struct Imag_h<1> // hermitian, use the imag function appropriately
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

//-----------------
} // namespace impl
//-----------------

//------------------
} // namespace XBLAS
//------------------

#endif // XBLAS_IMPL_HPP
