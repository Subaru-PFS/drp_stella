#ifndef PFS_DRP_STELLA_NORMALIZEDPOLYNOMIAL_H
#define PFS_DRP_STELLA_NORMALIZEDPOLYNOMIAL_H

#include <memory>
#include <numeric>
#include <vector>
#include <algorithm>

#include "ndarray.h"
#include "ndarray/eigen.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/afw/math/FunctionLibrary.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"


namespace pfs {
namespace drp {
namespace stella {
namespace math {


/// 1D ordinary polynomial, with inputs normalized to be in the range [-1,1)
///
/// Subclass of lsst::afw::math::PolynomialFunction1 (ordinary polynomial) with
/// the input normalization copied from lsst::afw::math::Chebyshev1Function1.
template <typename T>
class NormalizedPolynomial1 : public lsst::afw::math::PolynomialFunction1<T> {
  public:
    /// Ctor
    ///
    /// @param order : Polynomial order (identical for x and y)
    /// @param min : Minimum input value
    /// @param max : Maximum input value
    explicit NormalizedPolynomial1(
        unsigned int order,
        double min=-1,
        double max=1
    ) : lsst::afw::math::PolynomialFunction1<T>(order) {
        _initialize(min, max);
    }

    /// Ctor
    ///
    /// @param parameters : Polynomial coefficients
    /// @param min : Minimum input value
    /// @param max : Maximum input value
    explicit NormalizedPolynomial1(
        ndarray::Array<double, 1, 1> const& parameters,
        double min=-1,
        double max=1
    ) : NormalizedPolynomial1(utils::arrayToVector(parameters), min, max)
    {}

    /// Ctor
    ///
    /// @param parameters : Polynomial coefficients
    /// @param min : Minimum input value
    /// @param max : Maximum input value
    explicit NormalizedPolynomial1(
        std::vector<double> const& parameters,
        double min=-1,
        double max=1
    ) : lsst::afw::math::PolynomialFunction1<T>(parameters) {
        _initialize(min, max);
    }

    NormalizedPolynomial1(NormalizedPolynomial1 const&) = default;
    NormalizedPolynomial1(NormalizedPolynomial1&&) = default;
    NormalizedPolynomial1& operator=(NormalizedPolynomial1 const&) = default;
    NormalizedPolynomial1& operator=(NormalizedPolynomial1&&) = default;
    ~NormalizedPolynomial1() noexcept override = default;

    /// Polymorphic clone
    std::shared_ptr<lsst::afw::math::Function1<T>> clone() const override {
        return std::make_shared<NormalizedPolynomial1<T>>(this->getParameters(), getMin(), getMax());
    }

    //@{
    /// Return the bounds of input coordinates used for normalization
    double getMin() const { return _min; }
    double getMax() const { return _max; }
    //@}

    /// Normalize input coordinate
    double normalize(double x) const {
        return (x - _offset)*_scale;
    }

    //@{
    /// Evaluate at coordinates
    T operator()(double x) const noexcept override {
        return lsst::afw::math::PolynomialFunction1<T>::operator()(normalize(x));
    }
    ndarray::Array<T, 1, 1> operator()(ndarray::Array<double, 1, 1> const& x) const noexcept {
        ndarray::Array<T, 1, 1> result = ndarray::allocate(x.size());
        std::transform(x.begin(), x.end(), result.begin(),
                       [this](double value) { return operator()(value); });
        return result;
    }
    //@}

    /// Calculate the derivatives of the function w.r.t. the parameters
    ///
    /// Useful for constructing the design matrix when fitting.
    ///
    /// Strangely, this isn't implemented in the base class for 1D functions,
    /// so we implement it here.
    std::vector<double> getDFuncDParameters(double x) const {
        double const xNorm = (x - _offset)*_scale;
        int const num = this->getNParameters();
        std::vector<double> result;
        assert(num >= 1);
        result.push_back(1.0);
        for (int ii = 0; ii < num - 1; ++ii) {
            result.push_back(result[ii]*xNorm);
        }
        return result;
    }

    /// Calculate the design matrix for a least-squares fit
    ///
    /// This is the result from getDFuncDParameters, for each point in x.
    ndarray::Array<double, 2, 2> calculateDesignMatrix(ndarray::Array<double, 1, 1> const& x) const {
        std::size_t const numPoints = x.size();
        std::size_t const numParams = this->getNParameters();
        ndarray::Array<double, 2, 2> design = ndarray::allocate(numPoints, numParams);
        for (std::size_t ii = 0; ii < numPoints; ++ii) {
            design[ii].deep() = utils::vectorToArray(getDFuncDParameters(x[ii]));
        }
        return design;
    }

    /// Not persistable because we haven't written the persistence code
    bool isPersistable() const noexcept override { return false; }

  private:

    /// Initialize values used for normalization
    void _initialize(double min, double max) {
        _min = min;
        _max = max;
        _offset = 0.5*(min + max);
        _scale = 2.0/(max - min);
    }

    double _min, _max;  ///< Range of input coordinates
    double _offset;  ///< Offset for normalization
    double _scale;  ///< Scale for normalization
};


/// 2D ordinary polynomial, with inputs normalized to be in the range [-1,1)
///
/// Subclass of lsst::afw::math::PolynomialFunction2 (ordinary polynomial) with
/// the input normalization copied from lsst::afw::math::Chebyshev1Function2.
template <typename T>
class NormalizedPolynomial2 : public lsst::afw::math::PolynomialFunction2<T> {
  public:
    /// Ctor
    ///
    /// @param order : Polynomial order (identical for x and y)
    /// @param range : Bounds of input coordinates, for normalization
    explicit NormalizedPolynomial2(
        unsigned int order,
        lsst::geom::Box2D const& range=lsst::geom::Box2D(
            lsst::geom::Point2D(-1.0, -1.0),
            lsst::geom::Point2D(1.0, 1.0))
    ) : lsst::afw::math::PolynomialFunction2<T>(order) {
        _initialize(range);
    }

    /// Ctor
    ///
    /// @param parameters : Polynomial coefficients
    /// @param range : Bounds of input coordinates, for normalization
    explicit NormalizedPolynomial2(
        ndarray::Array<double, 1, 1> const& parameters,
        lsst::geom::Box2D const& range=lsst::geom::Box2D(
            lsst::geom::Point2D(-1.0, -1.0),
            lsst::geom::Point2D(1.0, 1.0))
    ) : NormalizedPolynomial2(utils::arrayToVector(parameters), range)
    {}

    /// Ctor
    ///
    /// @param parameters : Polynomial coefficients
    /// @param range : Bounds of input coordinates, for normalization
    explicit NormalizedPolynomial2(
        std::vector<double> const& parameters,
        lsst::geom::Box2D const& range=lsst::geom::Box2D(
            lsst::geom::Point2D(-1.0, -1.0),
            lsst::geom::Point2D(1.0, 1.0))
    ) : lsst::afw::math::PolynomialFunction2<T>(parameters) {
        _initialize(range);
    }

    NormalizedPolynomial2(NormalizedPolynomial2 const&) = default;
    NormalizedPolynomial2(NormalizedPolynomial2&&) = default;
    NormalizedPolynomial2& operator=(NormalizedPolynomial2 const&) = default;
    NormalizedPolynomial2& operator=(NormalizedPolynomial2&&) = default;
    ~NormalizedPolynomial2() noexcept override = default;

    /// Polymorphic clone
    std::shared_ptr<lsst::afw::math::Function2<T>> clone() const override {
        return std::make_shared<NormalizedPolynomial2<T>>(this->getParameters(), getXYRange());
    }

    /// Return the bounds of input coordinates used for normalization
    lsst::geom::Box2D getXYRange() const { return _range; }

    /// Normalize input coordinates
    lsst::geom::Point2D normalize(lsst::geom::Point2D const& xy) const {
        return lsst::geom::Point2D(
            (xy.getX() - _xOffset)*_xScale,
            (xy.getY() - _yOffset)*_yScale
        );
    }

    //@{
    /// Evaluate at coordinates
    T operator()(double x, double y) const noexcept override {
        auto const norm = normalize(lsst::geom::Point2D(x, y));
        return lsst::afw::math::PolynomialFunction2<T>::operator()(norm.getX(), norm.getY());
    }
    ndarray::Array<double, 1, 1> operator()(
        ndarray::Array<double, 1, 1> const& x,
        ndarray::Array<double, 1, 1> const& y
    ) const noexcept {
        utils::checkSize(x.size(), y.size(), "x vs y");
        std::size_t const num = x.size();
        ndarray::Array<double, 1, 1> result = ndarray::allocate(num);
        for (std::size_t ii = 0; ii < num; ++ii) {
            result[ii] = operator()(x[ii], y[ii]);
        }
        return result;
    }
    //@}

    /// Calculate the derivatives of the function w.r.t. the parameters
    ///
    /// Useful for constructing the design matrix when fitting.
    std::vector<double> getDFuncDParameters(double x, double y) const override {
        return lsst::afw::math::PolynomialFunction2<T>::getDFuncDParameters(
            (x - _xOffset)*_xScale,
            (y - _yOffset)*_yScale
        );
    }

    /// Calculate the design matrix for a least-squares fit
    ///
    /// This is the result from getDFuncDParameters, for each point in x,y.
    ndarray::Array<double, 2, 2> calculateDesignMatrix(
        ndarray::Array<double, 1, 1> const& x,
        ndarray::Array<double, 1, 1> const& y
    ) const {
        utils::checkSize(x.size(), y.size(), "x vs y");
        std::size_t const numPoints = x.size();
        std::size_t const numParams = this->getNParameters();
        ndarray::Array<double, 2, 2> design = ndarray::allocate(numPoints, numParams);
        for (std::size_t ii = 0; ii < numPoints; ++ii) {
            design[ii].deep() = utils::vectorToArray(getDFuncDParameters(x[ii], y[ii]));
        }
        return design;
    }

    /// Not persistable because we haven't written the persistence code
    bool isPersistable() const noexcept override { return false; }

  private:

    /// Initialize values used for normalization
    void _initialize(lsst::geom::Box2D const& range) {
        _range = range;
        _xOffset = 0.5*(range.getMinX() + range.getMaxX());
        _yOffset = 0.5*(range.getMinY() + range.getMaxY());
        _xScale = 2.0/(range.getMaxX() - range.getMinX());
        _yScale = 2.0/(range.getMaxY() - range.getMinY());
    }

    lsst::geom::Box2D _range;  ///< Bounds of input coordinates
    double _xOffset, _yOffset;  ///< Offset for normalization
    double _xScale, _yScale;  ///< Scale for normalization
};


/// Base class for N-dimensional functions with a fixed number of parameters.
///
/// lsst::afw::math does not support N-dimensional functions, so we define our
/// own. We prefer using ndarray::Array over std::vector wherever possible,
/// because experience has taught us that these are more useful.
template <typename T>
class FunctionN : public lsst::afw::math::Function<T> {
    using Function = lsst::afw::math::Function<T>;
  public:
    using Array1T = ndarray::Array<T, 1, 1>;
    using Array1D = ndarray::Array<double, 1, 1>;

    /// Ctor
    explicit FunctionN(unsigned int dims, unsigned int nParameters)
      : Function(nParameters), _dims(dims) {}
    explicit FunctionN(unsigned int dims, std::vector<double> const& parameters)
      : Function(parameters), _dims(dims) {}
    explicit FunctionN(unsigned int dims, Array1D const& parameters)
      : Function(utils::arrayToVector(parameters)), _dims(dims) {}

    FunctionN(FunctionN const&) = default;
    FunctionN(FunctionN&&) = default;
    FunctionN& operator=(FunctionN const&) = default;
    FunctionN& operator=(FunctionN&&) = default;
    ~FunctionN() noexcept override = default;

    /// Polymorphic clone
    virtual std::shared_ptr<FunctionN<T>> clone() const = 0;

    /// Return the number of dimensions
    unsigned int getDimensions() const noexcept { return this->_dims; }

    /// Return the parameters
    ///
    /// This overrides the definition in lsst::afw::math::Function, which
    /// returns a std::vector<double>. The array we return uses the memory
    /// from the parameters, so its lifetime is tied to the lifetime of the
    /// FunctionN object.
    Array1D getParameters() const noexcept {
        return ndarray::copy(utils::vectorToArray(Function::getParameters()));
    }

    /// Set the parameters
    ///
    /// This overrides the definition in lsst::afw::math::Function, which
    /// takes a std::vector<double>.
    void setParameters(Array1D const& parameters) {
        utils::checkSize(parameters.size(), std::size_t(this->getNParameters()), "parameters");
        Function::setParameters(utils::arrayToVector(parameters));
    }

    /// Evaluate the function at the given coordinates
    virtual T operator()(Array1D const& position) const = 0;

    /// Calculate the derivatives of the function w.r.t. the parameters
    virtual Array1T getDFuncDParameters(Array1D const& position) const = 0;

protected:

    /// Default constructor: intended only for serialization
    explicit FunctionN() : Function() {}

    unsigned int _dims;  ///< Number of dimensions
};


/// Factorial function
inline std::size_t factorial(std::size_t n) {
    assert(n <= 20);  // 20! >= 2**64
    static_assert(sizeof(size_t) >= 8);  // Ensure size_t is at least 64 bits
    if (n == 0 || n == 1) {
        return 1;
    }
    return n * factorial(n - 1);
}


/// Base class for N-dimensional polynomials
template <typename T>
class BasePolynomialFunctionN : public FunctionN<T> {
  public:
    using Array1D = typename FunctionN<T>::Array1D;

    /// Ctor
    explicit BasePolynomialFunctionN(unsigned int dims, unsigned int order)
      : FunctionN<T>(dims, nParametersFromOrder(dims, order)), _order(order) {}

    explicit BasePolynomialFunctionN(unsigned int dims, Array1D const& parameters)
      : FunctionN<T>(dims, parameters), _order(orderFromNParameters(dims, parameters.size())) {}

    BasePolynomialFunctionN(BasePolynomialFunctionN const&) = default;
    BasePolynomialFunctionN(BasePolynomialFunctionN&&) = default;
    BasePolynomialFunctionN& operator=(BasePolynomialFunctionN const&) = default;
    BasePolynomialFunctionN& operator=(BasePolynomialFunctionN&&) = default;
    ~BasePolynomialFunctionN() noexcept override = default;

    /// Return the polynomial order
    unsigned int getOrder() const noexcept { return _order; }

    bool isLinearCombination() const noexcept override { return true; }

    /// Return the number of parameters from the polynomial order
    ///
    /// @throws lsst::pex::exceptions::InvalidParameterError if order < 0 or dims < 1
    static int nParametersFromOrder(int dims, int order) {
        if (dims < 1) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "dims must be >= 1");
        }
        if (order < 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "order must be >= 0");
        }
        /// The number of parameters is given by the combination
        /// C(d + n, n) = (d + n)! / n! / d!
        /// where d is the number of dimensions and n is the polynomial order.
        return factorial(dims + order) / (factorial(order) * factorial(dims));
    }

    /// Compute polynomial order from the number of dimensions and parameters
    ///
    /// @throws lsst::pex::exceptions::InvalidParameterError if nParameters is invalid
    static int orderFromNParameters(unsigned int dims, std::size_t nParameters) {
        if (dims < 1) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "dims must be >= 1");
        }
        if (nParameters < 1) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "nParameters must be >= 1");
        }
        int order = 0;
        std::size_t nParametersFromOrder = 0;
        while (
            nParameters > (nParametersFromOrder = BasePolynomialFunctionN::nParametersFromOrder(dims, order))
        ) {
            ++order;
        }
        if (nParameters != nParametersFromOrder) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterError,
                "Cannot determine order from dims, nParameters"
            );
        }
        return order;
    }

  protected:
    int _order;  ///< order of polynomial

    /// Default constructor: intended only for serialization
    explicit BasePolynomialFunctionN() : FunctionN<T>(1), _order(0) {}
};


namespace detail {

inline bool compareExponents(std::vector<unsigned int> const& a, std::vector<unsigned int> const& b) {
    unsigned int sumA = std::accumulate(a.begin(), a.end(), 0u);
    unsigned int sumB = std::accumulate(b.begin(), b.end(), 0u);
    if (sumA != sumB) {
        return sumA > sumB;  // Sort by descending sum
    }
    // If sums are equal, sort in reverse lexicographical order
    return std::lexicographical_compare(
        a.begin(), a.end(), b.begin(), b.end(),
        [](unsigned int x, unsigned int y) { return x > y; }
    );
}

}  // namespace detail


/// N-dimensional ordinary polynomial
template <typename T>
class PolynomialFunctionN : public BasePolynomialFunctionN<T> {
  public:
    using Array1D = typename BasePolynomialFunctionN<T>::Array1D;
    using Array1T = ndarray::Array<T, 1, 1>;
    using Exponents = std::vector<std::vector<unsigned int>>;

    /// Ctor
    explicit PolynomialFunctionN(unsigned int dims, unsigned int order)
      : BasePolynomialFunctionN<T>(dims, order) {
        _initialize();
      }

    explicit PolynomialFunctionN(unsigned int dims, Array1D const& parameters)
      : BasePolynomialFunctionN<T>(dims, parameters) {
        _initialize();
      }

    PolynomialFunctionN(PolynomialFunctionN const&) = default;
    PolynomialFunctionN(PolynomialFunctionN&&) = default;
    PolynomialFunctionN& operator=(PolynomialFunctionN const&) = default;
    PolynomialFunctionN& operator=(PolynomialFunctionN&&) = default;
    ~PolynomialFunctionN() noexcept override = default;

    /// Polymorphic clone
    std::shared_ptr<FunctionN<T>> clone() const override {
        return std::make_shared<PolynomialFunctionN<T>>(this->getDimensions(), this->getParameters());
    }

    /// Evaluate the polynomial at the given coordinates
    T operator()(Array1D const& position) const override {
        utils::checkSize(position.size(), std::size_t(this->getDimensions()), "position vs dims");

        Array1T const& terms = _calculateTerms(position);
        T result = 0;
        for (std::size_t ii = 0; ii < this->getNParameters(); ++ii) {
            result += this->getParameter(ii) * terms[ii];
        }
        return result;
    }

    /// Calculate the derivatives of the function w.r.t. the parameters
    Array1T getDFuncDParameters(Array1D const& position) const override {
        utils::checkSize(position.size(), std::size_t(this->getDimensions()), "position vs dims");
        return ndarray::copy(_calculateTerms(position));
    }

    /// Get the exponents for each polynomial term
    ///
    /// The 'exponents' are the powers of each dimension for each parameter.
    ///
    /// They are sorted:
    /// * In descending order of the sum of the exponents
    /// * In reverse lexicographical order for the same sum
    static Exponents getExponents(
        unsigned int dims,
        unsigned int order
    ) {
        auto exponents = _getExponents(dims, order);
        std::sort(exponents.begin(), exponents.end(), detail::compareExponents);
        return exponents;
    }

    /// Return polynomial parameters with an additional dimension
    Array1D addDimension() const {
        unsigned int const dims = this->getDimensions();
        unsigned int const newDims = dims + 1;
        Array1D parameters = ndarray::allocate(this->nParametersFromOrder(newDims, this->getOrder()));
        parameters.deep() = 0.0;
        auto exponents = getExponents(newDims, this->getOrder());
        for (std::size_t ii = 0; ii < this->getNParameters(); ++ii) {
            double const param = this->getParameter(ii);
            if (param == 0.0) {
                continue;  // We've already set this parameter to zero
            }
            std::vector<unsigned int> exp{_exponents[ii].begin(), _exponents[ii].end()};
            exp.push_back(0);
            auto iter = std::find(exponents.begin(), exponents.end(), exp);
            assert(iter != exponents.end());
            assert(*iter == exp);
            std::ptrdiff_t const index = std::distance(exponents.begin(), iter);
            assert(index >= 0 && index < std::ptrdiff_t(parameters.size()));
            parameters[index] = param;
        }
        return parameters;
    }

    // Not persistable because we haven't written the persistence code
    bool isPersistable() const noexcept override { return false; }

  private:
    /// Initialize the exponents array used for evaluation
    void _initialize() {
        _powers = ndarray::allocate(this->getDimensions(), this->getOrder() + 1);
        _exponents = std::move(getExponents(this->getDimensions(), this->getOrder()));
        _terms = ndarray::allocate(this->getNParameters());
    }

    /// Get the exponents for the polynomial terms
    ///
    /// The 'exponents' are the powers of each dimension for each parameter.
    static Exponents _getExponents(
        unsigned int dims,
        unsigned int order,
        unsigned int fullDims = 0
    ) {
        if (fullDims == 0) {
            fullDims = dims;
        }

        Exponents exponents;
        exponents.reserve(BasePolynomialFunctionN<T>::nParametersFromOrder(dims, order));
        if (dims == 1) {
            for (unsigned int ii = 0; ii <= order; ++ii) {
                std::vector<unsigned int> sub;
                sub.reserve(fullDims);
                sub.push_back(ii);
                exponents.emplace_back(std::move(sub));
            }
            return exponents;
        }
        for (unsigned int ii = 0; ii <= order; ++ii) {
            Exponents subExponents = _getExponents(dims - 1, order - ii, fullDims);
            for (auto& sub : subExponents) {
                sub.push_back(ii);
                exponents.push_back(sub);
            }
        }
        return exponents;
    }

    /// Calculate the polynomial terms for the given position
    Array1T const& _calculateTerms(Array1D const& position) const {
        // Calculate the polynomial terms
        for (unsigned int ii = 0; ii < this->getDimensions(); ++ii) {
            double xx = 1.0;
            _powers[ii][0] = 1.0;
            for (unsigned int jj = 1; jj < this->getOrder() + 1; ++jj) {
                xx *= position[ii];
                _powers[ii][jj] = xx;
            }
        }

        // Multiply the powers for each parameter
        for (std::size_t ii = 0; ii < this->getNParameters(); ++ii) {
            std::vector<unsigned int> const& exponents = _exponents[ii];
            assert(exponents.size() == this->getDimensions());
            double term = 1.0;
            for (unsigned int jj = 0; jj < this->getDimensions(); ++jj) {
                assert(exponents[jj] < _powers[jj].size());
                term *= _powers[jj][exponents[jj]];
            }
            _terms[ii] = term;
        }
        return _terms;
    }

    Exponents _exponents;  ///< Exponents for each parameter

    // Workspace for polynomial calculation; saves re-allocating memory, but makes the class thread-unsafe
    mutable ndarray::Array<double, 2, 2> _powers;  ///< Powers for each dimension
    mutable Array1T _terms;  ///< Terms for each parameter
};


/// N-dimensional ordinary polynomial, with inputs normalized to be in the range [-1,1)
template <typename T>
class NormalizedPolynomialN : public PolynomialFunctionN<T> {
  public:
    using Array1D = typename PolynomialFunctionN<T>::Array1D;
    using Array1T = typename PolynomialFunctionN<T>::Array1T;

    /// Ctor
    explicit NormalizedPolynomialN(
        unsigned int order, Array1D const& min, Array1D const& max, bool newNorm=true
    ) : PolynomialFunctionN<T>(min.size(), order) {
        _initialize(min, max, newNorm);
    }
    explicit NormalizedPolynomialN(
        Array1D const& parameters, Array1D const& min, Array1D const& max, bool newNorm=false
    ) : PolynomialFunctionN<T>(min.size(), parameters) {
        _initialize(min, max, newNorm);
    }

    NormalizedPolynomialN(NormalizedPolynomialN const&) = default;
    NormalizedPolynomialN(NormalizedPolynomialN&&) = default;
    NormalizedPolynomialN& operator=(NormalizedPolynomialN const&) = default;
    NormalizedPolynomialN& operator=(NormalizedPolynomialN&&) = default;
    ~NormalizedPolynomialN() noexcept override = default;

    /// Polymorphic clone
    std::shared_ptr<FunctionN<T>> clone() const override {
        return std::make_shared<NormalizedPolynomialN<T>>(this->getParameters(), getMin(), getMax());
    }

    /// Return whether we're using the new normalization scheme
    bool getNewNorm() const noexcept { return _newNorm; }

    /// Return the bounds of input coordinates used for normalization
    Array1D getMin() const { return _min; }
    Array1D getMax() const { return _max; }

    //@{
    /// Normalize input coordinates
    Array1D normalize(Array1D & normalized, Array1D const& position) const {
        utils::checkSize(position.size(), std::size_t(this->getDimensions()), "position vs dims");
        utils::checkSize(normalized.size(), position.size(), "normalized vs position");
        for (std::size_t ii = 0; ii < position.size(); ++ii) {
            normalized[ii] = (position[ii] - _offset[ii]) * _scale[ii];
        }
        return normalized;
    }
    Array1D normalize(Array1D const& position) const {
        Array1D normalized = ndarray::allocate(position.size());
        return normalize(normalized, position);
    }
    //@}

    /// Evaluate at coordinates
    T operator()(Array1D const& position) const override {
        return PolynomialFunctionN<T>::operator()(normalize(_normalized, position));
    }

    /// Calculate the derivatives of the function w.r.t. the parameters
    Array1T getDFuncDParameters(Array1D const& position) const override {
        return PolynomialFunctionN<T>::getDFuncDParameters(normalize(_normalized, position));
    }

    // Not persistable because we haven't written the persistence code
    bool isPersistable() const noexcept override { return false; }

  protected:
    /// Initialize values used for normalization
    void _initialize(Array1D const& min, Array1D const& max, bool newNorm) {
        if (min.size() != this->getDimensions() || max.size() != this->getDimensions()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterError,
                "min and max must have the same size as dims"
            );
        }
        _newNorm = newNorm;
        _min = ndarray::copy(min);
        _max = ndarray::copy(max);

        if (newNorm) {
            // New offset/scale (matches LSST): normalizes to [-1, 1]
            _offset = ndarray::allocate(this->getDimensions());
            ndarray::asEigenArray(_offset) = 0.5 * (ndarray::asEigenArray(min) + ndarray::asEigenArray(max));

            _scale = ndarray::allocate(this->getDimensions());
            ndarray::asEigenArray(_scale) = 2.0 / (ndarray::asEigenArray(max) - ndarray::asEigenArray(min));
        } else {
            // Old offset/scale: normalizes to [0, 1]
            _offset = ndarray::copy(min);
            _scale = ndarray::allocate(this->getDimensions());
            ndarray::asEigenArray(_scale) = 1.0 / (ndarray::asEigenArray(max) - ndarray::asEigenArray(min));
        }

        _normalized = ndarray::allocate(this->getDimensions());
    }

    bool _newNorm;  ///< Use new normalization (-1..1) scheme? Otherwise, use old (0..1) scheme.
    Array1D _min;  ///< Minimum input values for normalization
    Array1D _max;  ///< Maximum input values for normalization
    Array1D _offset;  ///< Offset for normalization
    Array1D _scale;  ///< Scale for normalization

    mutable Array1D _normalized;  ///< Normalized input coordinates; saves re-allocating memory
};

}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
