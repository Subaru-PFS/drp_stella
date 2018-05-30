#include <Eigen/Eigen>

#include "ndarray/eigen.h"

#include "lsst/pex/exceptions.h"

#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

template <typename T>
std::vector<std::size_t> getIndicesInValueRange(
    ndarray::Array<T const, 1, 1> const& array,
    T lowRange,
    T highRange
) {
    std::vector<std::size_t> indices;
    indices.reserve(array.getNumElements());
    std::size_t pos = 0;
    for (auto it = array.begin(); it != array.end(); ++it, ++pos) {
        if ((lowRange <= *it) && (*it < highRange)) {
            indices.push_back(pos);
        }
    }
    return indices;
}


template <typename T>
std::vector<lsst::afw::geom::Point2I> getIndicesInValueRange(
    ndarray::Array<T, 2, 1> const& array,
    T lowRange,
    T highRange
) {
    std::vector<lsst::afw::geom::Point2I> indices;
    indices.reserve(array.getNumElements());
    int const height = array.getShape()[0], width = array.getShape()[1];
    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < width; ++xx) {
            T const value = array[yy][xx];
            if ((lowRange <= value) && (value < highRange)) {
                indices.emplace_back(xx, yy);
            }
        }
    }
    return indices;
}


template <typename T>
ndarray::Array<T, 1, 1> getSubArray(
    ndarray::Array<T, 1, 1> const& array,
    std::vector<std::size_t> const& indices
) {
    ndarray::Array<T, 1, 1> out = ndarray::allocate(indices.size());
    std::transform(indices.begin(), indices.end(), out.begin(),
                   [array](std::size_t ii) { return array[ii]; });
    return out;
}


template <typename T>
ndarray::Array<T, 1, 1> getSubArray(
    ndarray::Array<T, 2, 1> const& array,
    std::vector<lsst::afw::geom::Point2I> const& indices
) {
    ndarray::Array<T, 1, 1> out = ndarray::allocate(indices.size());
    std::transform(indices.begin(), indices.end(), out.begin(),
                   [array](lsst::afw::geom::Point2I const& pp) {
                       return array[pp.getY()][pp.getX()];
                   });
    return out;
}


template<typename T>
ndarray::Array<T, 1, 1> moment(ndarray::Array<T, 1, 1> const& array, int maxMoment)
{
    ndarray::Array<T, 1, 1> out = ndarray::allocate(maxMoment);
    out.deep() = 0.;
    assert(maxMoment >= 1 && maxMoment <= 4);
#if 0
    if ((maxMoment < 1) && (array.getShape()[0] < 2)) {
      cout << "Moment: ERROR: array must contain 2 OR more elements." << endl;
      return out;
    }
#else
    assert(array.getShape()[0] >= 2);
#endif

    std::size_t const num = array.getShape()[0];
    T mean = array.asEigen().mean();
    out[0] = mean;
    if (maxMoment == 1) {
        return out;
    }

    // Variance
    ndarray::Array<T, 1, 1> resid = ndarray::copy(array - mean);
    Eigen::Array<T, Eigen::Dynamic, 1> eigenResid = resid.asEigen();
    T var = (eigenResid.pow(2).sum() - std::pow(ndarray::sum(resid), 2)/T(num))/(T(num) - 1.);
    out[1] = var;
    if (maxMoment <= 2) {
        return out;
    }

    T std = std::sqrt(var);
    if (std == 0.) {
        return out;
    }

    // Skew
    out[2] = eigenResid.pow(3).sum()/(num*std::pow(std, 3));

    if (maxMoment <= 3) {
        return out;
    }

    // Kurtosis
    out[3] = eigenResid.pow(4).sum()/(num*pow(std, 4)) - 3.;;
    return out;
}


namespace {

// Functor for comparing vector values by their indices
template <typename T>
struct IndicesComparator {
    std::vector<T> const& data;
    IndicesComparator(std::vector<T> const& data_) : data(data_) {}
    bool operator()(std::size_t lhs, std::size_t rhs) const {
        return data[lhs] < data[rhs];
    }
};

} // anonymous namespace


template<typename T>
std::vector<std::size_t> sortIndices(std::vector<T> const& data) {
    std::size_t const num = data.size();
    std::vector<std::size_t> indices;
    indices.reserve(num);
    std::size_t index = 0;
    std::generate_n(indices.begin(), num, [&index]() { return index++; });
    std::sort(indices.begin(), indices.end(), IndicesComparator<T>(data));
    return indices;
}


template <typename T>
std::ptrdiff_t firstIndexWithValueGEFrom(
    ndarray::Array<T, 1, 1> const& array,
    T minValue,
    std::size_t fromIndex
) {
    assert(array.getNumElements() > 0 && fromIndex < array.getNumElements());
    auto const iter = std::find_if(array.begin() + fromIndex, array.end(),
                                   [minValue](T value) { return value >= minValue; });
    return iter == array.end() ? -1 : iter - array.begin();
}

template <typename T>
std::ptrdiff_t lastIndexWithZeroValueBefore(
    ndarray::Array<T, 1, 1> const& array,
    std::ptrdiff_t startPos
) {
    if (startPos < 0 || std::size_t(startPos) >= array.getNumElements()) {
        return -1;
    }
#if 0
    auto const end = std::make_reverse_iterator(array.begin())
    auto const iter = std::find_if(std::make_reverse_iterator(array.begin() + startPos), end,
                                   [](T value) { return std::fabs(value) < 0.00000000000000001; });
    return iter == end ? -1 :
#else
    std::ptrdiff_t index = startPos;
    for (auto i = array.begin() + startPos; i != array.begin(); --i, --index) {
        if (std::fabs(*i) < 0.00000000000000001) {
            return index;
        }
    }
    return -1;
#endif
}

template <typename T>
std::ptrdiff_t firstIndexWithZeroValueFrom(
    ndarray::Array<T, 1, 1> const& array,
    std::ptrdiff_t startPos
) {
    if (startPos < 0 || std::size_t(startPos) >= array.getNumElements()) {
        return -1;
    }
    auto const iter = std::find_if(array.begin() + startPos, array.end(),
                                   [](T value) { return std::fabs(value) < 0.00000000000000001; });
    return iter == array.end() ? -1 : iter - array.begin();
}

// Explicit instantiations
template std::vector<std::size_t> getIndicesInValueRange(
    ndarray::Array<float const, 1, 1> const&,
    float,
    float
);

template std::vector<lsst::afw::geom::Point2I> getIndicesInValueRange(
    ndarray::Array<float, 2, 1> const&,
    float,
    float
);

template ndarray::Array<float, 1, 1> getSubArray(
    ndarray::Array<float, 1, 1> const&,
    std::vector<std::size_t> const&
);

template ndarray::Array<float, 1, 1> getSubArray(
    ndarray::Array<float, 2, 1 > const&,
    std::vector<lsst::afw::geom::Point2I> const&
);

template ndarray::Array<float, 1, 1> moment(
    ndarray::Array<float, 1, 1> const&,
    int maxMoment
);

template std::vector<std::size_t> sortIndices(std::vector<float> const&);

template std::ptrdiff_t firstIndexWithValueGEFrom(
    ndarray::Array<std::size_t, 1, 1> const& array,
    std::size_t minValue,
    std::size_t fromIndex
);

template std::ptrdiff_t lastIndexWithZeroValueBefore(
    ndarray::Array<std::size_t, 1, 1> const& array,
    std::ptrdiff_t startPos
);

template std::ptrdiff_t firstIndexWithZeroValueFrom(
    ndarray::Array<std::size_t, 1, 1> const&,
    std::ptrdiff_t startPos
);


}}}} // namespace pfs::drp::stella::math

template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& obj) {
    for (auto const& ii : obj) {
        os << ii << " ";
    }
    os << endl;
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, lsst::afw::geom::Point<T, 2> const& point) {
    os << "(" << point.getX() << "," << point.getY() << ")";
    return os;
}

template std::ostream& operator<<(std::ostream&, std::vector<float> const&);
template std::ostream& operator<<(std::ostream&, std::vector<lsst::afw::geom::Point2I> const&);
template std::ostream& operator<<(std::ostream&, lsst::afw::geom::Point2I const&);
template std::ostream& operator<<(std::ostream&, lsst::afw::geom::Point2D const&);

#if 0


template<typename T>
size_t minIndex( ndarray::Array<T, 1, 1> const& in )
{
  T min = in[0];
  size_t minIndex = 0;
  size_t ind = 0;
  for ( auto it = in.begin(); it != in.end(); ++it, ++ind ) {
    if ( *it < min ) {
      min = *it;
      minIndex = ind;
    }
  }
  return minIndex;
}



template<typename T>
std::vector< size_t > getIndices( std::vector< T > const& vec_In )
{
  std::vector< size_t > vecOut(0);
  for ( size_t pos = 0; pos < vec_In.size(); ++pos ) {
    if ( int( vec_In[ pos ]) == 1 )
      vecOut.push_back(pos);
  }
  return vecOut;
}

template<typename T>
ndarray::Array< size_t, 1, 1 > getIndices( ndarray::Array< T, 1, 1 > const& arr_In )
{
  ndarray::Array< size_t, 1, 1 > arrOut = ndarray::allocate(std::accumulate(arr_In.begin(), arr_In.end(), 0));
  auto itOut = arrOut.begin();
  int pos = 0;
  for ( auto itIn = arr_In.begin(); itIn != arr_In.end(); ++itIn, ++pos ) {
    if ( int( *itIn) == 1 ) {
      *itOut = pos;
      ++itOut;
    }
  }
  return arrOut;
}

template<typename T>
ndarray::Array< size_t, 2, 1 > getIndices( ndarray::Array< T, 2, 1 > const& arr_In )
{
  T nInd = 0;
  for ( auto itRow = arr_In.begin(); itRow != arr_In.end(); ++itRow ) {
    nInd += std::accumulate(itRow->begin(), itRow->end(), 0);
  }
  ndarray::Array< size_t, 2, 1 > arrOut = ndarray::allocate(int(nInd), 2);
  auto itRowOut = arrOut.begin();
  int row = 0;
  for ( auto itRowIn = arr_In.begin(); itRowIn != arr_In.end(); ++itRowIn, ++row ) {
    int col = 0;
    for ( auto itColIn = itRowIn->begin(); itColIn != itRowIn->end(); ++itColIn, ++col ) {
      if ( int( *itColIn) == 1 ) {
        auto itColOut = itRowOut->begin();
        *itColOut = row;
        ++itColOut;
        *itColOut = col;
        ++itRowOut;
      }
    }
  }
  return arrOut;
}


template<typename T>
ndarray::Array<T, 1, 1> getSubArray( ndarray::Array<T, 2, 1> const& arr_In,
                                     std::vector< std::pair<size_t, size_t> > const& indices_In )
{
  ndarray::Array<T, 1, 1> arr_Out = ndarray::allocate(indices_In.size());
  for ( int iRow = 0; iRow < indices_In.size(); ++iRow ) {
    arr_Out[ iRow ] = arr_In[ ndarray::makeVector(int( indices_In[ iRow ].first), int( indices_In[ iRow ].second)) ];
#ifdef __DEBUG_GETSUBARRAY__
    cout << "getSubArray: arr_Out[" << iRow << "] = " << arr_Out[ iRow ] << endl;
#endif
  }
  return arr_Out;
}

template< typename T >
void insertSorted( std::vector< dataXY< T > > & dataXYVec_In,
                   dataXY< T > & toInsert_In )
{
  if ( dataXYVec_In.size() == 0 ) {
    dataXYVec_In.push_back(toInsert_In);
    return;
  }
  for ( auto it = dataXYVec_In.begin(); it != dataXYVec_In.end(); ++it ) {
    if ( it->x > toInsert_In.x ) {
      dataXYVec_In.insert(it, toInsert_In);
      return;
    }
  }
  dataXYVec_In.push_back(toInsert_In);
  return;
}



template<typename T>
bool resize( ndarray::Array<T, 1, 1> & arr, size_t const newSize )
{
#ifdef __DEBUG_RESIZE__
  cout << "math::resize: starting to resize array to new size " << newSize << endl;
#endif

  /// Create temporary array of final size and copy data from existing array
  ndarray::Array<T, 1, 1> arrOut = ndarray::allocate(newSize);
#ifdef __DEBUG_RESIZE__
  cout << "math::resize: space allocated for arrOut of size " << arrOut.getShape()[0] << endl;
#endif
  arrOut.deep() = 0;
  for ( auto itArrIn = arr.begin(), itArrOut = arrOut.begin(); (itArrIn != arr.end()) && (itArrOut != arrOut.end()); ++itArrIn, ++itArrOut ) {
    *itArrOut = *itArrIn;
  }
#ifdef __DEBUG_RESIZE__
  cout << "math::resize: arr of size " << arr.getShape()[0] << " copied to arrOut of size " << arrOut.getShape()[0] << endl;
#endif

  /// resize input array and copy data back
  arr = ndarray::allocate(newSize);
#ifdef __DEBUG_RESIZE__
  cout << "math::resize: space allocated for arr of size " << arr.getShape()[0] << endl;
#endif
  for ( auto itArrIn = arr.begin(), itArrOut = arrOut.begin(); itArrIn != arr.end(); ++itArrIn, ++itArrOut ) {
    *itArrIn = *itArrOut;
  }

#ifdef __DEBUG_RESIZE__
  cout << "math::resize: arr of size " << arr.getShape()[0] << " copied to arrOut of size " << arrOut.getShape()[0] << endl;
  cout << "math::resize: new arr = " << arr << endl;
#endif
  return true;
}

template<typename T>
bool resize( ndarray::Array<T, 2, 1> & arr,
             size_t const newSizeRows,
             size_t const newSizeCols )
{
  /// create temporary array of new size and copy existing array into it
  ndarray::Array<T, 2, 1> arrOut = ndarray::allocate(newSizeRows, newSizeCols);
  arrOut.deep() = 0;
  for ( auto itArrRowIn = arr.begin(), itArrRowOut = arrOut.begin(); (itArrRowIn != arr.end()) && (itArrRowOut != arrOut.end()); ++itArrRowIn, ++itArrRowOut ) {
    for ( auto itArrColIn = itArrRowIn->begin(), itArrColOut = itArrRowOut->begin(); (itArrColIn != itArrRowIn->end()) && (itArrColOut != itArrRowOut->end()); ++itArrColIn, ++itArrColOut ) {
      *itArrColOut = *itArrColIn;
    }
  }

  /// resize input array and copy data back
  arr = ndarray::allocate(newSizeRows, newSizeCols);
  for ( auto itArrRowIn = arr.begin(), itArrRowOut = arrOut.begin(); (itArrRowIn != arr.end()) && (itArrRowOut != arrOut.end()); ++itArrRowIn, ++itArrRowOut ) {
    for ( auto itArrColIn = itArrRowIn->begin(), itArrColOut = itArrRowOut->begin(); (itArrColIn != itArrRowIn->end()) && (itArrColOut != itArrRowOut->end()); ++itArrColIn, ++itArrColOut ) {
      *itArrColIn = *itArrColOut;
    }
  }
  return true;
}


template< typename T, typename U >
bool checkIfValuesAreInRange( ndarray::Array<T, 1, 1> const& numbers,
                              ndarray::Array<U, 1, 1> const& range )
{
  for ( auto it = numbers.begin(); it != numbers.end(); ++it ) {
    if ( (*it < range[0]) || (*it > range[1]) )
      return false;
  }
  return true;
}

template< typename T >
ndarray::Array< T, 1, 1 > vectorToNdArray( std::vector< T > & vector, bool deep )
{
  ndarray::Array< T, 1, 1 > ndArray;
  if ( deep ) {
    ndarray::Array< T, 1, 1 > temp = ndarray::external(vector.data(), ndarray::makeVector(int( vector.size())), ndarray::makeVector(1));
    ndArray = copy(temp);
  }
  else {
    ndArray = ndarray::external(vector.data(), ndarray::makeVector(int( vector.size())), ndarray::makeVector(1));
  }
  return ndArray;
}

template< typename T >
ndarray::Array< T const, 1, 1 > vectorToNdArray( std::vector< T > const& vec_In, bool deep )
{
  ndarray::Array< T const, 1, 1 > arr_Out;
  if ( deep ) {
    ndarray::Array< T const, 1, 1 > temp = ndarray::external(vec_In.data(), ndarray::makeVector(int(vec_In.size())), ndarray::makeVector(1));
    arr_Out = copy(temp);
  }
  else {
    arr_Out = ndarray::external(vec_In.data(), ndarray::makeVector(int(vec_In.size())), ndarray::makeVector(1));
  }
  return arr_Out;
}

template< typename T >
int isMonotonic( ndarray::Array< T, 1, 1 > const& arrIn )
{
  int I_M = 0;
  if ( arrIn.getShape()[ 0 ] < 2 )
    return I_M;
  float D_DA = arrIn[ 1 ] - arrIn[ 0 ];
  if ( D_DA < 0 )
    I_M = -1;
  else if ( D_DA > 0 )
    I_M = 1;
  if ( arrIn.getShape()[ 0 ] < 3 )
    return I_M;
  for ( int i_pos = 2; i_pos < arrIn.getShape()[ 0 ]; i_pos++ ) {
    D_DA = arrIn[ i_pos ] - arrIn[ i_pos - 1 ];
    if ( (D_DA < 0) && (I_M == 1) )
      return 0;
    if ( (D_DA > 0) && (I_M == -1) )
      return 0;
    if ( I_M == 0 ) {
      if ( D_DA < 0 )
        I_M = -1;
      else if ( D_DA > 0 )
        I_M = 1;
    }
  }
  return I_M;
}

template< typename T >
T calcRMS( ndarray::Array< T, 1, 1 > const& arrIn )
{
  T rms = 0;
  for ( auto itArr = arrIn.begin(); itArr != arrIn.end(); ++itArr )
    rms += T(pow(float( *itArr), 2.));
  rms = rms / arrIn.getShape()[ 0 ];
  return sqrt(rms);
}

template< typename T >
T lsToFit( ndarray::Array< T, 1, 1 > const& XXVecArr,
           ndarray::Array< T, 1, 1 > const& YVecArr,
           T const& XM )
{
  T D_Out;
  ///Normalize to preserve significance.
  ndarray::Array< T, 1, 1 > XVecArr = ndarray::allocate(XXVecArr.getShape()[ 0 ]);
  XVecArr.deep() = XXVecArr - XXVecArr[0];

  int NDegree = 2;
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: NDegree set to " << NDegree << endl;
#endif

  long N = XXVecArr.size();
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: N set to " << N << endl;
#endif

  ///Correlation matrix
  ndarray::Array< T, 2, 1 > CorrMArr = ndarray::allocate(ndarray::makeVector(NDegree + 1, NDegree + 1));

  ndarray::Array< T, 1, 1 > BVecArr = ndarray::allocate(NDegree + 1);

  ///0 - Form the normal equations
  CorrMArr[ ndarray::makeVector(0, 0) ] = N;
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: CorrMArr(0,0) set to " << CorrMArr[ndarray::makeVector(0, 0)] << endl;
#endif

  BVecArr[0] = std::accumulate(YVecArr.begin(), YVecArr.end(), 0.);
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: BVecArr(0) set to " << BVecArr[0] << endl;
#endif

  ndarray::Array< T, 1, 1 > ZVecArr = ndarray::allocate(XXVecArr.getShape()[ 0 ]);
  ZVecArr.deep() = XVecArr;
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
#endif

  ndarray::Array< T, 1, 1 > TempVecArr = ndarray::allocate(YVecArr.getShape()[ 0 ]);
  TempVecArr.deep() = YVecArr;
  TempVecArr.deep() = TempVecArr * ZVecArr;
  BVecArr[ 1 ] = std::accumulate(TempVecArr.begin(), TempVecArr.end(), 0.);
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: BVecArr(1) set to " << BVecArr[1] << endl;
#endif

  CorrMArr[ ndarray::makeVector(0, 1) ] = std::accumulate(ZVecArr.begin(), ZVecArr.end(), 0.);
  CorrMArr[ ndarray::makeVector(1, 0) ] = CorrMArr[ ndarray::makeVector(0, 1) ];
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: CorrMArr(0,1) set to " << CorrMArr[ndarray::makeVector(0, 1)] << endl;
  cout << "CFits::LsToFit: CorrMArr(1,0) set to " << CorrMArr[ndarray::makeVector(1, 0)] << endl;
#endif

  ZVecArr.deep() = ZVecArr * XVecArr;
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
#endif

  TempVecArr = ndarray::allocate(YVecArr.getShape()[ 0 ]);
  TempVecArr.deep() = YVecArr;
  TempVecArr.deep() = TempVecArr * ZVecArr;
  BVecArr[ 2 ] = std::accumulate(TempVecArr.begin(), TempVecArr.end(), 0.);
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: BVecArr(2) set to " << BVecArr[2] << endl;
#endif

  CorrMArr[ ndarray::makeVector(0, 2) ] = CorrMArr[ ndarray::makeVector(1, 1) ] = CorrMArr[ ndarray::makeVector(2, 0) ] = std::accumulate(ZVecArr.begin(), ZVecArr.end(), 0.);
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: CorrMArr(0,2) set to " << CorrMArr[ndarray::makeVector(0, 2)] << endl;
  cout << "CFits::LsToFit: CorrMArr(1,1) set to " << CorrMArr[ndarray::makeVector(1, 1)] << endl;
  cout << "CFits::LsToFit: CorrMArr(2,0) set to " << CorrMArr[ndarray::makeVector(2, 0)] << endl;
#endif

  ZVecArr.deep() = ZVecArr * XVecArr;
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
#endif

  CorrMArr[ ndarray::makeVector(1, 2) ] = CorrMArr[ ndarray::makeVector(2, 1) ] = sum(ZVecArr);
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: CorrMArr(1,2) set to " << CorrMArr[ndarray::makeVector(1, 2)] << endl;
  cout << "CFits::LsToFit: CorrMArr(2,1) set to " << CorrMArr[ndarray::makeVector(2, 1)] << endl;
#endif

  TempVecArr = ndarray::allocate(ZVecArr.getShape()[ 0 ]);
  TempVecArr.deep() = ZVecArr;
  TempVecArr.deep() = TempVecArr * XVecArr;
  CorrMArr[ ndarray::makeVector(2, 2) ] = std::accumulate(TempVecArr.begin(), TempVecArr.end(), 0.);
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: CorrMArr(2,2) set to " << CorrMArr[ndarray::makeVector(2, 2)] << endl;
#endif

  ndarray::Array< T, 2, 1 > CorrInvMArr = ndarray::allocate(ndarray::makeVector(CorrMArr.getShape()[ 0 ], CorrMArr.getShape()[ 1 ]));
  CorrInvMArr.asEigen() = CorrMArr.asEigen().inverse();
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: CorrInvMArr set to " << CorrInvMArr << endl;
#endif
  ndarray::Array< T, 1, 1 > p_CVecArr = ndarray::allocate(BVecArr.getShape()[ 1 ]);
  p_CVecArr.asEigen() = BVecArr.asEigen().transpose() * CorrInvMArr.asEigen();
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: p_CVecArr set to " << p_CVecArr << endl;
#endif

  float XM0 = XM - XXVecArr[0];
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: XM0 set to " << XM0 << endl;
#endif

  D_Out = p_CVecArr[ 0 ] + (p_CVecArr[ 1 ] * XM0) + (p_CVecArr[ 2 ] * pow(XM0, 2));
#ifdef __DEBUG_FITS_LSTOFIT__
  cout << "CFits::LsToFit: D_Out set to " << D_Out << endl;
#endif
  return D_Out;
}

template< typename T, int I >
ndarray::Array< T, 1, 1 > hInterPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                                     ndarray::Array< T, 1, 1 > const& XVecArr,
                                     ndarray::Array< int, 1, 1 > & SVecArr,
                                     ndarray::Array< T, 1, I > const& UVecArr,
                                     std::vector< string > const& CS_A1_In )
{
#ifdef __DEBUG_INTERPOL__
  cout << "CFits::HInterPol: VVecArr.size() = " << VVecArr.getShape()[ 0 ] << endl;
  cout << "CFits::HInterPol: XVecArr.size() = " << XVecArr.getShape()[ 0 ] << endl;
  cout << "CFits::HInterPol: SVecArr.size() = " << SVecArr.getShape()[ 0 ] << endl;
  cout << "CFits::HInterPol: UVecArr.size() = " << UVecArr.getShape()[ 0 ] << endl;
  cout << "CFits::HInterPol: CS_A1_In.size() = " << CS_A1_In.size() << endl;
#endif

  int M = VVecArr.getShape()[ 0 ];

  ndarray::Array< int, 1, 1 > IA1_Temp = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  IA1_Temp.deep() = 0;

  ndarray::Array< T, 1, 1 > DA1_Temp = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  DA1_Temp.deep() = 0.;

  ndarray::Array< T, 1, 1 > DA1_TempA = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  DA1_TempA.deep() = 0.;

  ndarray::Array< T, 1, 1 > DA1_VTempP1 = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  DA1_VTempP1.deep() = 0.;

  ndarray::Array< T, 1, 1 > DA1_VTemp = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  DA1_VTemp.deep() = 0.;

  ndarray::Array< T, 1, 1 > DA1_XTempP1 = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  DA1_XTempP1.deep() = 0.;

  ndarray::Array< T, 1, 1 > DA1_XTemp = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  DA1_XTemp.deep() = 0.;

  ndarray::Array< int, 1, 1 > IA1_STemp = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  IA1_STemp.deep() = 0;

  ndarray::Array< T, 1, 1 > PVecArr = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  PVecArr.deep() = 0.;

  ndarray::Array< T, 1, 1 > TmpVecArr = indGenNdArr(T(4));

  ndarray::Array< T, 1, 1 > T1VecArr = ndarray::allocate(4);
  T1VecArr.deep() = 0.;

  ndarray::Array< T, 1, 1 > T2VecArr = ndarray::allocate(4);
  T2VecArr.deep() = 0.;

  ndarray::Array< T, 1, 1 > X1VecArr = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  X1VecArr.deep() = 0.;

  ndarray::Array< T, 1, 1 > X0VecArr = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  X0VecArr.deep() = 0.;

  ndarray::Array< T, 1, 1 > X2VecArr = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  X2VecArr.deep() = 0.;

  ndarray::Array< T, 1, 1 > X0Arr = ndarray::allocate(4);
  X0Arr.deep() = 0.;

  ndarray::Array< T, 1, 1 > V0Arr = ndarray::allocate(4);
  V0Arr.deep() = 0.;

  ndarray::Array< T, 1, 1 > QArr = ndarray::allocate(SVecArr.getShape()[ 0 ]);
  QArr.deep() = 0.;

  /**
  Clip interval, which forces extrapolation.
  u[i] is between x[s[i]] and x[s[i]+1].
   **/
  int s0int;
  float s0;
  /// Least square fit quadratic, 4 points
  if ( pfs::drp::stella::utils::KeyWord_Set(CS_A1_In, std::string("LSQUADRATIC")) >= 0 ) {
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::HInterPol: KeywordSet(LSQUADRATIC)" << endl;
#endif
    SVecArr.deep() = where(SVecArr,
                           "<",
                           1,
                           1,
                           SVecArr);
    SVecArr.deep() = where(SVecArr,
                           ">",
                           M - 3,
                           M - 3,
                           SVecArr);
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::HInterPol: LSQUADRATIC: SVecArr.size() set to " << SVecArr.getShape()[ 0 ] << endl;
#endif
    PVecArr.deep() = VVecArr[ 0 ]; /// Result
    for ( int m = 0; m < SVecArr.getShape()[ 0 ]; m++ ) {
      s0 = float( SVecArr[ m ]) - 1.;
      s0int = ( int ) s0;
      TmpVecArr.deep() = TmpVecArr + s0;
      T1VecArr.deep() = XVecArr[ ndarray::view(s0int, s0int + 4) ];
      T2VecArr.deep() = VVecArr[ ndarray::view(s0int, s0int + 4) ];
#ifdef __DEBUG_INTERPOL__
      cout << "CFits::HInterPol: Starting LsToFit(T1VecArr, T2VecArr, UVecArr(m)" << endl;
#endif
      PVecArr[ m ] = lsToFit(T1VecArr, T2VecArr, UVecArr[ m ]);
    }
  }
  else if ( pfs::drp::stella::utils::KeyWord_Set(CS_A1_In, std::string("QUADRATIC")) >= 0 ) {
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::HInterPol: KeywordSet(QUADRATIC)" << endl;
#endif
    SVecArr.deep() = where(SVecArr,
                           "<",
                           1,
                           1,
                           SVecArr);
    SVecArr.deep() = where(SVecArr,
                           ">",
                           M - 2,
                           M - 2,
                           SVecArr);
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::HInterPol: QUADRATIC: SVecArr.size() set to " << SVecArr.getShape()[ 0 ] << endl;
#endif

    X1VecArr.deep() = getSubArray(XVecArr,
                                  SVecArr);

    IA1_Temp.deep() = SVecArr - 1;

    X0VecArr.deep() = getSubArray(XVecArr,
                                  IA1_Temp);

    IA1_Temp.deep() = SVecArr + 1;
    X2VecArr.deep() = getSubArray(XVecArr,
                                  IA1_Temp);

    IA1_Temp.deep() = SVecArr - 1;
    DA1_Temp.deep() = getSubArray(VVecArr,
                                  IA1_Temp);

    IA1_Temp.deep() = SVecArr + 1;
    DA1_TempA.deep() = getSubArray(VVecArr,
                                   IA1_Temp);

    PVecArr.deep() = DA1_Temp
            * (UVecArr - X1VecArr) * (UVecArr - X2VecArr)
            / ((X0VecArr - X1VecArr) * (X0VecArr - X2VecArr))
            + DA1_TempA
            * (UVecArr - X0VecArr) * (UVecArr - X1VecArr)
            / ((X2VecArr - X0VecArr) * (X2VecArr - X1VecArr));
  }
  else if ( pfs::drp::stella::utils::KeyWord_Set(CS_A1_In, std::string("SPLINE")) >= 0 ) {
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::HInterPol: KeywordSet(SPLINE)" << endl;
#endif
    SVecArr.deep() = where(SVecArr,
                           "<",
                           1,
                           1,
                           SVecArr);
    SVecArr.deep() = where(SVecArr,
                           ">",
                           M - 3,
                           M - 3,
                           SVecArr);
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::HInterPol: SPLINE: SVecArr.size() set to " << SVecArr.getShape()[ 0 ] << endl;
#endif
    PVecArr = ndarray::allocate(SVecArr.getShape()[ 0 ]);
    PVecArr.deep() = VVecArr(0);
    int SOld = -1;
    for ( int m = 0; m < SVecArr.getShape()[ 0 ]; m++ ) {
      s0 = SVecArr[ m ] - 1.;
      s0int = ( int ) s0;
      if ( abs(SOld - s0int) > 0 ) {
        X0Arr = ndarray::allocate(4);
        X0Arr.deep() = XVecArr[ ndarray::view(s0int, s0int + 4) ];
        V0Arr = ndarray::allocate(4);
        V0Arr.deep() = XVecArr[ ndarray::view(s0int, s0int + 4) ];
        QArr = splineI(X0Arr, V0Arr);
        SOld = s0int;
      }
      PVecArr[ m ] = splInt(X0Arr,
                            V0Arr,
                            QArr,
                            UVecArr[ m ]);
    }
  }
  else /// Linear, not regular
  {
    DA1_XTemp.deep() = getSubArray(XVecArr,
                                   SVecArr);
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::HInterPol: DA1_XTemp set to " << DA1_XTemp << endl;
#endif
    DA1_VTemp.deep() = getSubArray(VVecArr,
                                   SVecArr);
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::HInterPol: DA1_VTemp set to " << DA1_VTemp << endl;
#endif

    IA1_STemp.deep() = SVecArr + 1;
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::HInterPol: IA1_STemp set to " << IA1_STemp << endl;
#endif

    DA1_XTempP1 = getSubArray(XVecArr,
                              IA1_STemp);
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::HInterPol: DA1_XTempP1 set to " << DA1_XTempP1 << endl;
#endif

    DA1_VTempP1 = getSubArray(VVecArr,
                              IA1_STemp);
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::HInterPol: DA1_VTempP1 set to " << DA1_VTempP1 << endl;
#endif

    PVecArr.deep() = (UVecArr - DA1_XTemp)
            * (DA1_VTempP1 - DA1_VTemp)
            / (DA1_XTempP1 - DA1_XTemp)
            + DA1_VTemp;
  }
#ifdef __DEBUG_INTERPOL__
  cout << "CFits::HInterPol: Ready: Returning PVecArr = " << PVecArr << endl;
#endif

  return PVecArr;
}

template< typename T, typename U, int I >
ndarray::Array< U, 1, 1 > where( ndarray::Array< T, 1, I > const& arrayToCompareTo,
                                 std::string const& op,
                                 T const valueToCompareTo,
                                 U const valueIfTrue,
                                 U const valueIfFalse )
{
  if ( (op != "<") && (op != "<=") && (op != ">") && (op != ">=") && (op != "==") ) {
    std::string message("pfs::drp::stella::math::where: ERROR: op(=");
    message += op + ") not supported";
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  ndarray::Array< U, 1, 1 > arrOut = ndarray::allocate(arrayToCompareTo.getShape()[ 0 ]);
  auto itOut = arrOut.begin();
  for ( auto itComp = arrayToCompareTo.begin(); itComp != arrayToCompareTo.end(); ++itComp, ++itOut ) {
    if ( (op.compare("<") == 0) && (*itComp < valueToCompareTo) )
      *itOut = valueIfTrue;
    else if ( (op.compare("<=") == 0) && (*itComp <= valueToCompareTo) )
      *itOut = valueIfTrue;
    else if ( (op.compare(">") == 0) && (*itComp > valueToCompareTo) )
      *itOut = valueIfTrue;
    else if ( (op.compare(">=") == 0) && (*itComp >= valueToCompareTo) )
      *itOut = valueIfTrue;
    else if ( (op.compare("==") == 0) && (*itComp == valueToCompareTo) )
      *itOut = valueIfTrue;
    else
      *itOut = valueIfFalse;
  }
  return arrOut;
}

template< typename T, typename U >
ndarray::Array< U, 1, 1 > where( ndarray::Array< T, 1, 1 > const& arrayToCompareTo,
                                 std::string const& op,
                                 T const valueToCompareTo,
                                 U const valueIfTrue,
                                 ndarray::Array< U, 1, 1 > const& valuesIfFalse )
{
  if ( (op != "<") && (op != "<=") && (op != ">") && (op != ">=") && (op != "==") ) {
    std::string message("pfs::drp::stella::math::where: ERROR: op(=");
    message += op + ") not supported";
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if ( arrayToCompareTo.getShape()[ 0 ] != valuesIfFalse.getShape()[ 0 ] ) {
    std::string message("pfs::drp::stella::math::where: ERROR: input arrays must have same shape");
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  ndarray::Array< U, 1, 1 > arrOut = ndarray::allocate(arrayToCompareTo.getShape()[ 0 ]);
  auto itOut = arrOut.begin();
  auto itElse = valuesIfFalse.begin();
  for ( auto itComp = arrayToCompareTo.begin(); itComp != arrayToCompareTo.end(); ++itComp, ++itOut, ++itElse ) {
    if ( (op.compare("<") == 0) && (*itComp < valueToCompareTo) )
      *itOut = valueIfTrue;
    else if ( (op.compare("<=") == 0) && (*itComp <= valueToCompareTo) )
      *itOut = valueIfTrue;
    else if ( (op.compare(">") == 0) && (*itComp > valueToCompareTo) )
      *itOut = valueIfTrue;
    else if ( (op.compare(">=") == 0) && (*itComp >= valueToCompareTo) )
      *itOut = valueIfTrue;
    else if ( (op.compare("==") == 0) && (*itComp == valueToCompareTo) )
      *itOut = valueIfTrue;
    else
      *itOut = *itElse;
  }
  return arrOut;
}

template< typename T, typename U, int I >
ndarray::Array< U, 2, 1 > where( ndarray::Array< T, 2, I > const& arrayToCompareTo,
                                 std::string const& op,
                                 T const valueToCompareTo,
                                 U const valueIfTrue,
                                 U const valueIfFalse )
{
  if ( (op != "<") && (op != "<=") && (op != ">") && (op != ">=") && (op != "==") ) {
    std::string message("pfs::drp::stella::math::where: ERROR: op(=");
    message += op + ") not supported";
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
#ifdef __DEBUG_WHERE__
  cout << "where2D: array = " << arrayToCompareTo << endl;
  cout << "where2D: op = " << op << endl;
  cout << "where2D: valueToCompareTo = " << valueToCompareTo << endl;
  cout << "where2D: valueIfTrue = " << valueIfTrue << endl;
  cout << "where2D: valueIfFalse = " << valueIfFalse << endl;
#endif
  ndarray::Array< U, 2, 1 > arrOut = ndarray::allocate(arrayToCompareTo.getShape());
  auto itOutRow = arrOut.begin();
  int row = 0;
  for ( auto itCompRow = arrayToCompareTo.begin(); itCompRow != arrayToCompareTo.end(); ++itCompRow, ++itOutRow, ++row ) {
    auto itOutCol = itOutRow->begin();
    int col = 0;
    for ( auto itCompCol = itCompRow->begin(); itCompCol != itCompRow->end(); ++itCompCol, ++itOutCol, ++col ) {
      if ( (op.compare("<") == 0) && (*itCompCol < valueToCompareTo) ) {
#ifdef __DEBUG_WHERE__
        cout << "where2D: op = < && " << *itCompCol << " < " << valueToCompareTo << endl;
#endif
        *itOutCol = valueIfTrue;
      }
      else if ( (op.compare("<=") == 0) && (*itCompCol <= valueToCompareTo) ) {
#ifdef __DEBUG_WHERE__
        cout << "where2D: op <= < && " << *itCompCol << " <= " << valueToCompareTo << endl;
#endif
        *itOutCol = valueIfTrue;
      }
      else if ( (op.compare(">") == 0) && (*itCompCol > valueToCompareTo) ) {
#ifdef __DEBUG_WHERE__
        cout << "where2D: op = > && " << *itCompCol << " > " << valueToCompareTo << endl;
#endif
        *itOutCol = valueIfTrue;
      }
      else if ( (op.compare(">=") == 0) && (*itCompCol >= valueToCompareTo) ) {
#ifdef __DEBUG_WHERE__
        cout << "where2D: op = >= && " << *itCompCol << " >= " << valueToCompareTo << endl;
#endif
        *itOutCol = valueIfTrue;
      }
      else if ( (op.compare("==") == 0) && (*itCompCol == valueToCompareTo) ) {
#ifdef __DEBUG_WHERE__
        cout << "where2D: op = == && " << *itCompCol << " == " << valueToCompareTo << endl;
#endif
        *itOutCol = valueIfTrue;
      }
      else {
        *itOutCol = valueIfFalse;
      }
#ifdef __DEBUG_WHERE__
      cout << "where2D: *itOutCol = " << *itOutCol << endl;
#endif
    }
  }
  return arrOut;
}

template< typename T, typename U, int I, int J >
ndarray::Array< U, 2, 1 > where( ndarray::Array< T, 2, I > const& arrayToCompareTo,
                                 std::string const& op,
                                 T const valueToCompareTo,
                                 U const valueIfTrue,
                                 ndarray::Array< U, 2, J > const& valuesIfFalse )
{
  if ( (op != "<") && (op != "<=") && (op != ">") && (op != ">=") && (op != "==") ) {
    std::string message("pfs::drp::stella::math::where: ERROR: op(=");
    message += op + ") not supported";
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if ( arrayToCompareTo.getShape()[ 0 ] != valuesIfFalse.getShape()[ 0 ] ) {
    std::string message("pfs::drp::stella::math::where: ERROR: input arrays must have same shape");
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if ( arrayToCompareTo.getShape()[ 1 ] != valuesIfFalse.getShape()[ 1 ] ) {
    std::string message("pfs::drp::stella::math::where: ERROR: input arrays must have same shape");
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  ndarray::Array< U, 2, 1 > arrOut = ndarray::allocate(arrayToCompareTo.getShape());
  auto itOutRow = arrOut.begin();
  auto itElseRow = valuesIfFalse.begin();
  for ( auto itCompRow = arrayToCompareTo.begin(); itCompRow != arrayToCompareTo.end(); ++itCompRow, ++itOutRow, ++itElseRow ) {
    auto itOutCol = itOutRow->begin();
    auto itElseCol = itElseRow->begin();
    for ( auto itCompCol = itCompRow->begin(); itCompCol != itCompRow->end(); ++itCompCol, ++itOutCol, ++itElseCol ) {
      if ( (op.compare("<") == 0) && (*itCompCol < valueToCompareTo) )
        *itOutCol = valueIfTrue;
      else if ( (op.compare("<=") == 0) && (*itCompCol <= valueToCompareTo) )
        *itOutCol = valueIfTrue;
      else if ( (op.compare(">") == 0) && (*itCompCol > valueToCompareTo) )
        *itOutCol = valueIfTrue;
      else if ( (op.compare(">=") == 0) && (*itCompCol >= valueToCompareTo) )
        *itOutCol = valueIfTrue;
      else if ( (op.compare("==") == 0) && (*itCompCol == valueToCompareTo) )
        *itOutCol = valueIfTrue;
      else
        *itOutCol = *itElseCol;
    }
  }
  return arrOut;
}

template< typename T >
ndarray::Array< T, 1, 1 > splineI( ndarray::Array< T, 1, 1 > const& XVecArr,
                                   ndarray::Array< T, 1, 1 > const& YVecArr,
                                   T const YP1,
                                   T const YPN )
{
  int m, o, N = XVecArr.getShape()[ 0 ];
  float p, qn, sig, un;
  ndarray::Array< T, 1, 1 > UVecArr = ndarray::allocate(N - 1);
  ndarray::Array< T, 1, 1 > Y2VecArr = ndarray::allocate(N);

  if ( YP1 > 0.99e30 ) /// The lower boundary condition is set either to be "natural"
  {
    Y2VecArr[ 0 ] = UVecArr[ 0 ] = 0.0;
  }
  else /// or else to have a specified first derivative
  {
    Y2VecArr[ 0 ] = -0.5;
    UVecArr[ 0 ] = (3.0 / (XVecArr[ 1 ] - XVecArr[ 0 ])) * ((YVecArr[ 1 ] - YVecArr[ 0 ]) / (XVecArr[ 1 ] - XVecArr[ 0 ]) - YP1);
  }

  /**
  This is the decomposition loop of the tridiagonal algorithm. Y2VecArr and UVecArr are used for temporary storage of the decomposed factors.
   **/
  for ( m = 1; m < N - 1; ++m ) {
    sig = (XVecArr[ m ] - XVecArr[ m - 1 ]) / (XVecArr[ m + 1 ] - XVecArr[ m - 1 ]);
    p = sig * Y2VecArr[ m - 1 ] + 2.0;
    Y2VecArr[ m ] = (sig - 1.0) / p;
    UVecArr[ m ] = (YVecArr[ m + 1 ] - YVecArr[ m ]) / (XVecArr[ m + 1 ] - XVecArr[ m ]) - (YVecArr[ m ] - YVecArr[ m - 1 ]) / (XVecArr[ m ] - XVecArr[ m - 1 ]);
    UVecArr[ m ] = (6.0 * UVecArr[ m ] / (XVecArr[ m + 1 ] - XVecArr[ m - 1 ]) - sig * UVecArr[ m - 1 ]) / p;
  }
  if ( YPN > 0.99e30 ) /// The upper boundary condition is set either to be "natural"
    qn = un = 0.0;
  else /// or else to have a specified first derivative
  {
    qn = 0.5;
    un = (3.0 / (XVecArr[ N - 1 ] - XVecArr[ N - 2 ])) * (YPN - (YVecArr[ N - 1 ] - YVecArr[ N - 2 ]) / (XVecArr[ N - 1 ] - XVecArr[ N - 2 ]));
  }
  Y2VecArr[ N - 1 ] = (un - qn * UVecArr[ N - 2 ]) / (qn * Y2VecArr[ N - 2 ] + 1.0);

  /// This is the backsubstitution loop of the tridiagonal algorithm
  for ( o = N - 2; o >= 0; o-- ) {
    Y2VecArr[ o ] = Y2VecArr[ o ] * Y2VecArr[ o + 1 ] + UVecArr[ o ];
  }
  return Y2VecArr;
}

template< typename T >
ndarray::Array< T, 1, 1 > splineI( ndarray::Array< T, 1, 1 > const& XVecArr,
                                   ndarray::Array< T, 1, 1 > const& YVecArr )
{
  return splineI(XVecArr,
                 YVecArr,
                 T(1.0e30),
                 T(1.0e30));
}

template< typename T >
T splInt( ndarray::Array< T, 1, 1 > const& XAVecArr,
          ndarray::Array< T, 1, 1 > const& YAVecArr,
          ndarray::Array< T, 1, 1> const& Y2AVecArr,
          T X )
{
  int klo, khi, o, N;
  float h, b, a;
  T Y;

  N = XAVecArr.getShape()[ 0 ];
  /**
   *  We will find the right place in the table by means of bisection. This is optimal
   * if sequential calls to this routine are at random values of X. If sequential calls
   * are in order, and closely spaced, one would do better to store previous values of
   * klo and khi and test if they remain appropriate on the next call.
   **/
  klo = 1;
  khi = N;
  while ( khi - klo > 1 ) {
    o = (khi + klo) >> 1;
    if ( XAVecArr[ o ] > X )
      khi = o;
    else
      klo = o;
  } /// klo and khi now bracket the input value of X
  h = XAVecArr[ khi ] - XAVecArr[ klo ];
  if ( h == 0.0 ) { /// The XAVecArr(i)'s must be distinct
      throw std::runtime_error("splInt: ERROR: Bad XAVecArr input to routine SplInt");
  }
  a = (XAVecArr[ khi ] - X) / h;
  b = (X - XAVecArr[ klo ]) / h; /// Cubic Spline polynomial is now evaluated.
  Y = a * YAVecArr[ klo ] + b * YAVecArr[ khi ] + ((a * a * a - a) * Y2AVecArr[ khi ]) * (h * h) / 6.0;
  return Y;
}

/**
  InterPol linear, not regular
 **/
template< typename T >
ndarray::Array< T, 1, 1 > interPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                                    ndarray::Array< T, 1, 1 > const& XVecArr,
                                    ndarray::Array< T, 1, 1 > const& UVecArr )
{
  return interPol(VVecArr,
                  XVecArr,
                  UVecArr,
                  false);
}

template< typename T, int I >
ndarray::Array< T, 1, 1 > interPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                                    ndarray::Array< T, 1, 1 > const& XVecArr,
                                    ndarray::Array< T, 1, I > const& UVecArr,
                                    bool B_PreserveFlux )
{
  std::vector< std::string > cs_a1(1);
  cs_a1[ 0 ] = std::string(" ");
  if ( B_PreserveFlux ) {
    ndarray::Array< T, 1, 1 > D_A1_Out = ndarray::allocate(UVecArr.getShape()[ 0 ]);
    ndarray::Array< T, 1, 1 > D_A1_U = ndarray::allocate(2);
    ndarray::Array< T, 1, 1 > D_A1_X = ndarray::allocate(XVecArr.getShape()[ 0 ] + 1);
    D_A1_X[ 0 ] = XVecArr[ 0 ] - ((XVecArr[ 1 ] - XVecArr[ 0 ]) / 2.);
    D_A1_X[ D_A1_X.getShape()[ 0 ] - 1 ] = XVecArr[ XVecArr.getShape()[ 0 ] - 1 ] + ((XVecArr[ XVecArr.getShape()[ 0 ] - 1 ] - XVecArr[ XVecArr.getShape()[ 0 ] - 2]) / 2.);
    for ( int i_pix = 1; i_pix < XVecArr.getShape()[ 0 ]; ++i_pix ) {
      D_A1_X[ i_pix ] = XVecArr[ i_pix - 1 ] + ((XVecArr[ i_pix ] - XVecArr[ i_pix - 1 ]) / 2.);
    }
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::InterPol: XVecArr = " << XVecArr << endl;
    cout << "CFits::InterPol: D_A1_X = " << D_A1_X << endl;
#endif

    ndarray::Array< int, 1, 1 > I_A1_Ind = ndarray::allocate(D_A1_X.getShape()[ 0 ]);
    ndarray::Array< size_t, 1, 1 > P_I_A1_Ind;
    int I_Start = 0;
    int I_NInd = 0;
    float D_Start, D_End;
    for ( int i_pix = 0; i_pix < UVecArr.getShape()[ 0 ]; ++i_pix ) {
      if ( i_pix == 0 ) {
        D_A1_U[ 0 ] = UVecArr[ 0 ] - ((UVecArr[ 1 ] - UVecArr[ 0 ]) / 2.);
        D_A1_U[ 1 ] = UVecArr[ 0 ] + ((UVecArr[ 1 ] - UVecArr[ 0 ]) / 2.);
      }
      else if ( i_pix == UVecArr.getShape()[ 0 ] - 1 ) {
        D_A1_U[ 0 ] = UVecArr[ UVecArr.getShape()[ 0 ] - 1 ] - ((UVecArr[ UVecArr.getShape()[ 0 ] - 1 ] - UVecArr[ UVecArr.getShape()[ 0 ] - 2 ]) / 2.);
        D_A1_U[ 1 ] = UVecArr[ UVecArr.getShape()[ 0 ] - 1 ] + ((UVecArr[ UVecArr.getShape()[ 0 ] - 1 ] - UVecArr[ UVecArr.getShape()[ 0 ] - 2 ]) / 2.);
      }
      else {
        D_A1_U[ 0 ] = UVecArr[ i_pix ] - ((UVecArr[ i_pix ] - UVecArr[ i_pix - 1 ]) / 2.);
        D_A1_U[ 1 ] = UVecArr[ i_pix ] + ((UVecArr[ i_pix + 1 ] - UVecArr[ i_pix ]) / 2.);
      }
      I_A1_Ind = where(D_A1_X,
                       "<",
                       D_A1_U[ 0 ],
                       1,
                       0);
      P_I_A1_Ind = getIndices(I_A1_Ind);
      I_NInd = P_I_A1_Ind.getShape()[ 0 ];
      if ( I_NInd < 1 ) {
#ifdef __DEBUG_INTERPOL__
        cout << "CFits::InterPol: WARNING: 1. I_A1_Ind = " << I_A1_Ind << ": I_NInd < 1" << endl;
#endif
        I_Start = 0;
      }
      else {
        I_Start = P_I_A1_Ind[ I_NInd - 1 ];
      }
#ifdef __DEBUG_INTERPOL__
      cout << "CFits::InterPol: i_pix = " << i_pix << ": D_A1_U = " << D_A1_U << endl;
#endif
      I_A1_Ind = where(D_A1_X,
                       ">",
                       D_A1_U(1),
                       1,
                       0);
      P_I_A1_Ind = getIndices(I_A1_Ind);
      I_NInd = P_I_A1_Ind.getShape()[ 0 ];

      D_Start = D_A1_U[ 0 ];
      if ( D_A1_X[ I_Start ] > D_A1_U[ 0 ] )
        D_Start = D_A1_X[ I_Start ];
      D_A1_Out[ i_pix ] = 0.;
      if ( (D_A1_U[ 1 ] > D_A1_X[ 0 ]) && (D_A1_U[ 0 ] < D_A1_X[ D_A1_X.getShape()[ 0 ] - 1 ]) ) {
        do {
          if ( D_A1_U[ 1 ] < D_A1_X[ I_Start + 1 ] ) {
            D_End = D_A1_U[ 1 ];
          }
          else {
            D_End = D_A1_X[ I_Start + 1 ];
          }
#ifdef __DEBUG_INTERPOL__
          cout << "CFits::InterPol: i_pix = " << i_pix << ": D_Start = " << D_Start << ", D_End = " << D_End << endl;
#endif
          D_A1_Out[ i_pix ] += VVecArr[ I_Start ] * (D_End - D_Start) / (D_A1_X[ I_Start + 1 ] - D_A1_X[ I_Start ]);
          D_Start = D_End;
          if ( D_A1_U[ 1 ] >= D_A1_X[ I_Start + 1 ] )
            I_Start++;
#ifdef __DEBUG_INTERPOL__
          cout << "CFits::InterPol: i_pix = " << i_pix << ": D_A1_Out(" << i_pix << ") = " << D_A1_Out[i_pix] << endl;
#endif
          if ( I_Start + 1 >= D_A1_X.getShape()[ 0 ] )
            break;
        }
        while ( D_End < D_A1_U[ 1 ]-((D_A1_U[ 1 ] - D_A1_U[ 0 ]) / 100000000.) );
      }
    }
    return D_A1_Out;
  }

  return interPol(VVecArr,
                  XVecArr,
                  UVecArr,
                  cs_a1);
}

template< typename T, int I >
ndarray::Array< T, 1, 1 > interPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                                    ndarray::Array< T, 1, 1 > const& XVecArr,
                                    ndarray::Array< T, 1, I > const& UVecArr,
                                    std::vector< std::string > const& CS_A1_In )
{
#ifdef __DEBUG_INTERPOL__
  cout << "CFits::InterPol: VVecArr.size() = " << VVecArr.getShape()[ 0 ] << endl;
  cout << "CFits::InterPol: XVecArr.size() = " << XVecArr.getShape()[ 0 ] << endl;
  cout << "CFits::InterPol: UVecArr.size() = " << UVecArr.getShape()[ 0 ] << endl;
  cout << "CFits::InterPol: CS_A1_In.size() = " << CS_A1_In.size() << endl;
  cout << "CFits::InterPol(D_A1_V = " << VVecArr << ", D_A1_X = " << XVecArr << ", D_A1_U = " << UVecArr << ", CS_A1_In) Started" << endl;
#endif

  const int M = VVecArr.getShape()[ 0 ];
#ifdef __DEBUG_INTERPOL__
  cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, CS_A1_In): M set to " << M << endl;
#endif

  if ( XVecArr.getShape()[ 0 ] != M ) {
      string message("pfs::drp::stella::math::interPol: XVecArr and VVecArr must have same # of elements! ");
      message += to_string(XVecArr.getShape()[0]) + " != " + to_string(M);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  ndarray::Array< int, 1, 1 > SVecArr = valueLocate(XVecArr, UVecArr);
#ifdef __DEBUG_INTERPOL__
  cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, CS_A1_In): SVecArr set to " << SVecArr << endl;
#endif
  SVecArr.deep() = where(SVecArr,
                         "<",
                         0,
                         0,
                         SVecArr);
#ifdef __DEBUG_INTERPOL__
  cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, CS_A1_In): SVecArr set to " << SVecArr << endl;
#endif

  SVecArr.deep() = where(SVecArr,
                         ">",
                         M - 2,
                         M - 2,
                         SVecArr);
#ifdef __DEBUG_INTERPOL__
  cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, CS_A1_In): SVecArr set to " << SVecArr << endl;
  cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, CS_A1_In): Starting HInterPol " << endl;
#endif
  return hInterPol(VVecArr,
                   XVecArr,
                   SVecArr,
                   UVecArr,
                   CS_A1_In);
}

template< typename T, int I >
ndarray::Array< int, 1, 1 > valueLocate( ndarray::Array< T, 1, 1 > const& VecArr,
                                         ndarray::Array< T, 1, I > const& ValVecArr )
{
#ifdef __DEBUG_INTERPOL__
  cout << "CFits::ValueLocate: VecArr = " << VecArr << endl;
  cout << "CFits::ValueLocate: ValVecArr = " << ValVecArr << endl;
#endif
  if ( VecArr.getShape()[ 0 ] < 1 ) {
      throw std::runtime_error("CFits::ValueLocate: ERROR: VecArr.size() < 1 => Returning FALSE");
  }
  if ( ValVecArr.getShape()[ 0 ] < 1 ) {
      throw std::runtime_error("CFits::ValueLocate: ERROR: ValVecArr.size() < 1 => Returning FALSE");
  }
  ndarray::Array< int, 1, 1 > IntVecArr = ndarray::allocate(ValVecArr.getShape()[ 0 ]);

  int n;
  int N = VecArr.getShape()[ 0 ];
  int M = ValVecArr.getShape()[ 0 ];

  bool Increasing = false;
  int ii = 0;
  while ( VecArr[ ii ] == VecArr[ ii + 1 ] ) {
    ii++;
  }
  if ( VecArr[ ii + 1 ] > VecArr[ ii ] )
    Increasing = true;

#ifdef __DEBUG_INTERPOL__
  if ( Increasing )
    cout << "CFits::ValueLocate: Increasing = TRUE" << endl;
  else
    cout << "CFits::ValueLocate: Increasing = FALSE" << endl;
#endif

  /// For every element in ValVecArr
  for ( int m = 0; m < M; m++ ) {
#ifdef __DEBUG_INTERPOL__
    cout << "CFits::ValueLocate: ValVecArr(m) = " << ValVecArr[m] << endl;
#endif
    if ( Increasing ) {
      if ( ValVecArr[ m ] < VecArr[ 0 ] ) {
        IntVecArr[ m ] = 0 - 1;
      }
      else if ( VecArr[ N - 1 ] <= ValVecArr[ m ] ) {
        IntVecArr[ m ] = N - 1;
      }
      else {
        n = -1;
        while ( n < N - 1 ) {
          n++;
          if ( (VecArr[ n ] <= ValVecArr[ m ]) && (ValVecArr[ m ] < VecArr[ n + 1 ]) ) {
            IntVecArr[ m ] = n;
            break;
          }
        }
      }
#ifdef __DEBUG_INTERPOL__
      cout << "CFits::ValueLocate: Increasing = TRUE: IntVecArr(m) = " << IntVecArr[m] << endl;
#endif
    }
    else {/// if (Decreasing)
      if ( VecArr[ 0 ] <= ValVecArr[ m ] )
        IntVecArr[ m ] = 0 - 1;
      else if ( ValVecArr[ m ] < VecArr[ N - 1 ] )
        IntVecArr[ m ] = N - 1;
      else {
        n = -1;
        while ( n < N - 1 ) {
          n++;
          if ( (VecArr[ n + 1 ] <= ValVecArr[ m ]) && (ValVecArr[ m ] < VecArr[ n ]) ) {
            IntVecArr[ m ] = n;
            break;
          }
        }
      }
#ifdef __DEBUG_INTERPOL__
      cout << "CFits::ValueLocate: Increasing = FALSE: IntVecArr(m) = " << IntVecArr[m] << endl;
#endif
    }
  }
#ifdef __DEBUG_INTERPOL__
  cout << "CFits::ValueLocate: IntVecArr = " << IntVecArr << endl;
#endif
  return IntVecArr;
}

template< typename T >
int find( ndarray::Array< T, 1, 1 > const& arrToSearch,
          T val )
{
  int pos = 0;
  for ( auto it = arrToSearch.begin(); it != arrToSearch.end(); ++it, ++pos ) {
    if ( std::fabs(float(*it) - float(val)) < 0.000000000001 )
      return pos;
  }
  return -1;
}

template int find( ndarray::Array< size_t, 1, 1 > const&, size_t );
template ndarray::Array<float, 1, 1> interPol(ndarray::Array< float, 1, 1 > const&,
                                              ndarray::Array< float, 1, 1 > const&,
                                              ndarray::Array< float, 1, 1 > const&);
template ndarray::Array<float, 1, 1> interPol(ndarray::Array< float, 1, 1 > const&,
                                              ndarray::Array< float, 1, 1 > const&,
                                              ndarray::Array< float, 1, 1 > const&,
                                              bool);
template ndarray::Array<float, 1, 1> interPol(ndarray::Array< float, 1, 1 > const&,
                                              ndarray::Array< float, 1, 1 > const&,
                                              ndarray::Array< float, 1, 0 > const&,
                                              std::vector< std::string > const&);
template ndarray::Array<int, 1, 1> valueLocate(ndarray::Array< float, 1, 1 > const&,
                                               ndarray::Array< float, 1, 1 > const& );
template ndarray::Array<float, 1, 1> splineI( ndarray::Array< float, 1, 1 > const&,
                                              ndarray::Array< float, 1, 1 > const&, float, float);
template ndarray::Array< short int, 1, 1 > where( ndarray::Array< float, 1, 0 > const&,
                                                  std::string const&,
                                                  float const,
                                                  short int const,
                                                  short int const );
template ndarray::Array< int, 1, 1 > where( ndarray::Array< float, 1, 1 > const&,
                                            std::string const&,
                                            float const,
                                            int const,
                                            int const );
template ndarray::Array< int, 2, 1 > where( ndarray::Array< size_t, 2, 1 > const&,
                                            std::string const&,
                                            size_t const,
                                            int const,
                                            int const );
template ndarray::Array< int, 2, 1 > where( ndarray::Array< int, 2, 1 > const&,
                                            std::string const&,
                                            int const,
                                            int const,
                                            int const );
template ndarray::Array< int, 2, 1 > where( ndarray::Array< float, 2, 1 > const&,
                                            std::string const&,
                                            float const,
                                            int const,
                                            int const );
template ndarray::Array< int, 1, 1 > where( ndarray::Array< int, 1, 1 > const&,
                                            std::string const&,
                                            int const,
                                            int const,
                                            ndarray::Array< int, 1, 1 > const& );
template ndarray::Array< float, 2, 1 > where( ndarray::Array< float, 2, 1 > const&,
                                              std::string const&,
                                              float const,
                                              float const,
                                              ndarray::Array< float, 2, 1 > const& );
template float calcRMS( ndarray::Array< float, 1, 1 > const& );
template int isMonotonic( ndarray::Array< float, 1, 1 > const& );
template void insertSorted( std::vector< dataXY< float > > &, dataXY< float > & );
template ndarray::Array< unsigned long const, 1, 1> vectorToNdArray( std::vector<unsigned long> const&, bool );
template ndarray::Array<float, 1, 1> vectorToNdArray( std::vector<float> &, bool );
template ndarray::Array<unsigned long, 1, 1> vectorToNdArray( std::vector<unsigned long> &, bool );
template bool checkIfValuesAreInRange( ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const& range );
template ndarray::Array<float, 1, 1> convertRangeToUnity( ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const& );
template bool resize( ndarray::Array< float, 1, 1 > &, size_t );
template bool resize( ndarray::Array< float, 2, 1 > &, size_t, size_t );
template ndarray::Array<float, 1, 1> getSubArray( ndarray::Array<float, 1, 1> const&, ndarray::Array<size_t, 1, 1> const& );
template ndarray::Array<float, 1, 1> getSubArray( ndarray::Array<float, 2, 1> const&, ndarray::Array<size_t, 2, 1> const& );
template ndarray::Array<float, 1, 1> getSubArray( ndarray::Array<float, 2, 1> const&, std::vector< std::pair<size_t, size_t> > const& );
template std::vector<size_t> getIndices( std::vector<int> const& );
template ndarray::Array< size_t, 1, 1 > getIndices( ndarray::Array< int, 1, 1 > const& );
template ndarray::Array< size_t, 2, 1 > getIndices( ndarray::Array< int, 2, 1 > const& );
template ndarray::Array<size_t, 1, 1> getIndicesInValueRange( ndarray::Array<float, 1, 0> const&, float const, float const );
template ndarray::Array<size_t, 1, 1> getIndicesInValueRange( ndarray::Array<float, 1, 1> const&, float const, float const );
template ndarray::Array<size_t, 2, 1> getIndicesInValueRange( ndarray::Array<size_t, 2, 1> const&, size_t const, size_t const );
template ndarray::Array<size_t, 2, 1> getIndicesInValueRange( ndarray::Array<float, 2, 1> const&, float const, float const );
template ndarray::Array<float, 1, 1> replicate( float const val, int const size );
template ndarray::Array<float, 1, 1> indGenNdArr( float const );
template float min( ndarray::Array<float, 1, 1> const& );
template float max( ndarray::Array<float, 1, 1> const& );
template ndarray::Array<size_t, 1, 1> floor( const ndarray::Array<const float, 1, 1>&, const size_t );
template ndarray::Array<size_t, 1, 1> floor( const ndarray::Array<float, 1, 1>&, const size_t );
template int firstIndexWithZeroValueFrom( ndarray::Array<int, 1, 1> const& vec_In,
                                          const int startPos_In );
template int firstIndexWithValueGEFrom( ndarray::Array<int, 1, 1> const& vecIn,
                                        const int minValue,
                                        const int fromIndex );
template int lastIndexWithZeroValueBefore( ndarray::Array<int, 1, 1> const& vec_In,
                                           const int startPos_In );
template ndarray::Array<float, 1, 1> moment( const ndarray::Array<float, 1, 1> &D_A1_Arr_In, int I_MaxMoment_In );
template std::vector<int> sortIndices( const std::vector<float> &vec_In );
template ndarray::Array<size_t, 2, -2> calcMinCenMax( ndarray::Array<float, 1, 1> const&, float const, float const, int const, int const );
}/// end namespace math
}
}
}

#endif
