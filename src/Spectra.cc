#include "pfs/drp/stella/Spectra.h"

namespace pfsDRPStella = pfs::drp::stella;

/** @brief Construct a Spectrum with empty vectors of specified size (default 0)
 */
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::Spectrum(size_t length,
                                                                           size_t iTrace) 
  : _length(length),
    _iTrace(iTrace),
    _isWavelengthSet(false)
{
  _spectrum = ndarray::allocate(length);
  _mask = ndarray::allocate(length);
  _variance = ndarray::allocate(length);
  _wavelength = ndarray::allocate(length);
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::Spectrum(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> const& spectrum,
                                                                           size_t iTrace) 
  : _length(spectrum.getLength()),
    _iTrace(spectrum.getITrace()),
    _isWavelengthSet(spectrum.isWavelengthSet())
{
  _spectrum = ndarray::allocate(spectrum.getSpectrum().getShape()[0]);
  _mask = ndarray::allocate(spectrum.getMask().getShape()[0]);
  _variance = ndarray::allocate(spectrum.getVariance().getShape()[0]);
  _wavelength = ndarray::allocate(spectrum.getWavelength().getShape()[0]);
  _spectrum.deep() = spectrum.getSpectrum();
  _mask.deep() = spectrum.getMask();
  _variance.deep() = spectrum.getVariance();
  _wavelength.deep() = spectrum.getWavelength();
  if (iTrace != 0)
    _iTrace = iTrace;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum(const ndarray::Array<SpectrumT, 1, 1> & spectrum)
{
  /// Check length of input spectrum
  if (spectrum.getShape()[0] != _length){
    string message("pfsDRPStella::Spectrum::setSpectrum: ERROR: spectrum->size()=");
    message += to_string(spectrum.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _spectrum.deep() = spectrum;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setVariance(const ndarray::Array<VarianceT, 1, 1> & variance)
{
  /// Check length of input variance
  if (variance.getShape()[0] != _length){
    string message("pfsDRPStella::Spectrum::setVariance: ERROR: variance->size()=");
    message += to_string(variance.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _variance.deep() = variance;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setWavelength(const ndarray::Array<WavelengthT, 1, 1> & wavelength)
{
  /// Check length of input wavelength
  if (wavelength.getShape()[0] != _length){
    string message("pfsDRPStella::Spectrum::setWavelength: ERROR: wavelength->size()=");
    message += to_string(wavelength.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _wavelength.deep() = wavelength;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setMask(const ndarray::Array<MaskT, 1, 1> & mask)
{
  /// Check length of input mask
  if (mask.getShape()[0] != _length){
    string message("pfsDRPStella::Spectrum::setMask: ERROR: mask->size()=");
    message += to_string(mask.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _mask.deep() = mask;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setLength(const size_t length){
  pfsDRPStella::math::resize(_spectrum, length);
  pfsDRPStella::math::resize(_mask, length);
  pfsDRPStella::math::resize(_variance, length);
  pfsDRPStella::math::resize(_wavelength, length);
  if (length > _length){
    WavelengthT val = _wavelength[_length = 1];
    for (auto it = _wavelength.begin() + length; it != _wavelength.end(); ++it)
      *it = val;
  }
  _length = length;
  return true;
}

///SpectrumSet
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::SpectrumSet(size_t nSpectra, size_t length)
        : _spectra(new std::vector<PTR(pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)>())
{
  for (size_t i = 0; i < nSpectra; ++i){
    PTR(pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) spec(new pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>(length));
    _spectra->push_back(spec);
    (*_spectra)[i]->setITrace(i);
  }
}
    
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::SpectrumSet(const PTR(std::vector<PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)>) &spectrumVector)
        : _spectra(spectrumVector)
{}
//  for (int i = 0; i < spectrumVector->size(); ++i){
//    (*_spectra)[i]->setITrace(i);
//  }
//}
    
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum(const size_t i,     /// which spectrum?
                     const PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) & spectrum /// the Spectrum for the ith aperture
                      )
{
  if (i > _spectra->size()){
    string message("SpectrumSet::setSpectrum(i=");
    message += to_string(i) + "): ERROR: i > _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  if (i == static_cast<int>(_spectra->size())){
    _spectra->push_back(spectrum);
  }
  else{
    (*_spectra)[i] = spectrum;
  }
  return true;
}

/// add one Spectrum to the set
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
void pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::addSpectrum(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> const& spectrum /// the Spectrum to add
)
{
  PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) ptr(new Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>(spectrum));
  _spectra->push_back(ptr);
  return;
}
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
void pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::addSpectrum(PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) const& spectrum /// the Spectrum to add
)
{
  _spectra->push_back(spectrum);
  return;
}
  
template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT>)& pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum(const size_t i){
  if (i >= _spectra->size()){
    string message("SpectrumSet::getSpectrum(i=");
    message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  return _spectra->at(i); 
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT>) const& pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum(const size_t i) const { 
  if (i >= _spectra->size()){
    string message("SpectrumSet::getSpectrum(i=");
    message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  return PTR(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT>)(_spectra->at(i)); 
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::erase(const size_t iStart, const size_t iEnd){
  if (iStart >= _spectra->size()){
    string message("SpectrumSet::erase(iStart=");
    message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iStart >= _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  if (iEnd >= _spectra->size()){
    string message("SpectrumSet::erase(iStart=");
    message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iEnd >= _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  if (iEnd > 0){
    if (iStart > iEnd){
      string message("SpectrumSet::erase(iStart=");
      message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iStart > iEnd";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
  }
  if (iStart == (_spectra->size()-1)){
    _spectra->pop_back();
  }
  else{
    if (iEnd == 0)
      _spectra->erase(_spectra->begin() + iStart);
    else
      _spectra->erase(_spectra->begin() + iStart, _spectra->begin() + iEnd);
  }
  return true;
}


template<typename T>
PTR(T) pfsDRPStella::utils::getPointer(T &obj){
  PTR(T) pointer(new T(obj));
  return pointer;
}

//template class pfsDRPStella::Spectrum<float>;
//template class pfsDRPStella::Spectrum<double>;
template class pfsDRPStella::Spectrum<float, unsigned int, float, float>;
template class pfsDRPStella::Spectrum<double, unsigned int, float, float>;
template class pfsDRPStella::Spectrum<float, unsigned short, float, float>;
template class pfsDRPStella::Spectrum<double, unsigned short, float, float>;
//template class pfsDRPStella::Spectrum<float, unsigned int, double, double>;
//template class pfsDRPStella::Spectrum<double, unsigned int, double, double>;
//template class pfsDRPStella::Spectrum<float, unsigned short, double, double>;
//template class pfsDRPStella::Spectrum<double, unsigned short, double, double>;

//template class pfsDRPStella::SpectrumSet<float>;
//template class pfsDRPStella::SpectrumSet<double>;
template class pfsDRPStella::SpectrumSet<float, unsigned int, float, float>;
template class pfsDRPStella::SpectrumSet<double, unsigned int, float, float>;
template class pfsDRPStella::SpectrumSet<float, unsigned short, float, float>;
template class pfsDRPStella::SpectrumSet<double, unsigned short, float, float>;
//template class pfsDRPStella::SpectrumSet<float, unsigned int, double, double>;
//template class pfsDRPStella::SpectrumSet<double, unsigned int, double, double>;
//template class pfsDRPStella::SpectrumSet<float, unsigned short, double, double>;
//template class pfsDRPStella::SpectrumSet<double, unsigned short, double, double>;

template PTR(afwImage::MaskedImage<float, unsigned short, float>) pfsDRPStella::utils::getPointer(afwImage::MaskedImage<float, unsigned short, float> &);
template PTR(afwImage::MaskedImage<double, unsigned short, float>) pfsDRPStella::utils::getPointer(afwImage::MaskedImage<double, unsigned short, float> &);
template PTR(std::vector<unsigned short>) pfsDRPStella::utils::getPointer(std::vector<unsigned short> &);
template PTR(std::vector<unsigned int>) pfsDRPStella::utils::getPointer(std::vector<unsigned int> &);
template PTR(std::vector<int>) pfsDRPStella::utils::getPointer(std::vector<int> &);
template PTR(std::vector<float>) pfsDRPStella::utils::getPointer(std::vector<float> &);
template PTR(std::vector<double>) pfsDRPStella::utils::getPointer(std::vector<double> &);
template PTR(pfsDRPStella::Spectrum<float, unsigned short, float, float>) pfsDRPStella::utils::getPointer(pfsDRPStella::Spectrum<float, unsigned short, float, float> &);
template PTR(pfsDRPStella::Spectrum<double, unsigned short, float, float>) pfsDRPStella::utils::getPointer(pfsDRPStella::Spectrum<double, unsigned short, float, float> &);

