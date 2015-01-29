#include "pfs/drp/stella/Spectra.h"

namespace pfsDRPStella = pfs::drp::stella;

/** @brief Construct a Spectrum with empty vectors of specified size (default 0)
 */
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::Spectrum(size_t length,
                                                                           size_t iTrace) 
  : _length(length),
    _spectrum(new std::vector<SpectrumT>(length)),
    _mask(new std::vector<MaskT>(length)),
    _variance(new std::vector<VarianceT>(length)),
    _wavelength(new std::vector<WavelengthT>(length)),
    _iTrace(iTrace),
    _isWavelengthSet(false)
{
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::Spectrum(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> & spectrum,
                                                                           size_t iTrace) 
  : _length(spectrum.getLength()),
    _spectrum(spectrum.getSpectrum()),
    _mask(spectrum.getMask()),
    _variance(spectrum.getVariance()),
    _wavelength(spectrum.getWavelength()),
    _iTrace(spectrum.getITrace()),
    _isWavelengthSet(spectrum.isWavelengthSet())
{
  if (iTrace != 0)
    _iTrace = iTrace;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum(const PTR(std::vector<SpectrumT>) & spectrum)
{
  /// Check length of input spectrum
  if (spectrum->size() != _length){
    string message("pfsDRPStella::Spectrum::setSpectrum: ERROR: spectrum->size()=");
    message += to_string(spectrum->size()) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _spectrum = spectrum;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setVariance(const PTR(std::vector<VarianceT>) & variance)
{
  /// Check length of input variance
  if (variance->size() != _length){
    string message("pfsDRPStella::Spectrum::setVariance: ERROR: variance->size()=");
    message += to_string(variance->size()) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _variance = variance;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setWavelength(const PTR(std::vector<WavelengthT>) & wavelength)
{
  /// Check length of input wavelength
  if (wavelength->size() != _length){
    string message("pfsDRPStella::Spectrum::setWavelength: ERROR: wavelength->size()=");
    message += to_string(wavelength->size()) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _wavelength = wavelength;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setMask(const PTR(std::vector<MaskT>) & mask)
{
  /// Check length of input mask
  if (mask->size() != _length){
    string message("pfsDRPStella::Spectrum::setMask: ERROR: mask->size()=");
    message += to_string(mask->size()) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _mask = mask;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setLength(const size_t length){
  if (length < _length){
    _spectrum->resize(length);
    _mask->resize(length);
    _variance->resize(length);
    _wavelength->resize(length);
  }
  else{
    _spectrum->resize(length, 0);
    _mask->resize(length, 0);
    _variance->resize(length, 0);
    _wavelength->resize(length, (*_wavelength)[_length - 1]);
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
    string message("pfsDRPStella::SpectrumSet::setSpectrum: ERROR: i=");
    message += to_string(i) + string(" > _spectra->size()=") + to_string(_spectra->size());
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
bool pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::addSpectrum(const PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) & spectrum /// the Spectrum to add
)
{
  try{
    _spectra->push_back(spectrum);
  }
  catch (std::exception &e){
    string message("SpectrumSet.addSpectrum: Exception <");
    message += e.what() + string(">");
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  return true;
}
  
template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT>)& pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum(const size_t i){
  if (i >= _spectra->size()){
    string message("SpectrumSet::getSpectrum(i=");
    message += to_string(i) + string("): ERROR: i > _spectra->size()=") + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  return _spectra->at(i); 
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT>) const& pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum(const size_t i) const { 
  if (i >= _spectra->size()){
    string message("SpectrumSet::getSpectrum(i=");
    message += to_string(i) + string("): ERROR: i > _spectra->size()=") + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  return _spectra->at(i); 
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::erase(const size_t iStart, const size_t iEnd){
  if (iStart >= _spectra->size()){
    string message("SpectrumSet::erase(iStart=");
    message += to_string(iStart) + string("): ERROR: iStart >= _spectra->size()=") + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (iEnd >= _spectra->size()){
    string message("SpectrumSet::erase(iEnd=");
    message += to_string(iEnd) + string("): ERROR: iEnd >= _spectra->size()=") + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
    if ((iEnd > 0) && (iStart > iEnd)){
      string message("SpectrumSet::erase(iStart=");
      message += to_string(iStart) + string("): ERROR: iStart > iEnd=") + to_string(iEnd);
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
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

//template class pfsDRPStella::SpectrumSet<float>;
//template class pfsDRPStella::SpectrumSet<double>;
template class pfsDRPStella::SpectrumSet<float, unsigned int, float, float>;
template class pfsDRPStella::SpectrumSet<double, unsigned int, float, float>;
template class pfsDRPStella::SpectrumSet<float, unsigned short, float, float>;
template class pfsDRPStella::SpectrumSet<double, unsigned short, float, float>;

template PTR(afwImage::MaskedImage<float, unsigned short, float>) pfsDRPStella::utils::getPointer(afwImage::MaskedImage<float, unsigned short, float> &);
template PTR(afwImage::MaskedImage<double, unsigned short, float>) pfsDRPStella::utils::getPointer(afwImage::MaskedImage<double, unsigned short, float> &);
template PTR(std::vector<unsigned short>) pfsDRPStella::utils::getPointer(std::vector<unsigned short> &);
template PTR(std::vector<unsigned int>) pfsDRPStella::utils::getPointer(std::vector<unsigned int> &);
template PTR(std::vector<int>) pfsDRPStella::utils::getPointer(std::vector<int> &);
template PTR(std::vector<float>) pfsDRPStella::utils::getPointer(std::vector<float> &);
template PTR(std::vector<double>) pfsDRPStella::utils::getPointer(std::vector<double> &);
template PTR(pfsDRPStella::Spectrum<float, unsigned short, float, float>) pfsDRPStella::utils::getPointer(pfsDRPStella::Spectrum<float, unsigned short, float, float> &);
template PTR(pfsDRPStella::Spectrum<double, unsigned short, float, float>) pfsDRPStella::utils::getPointer(pfsDRPStella::Spectrum<double, unsigned short, float, float> &);

