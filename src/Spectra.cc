#include "pfs/drp/stella/Spectra.h"

namespace pfsDRPStella = pfs::drp::stella;

/** @brief Construct a Spectrum with empty vectors of specified size (default 0)
 */
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::Spectrum(
  unsigned int length,
  unsigned int iTrace
) : _length(length),
    _spectrum(new std::vector<SpectrumT>(length)),
    _mask(new std::vector<MaskT>(length)),
    _variance(new std::vector<VarianceT>(length)),
    _wavelength(new std::vector<WavelengthT>(length)),
    _iTrace(iTrace),
    _isWavelengthSet(false)
{
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::Spectrum(
  Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> & spectrum,
  unsigned int iTrace
) : _length(spectrum.getLength()),
    _spectrum(spectrum.getSpectrum()),
    _mask(spectrum.getMask()),
    _variance(spectrum.getVariance()),
    _wavelength(spectrum.getWavelength()),
    _iTrace(spectrum.getITrace()),
    _isWavelengthSet(spectrum.isWavelengthSet())
{
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum(const PTR(std::vector<SpectrumT>) & spectrum)
{
  /// Check length of input spectrum
  if (spectrum->size() != _length){
    cout << "pfsDRPStella::Spectrum::setSpectrum: ERROR: spectrum->size()=" << spectrum->size() << " != _length=" << _length << " => Returniing FALSE" << endl;
    return false;
  }
  _spectrum = spectrum;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setVariance(const PTR(std::vector<VarianceT>) & variance)
{
  /// Check length of input variance
  if (variance->size() != _length){
    cout << "pfsDRPStella::Spectrum::setSpectrum: ERROR: variance->size()=" << variance->size() << " != _length=" << _length << " => Returniing FALSE" << endl;
    return false;
  }
  _variance = variance;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setWavelength(const PTR(std::vector<WavelengthT>) & wavelength)
{
  /// Check length of input wavelength
  if (wavelength->size() != _length){
    cout << "pfsDRPStella::Spectrum::setSpectrum: ERROR: wavelength->size()=" << wavelength->size() << " != _length=" << _length << " => Returniing FALSE" << endl;
    return false;
  }
  _wavelength = wavelength;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setMask(const PTR(std::vector<MaskT>) & mask)
{
  /// Check length of input mask
  if (mask->size() != _length){
    cout << "pfsDRPStella::Spectrum::setSpectrum: ERROR: mask->size()=" << mask->size() << " != _length=" << _length << " => Returniing FALSE" << endl;
    return false;
  }
  _mask = mask;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setLength(const unsigned int length){
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
    
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum(const unsigned int i,     /// which spectrum?
                     const PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) & spectrum /// the Spectrum for the ith aperture
                      )
{
  if (i > _spectra->size()){
    cout << "pfsDRPStella::SpectrumSet::setSpectrum: ERROR: i=" << i << " > _spectra->size()=" << _spectra->size() << " => Returning FALSE" << endl;
    return false;
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
void pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::addSpectrum(const PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) & spectrum /// the Spectrum to add
)
{
  try{
    _spectra->push_back(spectrum);
  }
  catch (std::exception &e){
    cout << "SpectrumSet.addSpectrum: Exception <" << e.what() << ">" << endl;
  }
}
  
template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT>)& pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum(const unsigned int i ///< desired aperture
){
  if (i >= _spectra->size()){
    string message("SpectrumSet::getSpectrum(i=");
    message += to_string(i) + string("): ERROR: i > _spectra->size()=") + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  return _spectra->at(i); 
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT>) const& pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum(const unsigned int i ///< desired aperture
) const { 
  if (i >= _spectra->size()){
    string message("SpectrumSet::getSpectrum(i=");
    message += to_string(i) + string("): ERROR: i > _spectra->size()=") + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  return _spectra->at(i); 
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::erase(const unsigned int iStart, const unsigned int iEnd){
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

template class pfsDRPStella::Spectrum<float>;
template class pfsDRPStella::Spectrum<double>;
template class pfsDRPStella::Spectrum<float, unsigned int, float, float>;
template class pfsDRPStella::Spectrum<double, unsigned int, float, float>;

template class pfsDRPStella::SpectrumSet<float>;
template class pfsDRPStella::SpectrumSet<double>;
template class pfsDRPStella::SpectrumSet<float, unsigned int, float, float>;
template class pfsDRPStella::SpectrumSet<double, unsigned int, float, float>;
