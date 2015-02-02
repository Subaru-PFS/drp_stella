#if !defined(PFS_DRP_STELLA_SPECTRA_H)
#define PFS_DRP_STELLA_SPECTRA_H

#include <vector>
#include <iostream>
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "blitz.h"
#include <fitsio.h>
#include <fitsio2.h>
#include "math/Math.h"
#include "utils/Utils.h"

#define stringify( name ) # name

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;

using namespace std;
namespace pfs { namespace drp { namespace stella {
/**
 * \brief Describe a single fiber trace
 */
template<typename SpectrumT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel>
class Spectrum {
  public:

    // Class Constructors and Destructor
    explicit Spectrum(
      unsigned int length = 0, unsigned int iTrace = 0
    );
    
    explicit Spectrum(
      Spectrum &spectrum,
      unsigned int iTrace = 0
    );
    
    virtual ~Spectrum() {}

    /// Return a shared pointer to the spectrum
    PTR(std::vector<SpectrumT>) getSpectrum() { return _spectrum; }
    const PTR(const std::vector<SpectrumT>) getSpectrum() const { return _spectrum; }
    
    /// Set the spectrum
    /// sets this->_spectrum to spectrum and returns TRUE if spectrum->size() == this->getLength(), otherwise returns false
    /// pre: set length of this to spectrum.size() to adjust length of all vectors in this
    bool setSpectrum(const PTR(std::vector<SpectrumT>) & spectrum);

    /// Return the pointer to the variance of this spectrum
    PTR(std::vector<VarianceT>) getVariance() { return _variance; }
    const PTR(const std::vector<VarianceT>) getVariance() const { return _variance; }

    /// Set the variance pointer of this fiber trace to variance
    /// sets this->_variance to variance and returns TRUE if variance->size() == this->getLength(), otherwise returns false
    bool setVariance(const PTR(std::vector<VarianceT>) & variance);

    /// Return the pointer to the wavelength vector of this spectrum
    PTR(std::vector<WavelengthT>) getWavelength() { return _wavelength; }
    const PTR(const std::vector<WavelengthT>) getWavelength() const { return _wavelength; }

    /// Set the wavelength vector of this spectrum
    /// sets this->_wavelength to wavelength and returns TRUE if wavelength->size() == this->getLength(), otherwise returns false
    bool setWavelength(const PTR(std::vector<WavelengthT>) & wavelength);

    /// Return the pointer to the mask vector of this spectrum
    PTR(std::vector<MaskT>) getMask() { return _mask; }
    const PTR(const std::vector<MaskT>) getMask() const { return _mask; }

    /// Set the mask vector of this spectrum
    /// sets this->_mask to mask and returns TRUE if mask->size() == this->getLength(), otherwise returns false
    bool setMask(const PTR(std::vector<MaskT>) & mask);

    unsigned int getLength() const {return _length;}
    
    /// Resize all vectors to size length.
    /// If length is smaller than the current container size, the contents of all vectors are reduced to their first length elements, 
    /// removing those beyond (and destroying them).
    /// If length is greater than the current container size, the contents of all vectors are expanded by inserting at the end as 
    /// many elements as needed to reach a size of length. The new elements of all vectors except for _wavelength are initialized 
    /// with 0. The new elements of _wavelength are initialized with _wavelength(_length - 1).
    bool setLength(const unsigned int length);
    
    unsigned int getITrace() const {return _iTrace;}
    void setITrace(unsigned int iTrace){_iTrace = iTrace;}
    
    bool isWavelengthSet() const {return _isWavelengthSet;}
//    void setIsWavelengthSet(bool isWavelengthSet) {_isWavelengthSet = isWavelengthSet;}
    
  private:
    unsigned int _length;
    PTR(std::vector<SpectrumT>) _spectrum;
    PTR(std::vector<MaskT>) _mask;/// 0: all pixels of the wavelength element used for extraction were okay
                                  /// 1: at least one pixel was not okay but the extraction algorithm believes it could fix it
                                  /// 2: at least one pixel was problematic
    PTR(std::vector<VarianceT>) _variance;
    PTR(std::vector<WavelengthT>) _wavelength;
    unsigned int _iTrace;/// for logging / debugging purposes only
    bool _isWavelengthSet;

  protected:
};

/************************************************************************************************************/
/**
 * \brief Describe a set of spectra
 *
 */
template<typename SpectrumT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel>
class SpectrumSet {
  public:
    /// Class Constructors and Destructor
      
    /// Creates a new SpectrumSet object of size 0
    explicit SpectrumSet(unsigned int nSpectra=0)
        : _spectra(new std::vector<PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)>(nSpectra))
        {}
        
    /// Copy constructor
    /// If spectrumSet is not empty, the object shares ownership of spectrumSet's spectrum vector and increases the use count.
    /// If spectrumSet is empty, an empty object is constructed (as if default-constructed).
    explicit SpectrumSet(const SpectrumSet &spectrumSet)
        : _spectra(spectrumSet.getSpectra())
        {}

    /// Construct an object with a copy of spectrumVector
    explicit SpectrumSet(const std::vector<PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)> &spectrumVector)
        : _spectra(new std::vector<PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)>(spectrumVector))
        {}
        
    virtual ~SpectrumSet() {}

    /// Return the number of spectra/apertures
    int size() const { return _spectra->size(); }

    /// Return the Spectrum for the ith aperture
    PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) &getSpectrum(const unsigned int i ///< desired aperture
                             );

    PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) const& getSpectrum(const unsigned int i ///< desired aperture
                                   ) const;

    /// Set the ith Spectrum
    bool setSpectrum(const unsigned int i,     /// which spectrum?
                     const PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) & spectrum /// the Spectrum for the ith aperture
                      );

    /// add one Spectrum to the set
    void addSpectrum(const PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) & spectrum /// the Spectrum to add
    );

    PTR(std::vector<PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)>) getSpectra() const { return _spectra; }

    
    /// Removes from the vector either a single element (position) or a range of elements ([first,last)).
    /// This effectively reduces the container size by the number of elements removed, which are destroyed.
    bool erase(const unsigned int iStart, const unsigned int iEnd=0);
    
  private:
    PTR(std::vector<PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)>) _spectra; // spectra for each aperture
};


namespace math{
}

}}}
#endif