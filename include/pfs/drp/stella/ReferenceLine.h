#if !defined(PFS_DRP_STELLA_REFERENCELINE_H)
#define PFS_DRP_STELLA_REFERENCELINE_H

#include <string>

namespace pfs { namespace drp { namespace stella {
/**
 * \brief Describe a calibration line
 */
struct ReferenceLine {
    enum Status {                       // Line's status
        NOWT=0,
        FIT=1,                          // line was found and fit; n.b. powers of 2
        RESERVED=2,                     // line was not used in estimating rms
        MISIDENTIFIED=4,                // line was misidentified
        CLIPPED=8,                      // line was clipped in fitting distortion
        SATURATED=16,                   // (centre of) line was saturated
        INTERPOLATED=32,                // (centre of) line was interpolated
        CR=64,                          // line is contaminated by a cosmic ray
    };

    ReferenceLine(std::string const& _description, Status _status=NOWT, double _wavelength=0,
                  double _guessedIntensity=0, double _guessedPosition=0,
                  double _fitIntensity=0, double _fitPosition=0, double _fitPositionErr=0
                 ) :
        description(_description),
        status(_status),
        wavelength(_wavelength),
        guessedIntensity(_guessedIntensity),
        guessedPosition(_guessedPosition),
        fitIntensity(_fitIntensity),
        fitPosition(_fitPosition),
        fitPositionErr(_fitPositionErr)
    {}

    std::string description;            // description of line (e.g. Hg[II])
    int status;                         // status of line
    double wavelength;                   // vacuum wavelength, nm
    double guessedIntensity;             // input guess for intensity (amplitude of peak)
    double guessedPosition;              // input guess for pixel position
    double fitIntensity;                 // estimated intensity (amplitude of peak)
    double fitPosition;                  // fit line position
    double fitPositionErr;               // estimated standard deviation of fitPosition
};

}}} // namespace pfs::drp::stella

#endif
