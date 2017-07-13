#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "pfs/drp/stella/Lines.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

void declareNistLine(py::module &mod)
{
    py::class_<NistLine, std::shared_ptr<NistLine>> cls(mod, "NistLine");
    cls.def(py::init<>());
    cls.def(py::init<NistLine const&>());
    cls.def_readwrite("element", &NistLine::element);
    cls.def_readwrite("flags", &NistLine::flags);
    cls.def_readwrite("id", &NistLine::id);
    cls.def_readwrite("ion", &NistLine::ion);
    cls.def_readwrite("laboratoryWavelength", &NistLine::laboratoryWavelength);
    cls.def_readwrite("predictedStrength", &NistLine::predictedStrength);
    cls.def_readwrite("sources", &NistLine::sources);
    cls.def("getPointer", &NistLine::getPointer);
}

void declareNistLineMeas(py::module &mod)
{
    py::class_<NistLineMeas, std::shared_ptr<NistLineMeas>> cls(mod, "NistLineMeas");
    cls.def(py::init<>());
    cls.def(py::init<NistLineMeas const&>());
    cls.def_readwrite("eGaussCoeffsLambda", &NistLineMeas::eGaussCoeffsLambda);
    cls.def_readwrite("eGaussCoeffsPixel", &NistLineMeas::eGaussCoeffsPixel);
    cls.def_readwrite("gaussCoeffsLambda", &NistLineMeas::gaussCoeffsLambda);
    cls.def_readwrite("gaussCoeffsPixel", &NistLineMeas::gaussCoeffsPixel);
    cls.def_readwrite("flags", &NistLineMeas::flags);
    cls.def_readwrite("nistLine", &NistLineMeas::nistLine);
    cls.def_readwrite("pixelPosPredicted", &NistLineMeas::pixelPosPredicted);
    cls.def_readwrite("wavelengthFromPixelPosAndPoly", &NistLineMeas::wavelengthFromPixelPosAndPoly);
    cls.def("setNistLine", &NistLineMeas::setNistLine);
    cls.def("getPointer", &NistLineMeas::getPointer);
}

PYBIND11_PLUGIN(lines) {
    py::module mod("lines");

    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };

    declareNistLine(mod);
    declareNistLineMeas(mod);

    mod.def("getNistLineMeasVec", &getNistLineMeasVec);
    mod.def("getNistLineVec", &getNistLineVec);
//    mod.def("createLineListFromWLenPix", (std::vector<PTR(NistLineMeas)>(*)(ndarray::Array<float, 2, 1> const&))&createLineListFromWLenPix, "linesIn"_a);
    mod.def("createLineListFromWLenPix", &createLineListFromWLenPix);
    mod.def("getIndexOfLineWithID", &getIndexOfLineWithID, "lines"_a, "id"_a);
    mod.def("getLinesWithFlags", &getLinesWithFlags, "lines"_a, "needFlags"_a, "ignoreFlags"_a = "");
    mod.def("getLinesWithID", &getLinesWithID, "lines"_a, "id"_a);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
