#ifndef __PFS_DRP_STELLA_UTILS_H__
#define __PFS_DRP_STELLA_UTILS_H__

namespace pfs {
namespace drp {
namespace stella {
namespace utils{

/*
 * @brief: Test functionality of PolyFit
 * We can't include the tests in Python as the keyword arguments (vector of void pointers)
 * does not get swigged.
 */
void testPolyFit();


}}}} // namespace pfs::drp::stella::utils
#endif
