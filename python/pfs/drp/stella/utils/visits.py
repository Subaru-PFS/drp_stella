import re
from typing import List

VISIT_REGEX = re.compile(r"^(\d+)\.\.(\d+)(?::(\d+))?$")


def parseIntegerList(string: str) -> List[int]:
    """Parse a string of integers into a list of integers

    This supports the LSST Gen2 Butler syntax: the integers may be separated by
    a caret (``^``) or of the form ``<start>..<stop>`` (e.g. ``"1..3"``) which is
    interpreted as ``"1^2^3"`` (inclusive, unlike a python range). So
    ``"0^2..4^7..9"`` is equivalent to ``"0^2^3^4^7^8^9"``. You may also specify
    a stride using the form ``<start>..<stop>:<stride>``, e.g., ``"1..5:2"`` is
    ``"1^3^5"``.

    This is adapted from the LSST Gen2 middleware's
    ``lsst.pipe.base.argumentParser.IdValueAction``.

    Parameters
    ----------
    string : `str`
        String to parse.

    Returns
    -------
    values : `list` of `int`
        List of integers.
    """
    result = []
    for value in string.split("^"):
        mat = re.search(VISIT_REGEX, value)
        if mat:
            start = int(mat.group(1))
            stop = int(mat.group(2))
            stride = mat.group(3)
            stride = int(stride) if stride else 1
            result += list(range(start, stop + 1, stride))
        else:
            result.append(int(value))
    return result
