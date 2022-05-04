from dataclasses import is_dataclass
from typing import Iterable, Tuple, Type, TypeVar, Dict, List, Iterator, Union, Any, Protocol

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame, concat

import astropy.io.fits


class Row(Protocol):
    """How to identify a dataclass that serves as a row of the table

    This exists solely for the sake of type hints.

    TableBase only works if the ``RowClass`` is a dataclass.
    """
    __dataclass_fields__: Dict[str, Any]


Table = TypeVar("Table", bound="TableBase")


class TableBase:
    """Base class for tables

    Subclasses should define the class variables:
    * ``RowClass`` (`type`): a `dataclass` that will serve as the container for
      a single row. This defines the schema of the table.
    * ``fitsExtName`` (`str`): FITS extension name.
    * ``damdver`` (`int`; optional): Datamodel version, written to ``DAMD_VER``
      in the FITS header.

    The ``_schema`` class variable is set automatically.

    Subclasses will have properties (returning arrays) defined for each of the
    dataclass fields in the ``RowClass``.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Data for table.
    """
    RowClass: Type[Row]
    fitsExtName: str
    damdver: int = 1
    _schema: Iterable[Tuple[str, type]]

    def __init__(self, data: DataFrame):
        self.data = data

    def __init_subclass__(cls: Type[Table]):
        """Initialise sub-class

        This is where the magic happens! This is called when you subclass from
        ``TableBase``.

        Sets the ``_schema`` class variable, so all the inherited methods know
        what they're working with, and sets up the properties.
        """
        def getSchema(cls: Type[Row]) -> Dict[str, type]:
            schema = {}
            for bb in reversed(cls.__bases__):
                schema.update(getSchema(bb))
            if is_dataclass(cls):
                schema.update({ff.name: ff.type for ff in cls.__dataclass_fields__.values()})
            return schema

        schema = getSchema(cls.RowClass)
        setattr(cls, "_schema", tuple(schema.items()))
        for name in schema:
            def func(self, name: str = name) -> ArrayLike:
                """Return column array"""
                return self.data[name].values

            setattr(cls, name, property(func))

    @classmethod
    def fromRows(cls: Type[Table], rows: Iterable[Row]) -> Table:
        """Construct from a list of rows

        Parameters
        ----------
        rows : iterable of ``RowClass``
            Rows of table. Each row must contain at least the attributes in the
            ``schema``.
        """
        return cls.fromColumns(**{name: np.array([getattr(ll, name) for ll in rows], dtype=dtype) for
                                  name, dtype in cls._schema})

    @classmethod
    def fromColumns(cls: Type[Table], **columns) -> Table:
        """Construct from columns

        Parameters
        ----------
        **columns
            Columns of table. Must include at least the names in the ``schema``,
            and each column must be ``array_like``.
        """
        return cls(DataFrame({name: np.array(columns[name], dtype=dtype) for name, dtype in cls._schema}))

    def toDataFrame(self) -> DataFrame:
        """Convert to a `pandas.DataFrame`"""
        return self.data

    @property
    def rows(self) -> List[Any]:
        """Return list of rows"""
        return [self.RowClass(**row[1].to_dict()) for row in self.data.iterrows()]

    def __len__(self) -> int:
        """Number of lines"""
        return len(self.data)

    def __bool__(self) -> bool:
        """Is non-empty?"""
        return len(self) != 0

    def __iter__(self) -> Iterator[Any]:
        """Iterator"""
        return iter(self.rows)

    def __getitem__(self: Table, index: Union[int, slice, np.ndarray]) -> Union[Any, Table]:
        """Retrieve row(s)"""
        if isinstance(index, int):
            return self.RowClass(**self.data.iloc[index].to_dict())
        if isinstance(index, (slice, np.ndarray)):
            return type(self)(self.data.iloc[index])
        raise RuntimeError(f"Cannot interpret index: {index}")

    def extend(self, other: Table):
        """Extend the table

        This is an inefficient way of populating a table.

        Parameters
        ----------
        other : `TableBase`
            Table to use to extend this table.
        """
        self.data = concat((self.data, other.data))

    def __add__(self: Table, rhs: Table) -> Table:
        """Addition

        This is an inefficient way of populating a table.
        """
        return type(self)(concat((self.data, rhs.data)))

    def __iadd__(self: Table, rhs: Table) -> Table:
        """In-place addition

        This is an inefficient way of populating a table.
        """
        self.extend(rhs)
        return self

    def copy(self: Table) -> Table:
        """Return a deep copy"""
        return type(self)(self.data.copy())

    @classmethod
    def empty(cls: Type[Table]) -> Table:
        """Construct an empty table"""
        return cls.fromColumns(**{name: [] for name, _ in cls._schema})

    @classmethod
    def readFits(cls: Type[Table], filename: str) -> Table:
        """Read from file

        Parameters
        ----------
        filename : `str`
            Name of file from which to read.

        Returns
        -------
        self : cls
            Constructed object from reading file.
        """
        data = {}
        with astropy.io.fits.open(filename) as fits:
            hdu = fits[cls.fitsExtName]
            data = {name: hdu.data[name].astype(dtype) for name, dtype in cls._schema}
        return cls(DataFrame(data))

    def writeFits(self, filename: str):
        """Write to file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value.
        format = {int: "K", float: "D", np.int32: "J", np.float32: "E", bool: "L", np.uint8: "B",
                  np.int16: "I", np.int64: "K", np.float64: "D"}

        def getFormat(name, dtype):
            """Determine suitable FITS column format

            This is a simple mapping except for string types.

            Parameters
            ----------
            name : `str`
                Column name, so we can get the data if we need to inspect it.
            dtype : `type`
                Data type.

            Returns
            -------
            format : `str`
                FITS column format string.
            """
            if issubclass(dtype, str):
                length = max(len(ss) for ss in getattr(self, name)) if len(self) > 0 else 0
                length = max(1, length)  # Minimum length of 1 makes astropy happy
                return f"{length}A"
            return format[dtype]

        columns = [astropy.io.fits.Column(name=name, format=getFormat(name, dtype), array=getattr(self, name))
                   for name, dtype in self._schema]
        hdu = astropy.io.fits.BinTableHDU.from_columns(columns, name=self.fitsExtName)

        hdu.header["INHERIT"] = True
        hdu.header["DAMD_VER"] = (self.damdver, f"{self.__class__.__name__} datamodel version")

        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(), hdu])
        with open(filename, "wb") as fd:
            fits.writeto(fd)
