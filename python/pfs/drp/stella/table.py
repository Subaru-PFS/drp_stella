from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, Iterator, List, Protocol, Type, TypeVar, Union, overload

import numpy as np
from pandas import DataFrame, concat
from pfs.datamodel import PfsTable, POD

__all__ = ("Table",)


class Row(Protocol):
    """Description of what table `Row` objects can do

    These are essentially structs, holding and providing data.
    """

    def __init__(self, **kwargs: POD):
        pass

    def __getattr__(self, name: str) -> POD:
        pass


SubTable = TypeVar("SubTable", bound="Table")
"""A generic sub-class of Table"""


class Table:
    """Base class for tables

    This is a more useful representation (backed by a `pandas.DataFrame`) for
    manipulation of tables than the datamodel representation (base class
    `pfs.datamodel.PfsTable`, backed by `numpy.ndarray`) which is used solely
    for I/O.

    Subclasses should define the class variable ``DamdClass`` (`type`): a
    subclass of `pfs.datamodel.PfsTable` that contains the ``schema`` and
    implements the I/O.

    Subclasses will have properties (returning arrays) defined for each of the
    fields in the schema.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Data for table.
    """

    DamdClass: ClassVar[PfsTable]
    """Sub-class of `PfsTable` that defines the schema and implements the I/O"""

    RowClass: ClassVar[Type[Row]]
    """Class used for representing rows

    This is created dynamically from the ``DamdClass``'s schema.
    """

    _schema: ClassVar[Dict[str, type]]
    """Copy of the ``DamdClass``'s ``schema``"""

    def __init__(self, data: DataFrame):
        self.data = data

    def __init_subclass__(cls: Type["Table"]):
        """Initialise sub-class

        Sets up the properties based on the schema.
        """
        cls._schema = cls.DamdClass.getSchemaDict()
        cls.RowClass = dataclass(type(cls.__name__ + "Row", (object,), dict(__annotations__=cls._schema)))
        for name in cls._schema:

            def getter(self, name: str = name) -> np.ndarray:
                return self.data[name].values

            setattr(cls, name, property(getter))

    def __getattr__(self, name: str) -> np.ndarray:
        """Get column

        This method mostly exists for the benefit of type checkers, as it sets
        the type for the columns.
        """
        if name in self._schema:
            return self.data[name].values
        raise AttributeError(f"Unknown attribute: {name}")

    @classmethod
    def fromRows(cls: Type[SubTable], rows: Iterable[Row]) -> SubTable:
        """Construct from a list of rows

        Parameters
        ----------
        rows : iterable of ``RowClass``
            Rows of table. Each row must contain at least the attributes in the
            ``schema``.
        """
        return cls.fromColumns(
            **{
                name: np.array([getattr(ll, name) for ll in rows], dtype=dtype.dtype)
                for name, dtype in cls._schema.items()
            }
        )

    @classmethod
    def fromColumns(cls: Type[SubTable], **columns) -> SubTable:
        """Construct from columns

        Parameters
        ----------
        **columns
            Columns of table. Must include at least the names in the ``schema``,
            and each column must be ``array_like``.
        """
        return cls(
            DataFrame(
                {name: np.array(columns[name], dtype=dtype.dtype) for name, dtype in cls._schema.items()}
            )
        )

    @classmethod
    def fromMultiple(cls: Type[SubTable], *tables: "Table") -> SubTable:
        """Construct from multiple tables

        Parameters
        ----------
        *tables : `Table`
            Tables to concatenate.
        """
        return cls(concat([tt.data for tt in tables]))

    def toDataFrame(self) -> DataFrame:
        """Convert to a `pandas.DataFrame`"""
        return self.data

    @property
    def rows(self) -> List[Row]:
        """Return list of rows

        This is convenient, but can be slow since it creates a new `Row` object
        for each row.
        """
        return [self.RowClass(**row[1].to_dict()) for row in self.data.iterrows()]

    def __len__(self) -> int:
        """Number of lines"""
        return len(self.data)

    def __bool__(self) -> bool:
        """Is non-empty?"""
        return len(self) != 0

    def __iter__(self) -> Iterator[Row]:
        """Iterator

        This is convenient, but can be slow since it creates a new `Row` object
        for each row.
        """
        return iter(self.rows)

    @overload
    def __getitem__(self, index: int) -> Row:
        ...

    @overload
    def __getitem__(self, index: Union[slice, np.ndarray]) -> "Table":
        ...

    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> Union[Row, "Table"]:
        """Retrieve row(s)"""
        if isinstance(index, int):
            return self.RowClass(**self.data.iloc[index].to_dict())
        if isinstance(index, (slice, np.ndarray)):
            return type(self)(self.data.iloc[index])
        raise RuntimeError(f"Cannot interpret index: {index}")

    def extend(self, other: "Table"):
        """Extend the table

        This is an inefficient way of populating a table.

        Parameters
        ----------
        other : `Table`
            Table to use to extend this table.
        """
        self.data = concat((self.data, other.data))

    def __add__(self, rhs: "Table") -> "Table":
        """Addition

        This is an inefficient way of populating a table.
        """
        return type(self)(concat((self.data, rhs.data)))

    def __iadd__(self, rhs: "Table") -> "Table":
        """In-place addition

        This is an inefficient way of populating a table.
        """
        self.extend(rhs)
        return self

    def copy(self) -> "Table":
        """Return a deep copy"""
        return type(self)(self.data.copy())

    @classmethod
    def empty(cls: Type[SubTable]) -> SubTable:
        """Construct an empty table"""
        return cls.fromColumns(**{name: [] for name, _ in cls._schema.items()})

    @classmethod
    def fromPfsTable(cls: Type[SubTable], table: PfsTable) -> SubTable:
        """Construct from `PfsTable`

        Parameters
        ----------
        table : `PfsTable`
            Datamodel representation of table.

        Returns
        -------
        self : cls
            Constructed table.
        """
        return cls.fromColumns(**table.columns)

    def toPfsTable(self) -> PfsTable:
        """Convert to `PfsTable`

        Returns
        -------
        table : `PfsTable`
            Datamodel representation of table.
        """
        return self.DamdClass(**{name: getattr(self, name) for name in self._schema})

    @classmethod
    def readFits(cls: Type[SubTable], filename: str) -> SubTable:
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
        return cls.fromPfsTable(cls.DamdClass.readFits(filename))

    def writeFits(self, filename: str):
        """Write to file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        self.toPfsTable().writeFits(filename)
