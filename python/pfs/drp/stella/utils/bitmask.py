from enum import IntFlag
import collections

__all__ = ("Bitmask",)


class Bitmask(IntFlag):
    """A bitmask enumerated type with documented members

    From https://stackoverflow.com/a/50473952/834250

    Example:

        class BurgerIngredients(Bitmask):
            BUN = 0x01, "Bread bun"
            PATTY = 0x02, "Hamburger patty"
            LETTUCE = 0x04, "Green healthy stuff"
            TOMATO = 0x08, "Red healthy stuff"
            PICKLES = 0x10, "You either love it or hate it"
    """
    def __new__(cls, value, doc):
        self = int.__new__(cls, value)
        self._value_ = value
        self.__doc__ = doc
        return self

    @classmethod
    def fromNames(cls, *names):
        """Construct from names of enumerated types

        Parameters
        ----------
        *names : `str`
            Names of enumerated types.

        Returns
        -------
        value : cls
            Value from the OR of all provided names.
        """
        if (len(names) == 1 and isinstance(names[0], collections.abc.Sequence) and
                not isinstance(names[0], str)):
            names = names[0]
        value = cls(0)
        for nn in names:
            value |= getattr(cls, nn)
        return value
