"""Monkey-patch astropy to remove thread locks from lazyproperty.

astropy.utils.decorators.lazyproperty is used in a variety of places in
astropy, most notably for PFS in reading FITS tables. In astropy 4.3 (and
continuing to at least 5.1), the implementation of lazyproperty contains thread
locks which can break multiprocessing (under certain conditions). The problem is
that the fork used to generate new processes copies the lock wholesale (see,
e.g., https://stackoverflow.com/a/45889668/834250), and it becomes possible to
deadlock.

Here, we monkey-patch astropy to use a pre-4.3 version of lazyproperty, to avoid
the risk of deadlocking.

Bruce Merry (author of the astropy thread locking code) has pointed out that
there are intrinsic problems with multiprocessing using the "fork" start method
and maxtasksperchild=1, and I believe we're running afoul of those for a reason
I haven't been able to discern. I've not been able to get the "forkserver" or
"spawn" start methods to work reliably in order to work around this problem that
way. But as far as we know, this is a Gen2-only problem, and Gen2 is dying, so
I'm putting in the only workaround that I've been able to get to work, which is
stripping the locks out of lazyproperty. We should be able to remove it once we
burn Gen2 with fire.
"""

import astropy.utils.decorators
import astropy.units.quantity_helper.converters

_NotFound = astropy.utils.decorators._NotFound


class lazyproperty(property):  # noqa N801: name follows astropy
    """
    Works similarly to property(), but computes the value only once.

    This essentially memorizes the value of the property by storing the result
    of its computation in the ``__dict__`` of the object instance.  This is
    useful for computing the value of some property that should otherwise be
    invariant.  For example::

        >>> class LazyTest:
        ...     @lazyproperty
        ...     def complicated_property(self):
        ...         print('Computing the value for complicated_property...')
        ...         return 42
        ...
        >>> lt = LazyTest()
        >>> lt.complicated_property
        Computing the value for complicated_property...
        42
        >>> lt.complicated_property
        42

    As the example shows, the second time ``complicated_property`` is accessed,
    the ``print`` statement is not executed.  Only the return value from the
    first access off ``complicated_property`` is returned.

    By default, a setter and deleter are used which simply overwrite and
    delete, respectively, the value stored in ``__dict__``. Any user-specified
    setter or deleter is executed before executing these default actions.
    The one exception is that the default setter is not run if the user setter
    already sets the new value in ``__dict__`` and returns that value and the
    returned value is not ``None``.

    """

    def __init__(self, fget, fset=None, fdel=None, doc=None):
        super().__init__(fget, fset, fdel, doc)
        self._key = self.fget.__name__

    def __get__(self, obj, owner=None):
        try:
            obj_dict = obj.__dict__
            val = obj_dict.get(self._key, _NotFound)
            if val is _NotFound:
                val = self.fget(obj)
                obj_dict[self._key] = val
            return val
        except AttributeError:
            if obj is None:
                return self
            raise

    def __set__(self, obj, val):
        obj_dict = obj.__dict__
        if self.fset:
            ret = self.fset(obj, val)
            if ret is not None and obj_dict.get(self._key) is ret:
                # By returning the value set the setter signals that it
                # took over setting the value in obj.__dict__; this
                # mechanism allows it to override the input value
                return
        obj_dict[self._key] = val

    def __delete__(self, obj):
        if self.fdel:
            self.fdel(obj)
        obj.__dict__.pop(self._key, None)  # Delete if present

    def __reduce__(self):
        return type(self), (self.fget, self.fset, self.fdel, self.doc)


class UfuncHelpers(dict):
    """Registry of unit conversion functions to help ufunc evaluation.

    Based on dict for quick access, but with a missing method to load
    helpers for additional modules such as scipy.special and erfa.

    Such modules should be registered using ``register_module``.
    """

    def __init__(self, *args, **kwargs):
        self.modules = {}
        self.UNSUPPORTED = set()  # Upper-case for backwards compatibility
        super().__init__(*args, **kwargs)

    @classmethod
    def copy(cls, other):
        """Copy the helpers from another UfuncHelpers instance."""
        new = cls()
        for key, val in other.items():
            new[key] = val
        new.modules = other.modules.copy()
        new.UNSUPPORTED = other.UNSUPPORTED.copy()
        return new

    def register_module(self, module, names, importer):
        """Register (but do not import) a set of ufunc helpers.

        Parameters
        ----------
        module : str
            Name of the module with the ufuncs (e.g., 'scipy.special').
        names : iterable of str
            Names of the module ufuncs for which helpers are available.
        importer : callable
            Function that imports the ufuncs and returns a dict of helpers
            keyed by those ufuncs.  If the value is `None`, the ufunc is
            explicitly *not* supported.
        """
        self.modules[module] = {"names": names, "importer": importer}

    def import_module(self, module):
        """Import the helpers from the given module using its helper function.

        Parameters
        ----------
        module : str
            Name of the module. Has to have been registered beforehand.
        """
        module_info = self.modules.pop(module)
        self.update(module_info["importer"]())

    def __missing__(self, ufunc):
        """Called if a ufunc is not found.

        Check if the ufunc is in any of the available modules, and, if so,
        import the helpers for that module.
        """
        if ufunc in self.UNSUPPORTED:
            raise TypeError(f"Cannot use ufunc '{ufunc.__name__}' with quantities")

        for module, module_info in list(self.modules.items()):
            if ufunc.__name__ in module_info["names"]:
                # A ufunc with the same name is supported by this module.
                # Of course, this doesn't necessarily mean it is the
                # right module. So, we try let the importer do its work.
                # If it fails (e.g., for `scipy.special`), then that's
                # fine, just raise the TypeError.  If it succeeds, but
                # the ufunc is not found, that is also fine: we will
                # enter __missing__ again and either find another
                # module or get the TypeError there.
                try:
                    self.import_module(module)
                except ImportError:  # pragma: no cover
                    pass
                else:
                    return self[ufunc]

        raise TypeError(
            f"unknown ufunc {ufunc.__name__}.  If you believe this ufunc "
            "should be supported, please raise an issue on "
            "https://github.com/astropy/astropy"
        )

    def __setitem__(self, key, value):
        # Implementation note: in principle, we could just let `None`
        # mean that something is not implemented, but this means an
        # extra if clause for the output, slowing down the common
        # path where a ufunc is supported.
        if value is None:
            self.UNSUPPORTED |= {key}
            self.pop(key, None)
        else:
            super().__setitem__(key, value)
            self.UNSUPPORTED -= {key}


# Monkey-patch astropy
astropy.utils.decorators.lazyproperty = lazyproperty
astropy.utils.lazyproperty = lazyproperty

UFUNC_HELPERS = UfuncHelpers.copy(
    astropy.units.quantity_helper.converters.UFUNC_HELPERS
)
astropy.units.quantity_helper.converters.UfuncHelpers = UfuncHelpers
astropy.units.quantity_helper.converters.UFUNC_HELPERS = UFUNC_HELPERS
astropy.units.quantity_helper.UFUNC_HELPERS = UFUNC_HELPERS
