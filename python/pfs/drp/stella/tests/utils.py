import sys
import unittest
import functools
from contextlib import contextmanager

"""Decorators for tests

These have been added to LSST, but haven't been included in a release yet.
"""

__all__ = ["classParameters", "methodParameters", "runTests"]


def classParameters(**settings):
    """Class decorator for generating unit tests

    This decorator generates classes with class variables according to the
    supplied ``settings``.

    Parameters
    ----------
    **settings : `dict` (`str`: iterable)
        The class variables to set. Each should be an iterable of the same
        length.

    Example
    -------

        @classParameters(foo=[1, 2], bar=[3, 4])
        class MyTestCase(unittest.TestCase):
            ...

    will generate two classes, as if you wrote:

        class MyTestCase_1_3(unittest.TestCase):
            foo = 1
            bar = 3
            ...

        class MyTestCase_2_4(unittest.TestCase):
            foo = 2
            bar = 4
            ...

    Note that the values are embedded in the class name.
    """
    def decorator(cls):
        num = len(next(iter(settings.values())))
        module = sys.modules[cls.__module__].__dict__
        for name, values in settings.items():
            assert len(values) == num, f"Length mismatch for {name}: {len(values)} vs {num}"
        for ii in range(num):
            values = [settings[kk][ii] for kk in settings]
            name = f"{cls.__name__}_{'_'.join(str(vv) for vv in values)}"
            bindings = dict(cls.__dict__)
            bindings.update(dict(zip(settings.keys(), values)))
            module[name] = type(name, (cls,), bindings)
    return decorator


def methodParameters(**settings):
    """Method decorator for unit tests

    This decorator iterates over the supplied settings, using
    ``TestCase.subTest`` to communicate the values in the event of a failure.

    Parameters
    ----------
    **settings : `dict` (`str`: iterable)
        The parameter combinations to test. Each should be an iterable of the
        same length.

    Example
    -------

        @methodParameters(foo=[1, 2], bar=[3, 4])
        def testSomething(self, foo, bar):
            ...

    will run:

        testSomething(foo=1, bar=3)
        testSomething(foo=2, bar=4)
    """
    num = len(next(iter(settings.values())))
    for name, values in settings.items():
        assert len(values) == num, f"Length mismatch for {name}: {len(values)} vs {num}"

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            for ii in range(num):
                values = [settings[kk][ii] for kk in settings]
                kwargs.update(dict(zip(settings.keys(), values)))
                with self.subTest(**kwargs):
                    func(self, *args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def runTests(_globals={}, profileTop=30):
    """Run tests

    This is intended for use within individual test files:

        if __name__ == "__main__":
            runTests(globals())

    This runs the ``setup_module`` function, if it exists.

    If the ``--display`` command-line argument is provided, the value is put in
    the global ``display`` variable.

    If the ``--profile`` command-line argument is provided, the tests will be
    profiled, and the profile of the top ``profileTop`` function calls will be
    printed at completion.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser(__file__)
    parser.add_argument("--display", help="Display backend")
    parser.add_argument("--profile", action="store_true", help="Profile tests?")
    parser.add_argument("--debug", action="store_true", help="Activate LSST debug mechanism?")
    args, argv = parser.parse_known_args()
    if _globals and args.display:
        _globals["display"] = args.display

    if "setup_module" in _globals:
        _globals["setup_module"](sys.modules["__main__"])

    if args.debug:
        import debug  # noqa F401: activate LSST debug mechanism

    if args.profile:
        import cProfile
        import pstats
        profile = cProfile.Profile()
        profile.enable()

    unittest.main(failfast=True, argv=[__file__] + argv, exit=not args.profile)

    if args.profile:
        profile.disable()
        stats = pstats.Stats(profile)
        stats.sort_stats("cumulative")
        stats.print_stats(profileTop)
