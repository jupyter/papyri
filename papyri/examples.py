""""

Examples Modules
----------------

This module contains examples, none of the functions, classes or other attribute
should have any effects when called.

"""


def example1():
    """
    first example.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    ... fig, ax = plt.subplots()
    ... ax.plot(range(10))

    And now

    >>> ax.plot(range(5, 15))
    ... plt.show()

    """


def example2():
    """
    Examples
    --------
    >>> import numpy as np
    ... import matplotlib.pyplot as plt
    ... from scipy.interpolate import UnivariateSpline

    >>> x = np.linspace(-3, 3, 50)
    ... y = np.exp(-(x ** 2)) + 0.1 * np.random.randn(50)
    ... plt.plot(x, y, "ro", ms=5)

    Use the default value for the smoothing parameter:

    >>> spl = UnivariateSpline(x, y)
    ... xs = np.linspace(-3, 3, 1000)
    ... plt.plot(xs, spl(xs), "g", lw=3)

    Manually change the amount of smoothing:

    >>> spl.set_smoothing_factor(0.5)
    ... plt.plot(xs, spl(xs), "b", lw=3)
    ... plt.show()

    """
    pass


# TODO: uncomment once implemented<D-{>
# def transition_block_Not_implemented():
#     """
#     Tree sitter supports RST transitions with multiple dashes,
#
#     we don't handle them yet:
#
#
#     -----
#
#
#     This will fail
#
#     """


def dummy():
    """
    I have no docs.
    """
    pass


def test_duplicate_see_also():
    """
    There are cases where see also has two identical elements

    See https://github.com/numpy/numpy/pull/19929

    See Also
    --------
    dummy, dummy

    """
    pass
