""""

Examples Modules Heading 1
==========================

This module contains examples, none of the functions, classes or other attribute
should have any effects when called. Paragraphs should be able to contain plain
`links <https://github.com/jupyter/papyri>`__, or link via directives: :any:`papyri`

Heading 2
---------

Heading 3
~~~~~~~~~

heading 4
.........


FieldList
---------

In my understnading FieldList should in general not happen in the final
Documents, they are supposed to be processed by the software reading them, and
typically appear at the beginning of directive.


:fieldlistitem:
    at least that the name tree sitter rst give it.
:another: is a another fieldlistItem in this field list
    and it's description


Enumeration and unordered lists
-------------------------------

Enumeration list.

1. item 1
2. item 2
3. item 3

Unordered list

- item 1
- item 2
- item 3


Definition Lists
----------------


term 1
    Definition 1.

term 2
    Definition 2, paragraph 1.

    Definition 2, paragraph 2.


Admonitions
-----------

.. note::

    An admonition note !

This is a link to `Jupyter website <jupyter.org>`__

Code (title 2)
--------------

.. code::

    from package import module
    module.call("string")

Math
----

Math shoudl both work as standalone formulas that takes a block:


.. math::

    \\pi = 3.14159

More complex formula:

.. math::

    \\int_0^\\infty f(\\delta v) \\partial v

.. math::

    \\nabla^2X = \\vec {rot} (\\vec {rot}(X)) = \\vec{grad}(div(X))  - \\Delta X


But should be also available in inline context, for example we know that
:math:`\\pi^2 \\simeq 10`, and that :math:`1+1=10` when counting in binary.

Literal Blocks (verbatim)
-------------------------

::

    This should be a verb-
    -batim block


This is a Transition:

-----

This is q block quote, to do, we know that Attributions are not supported right now.

    "We have weaponed the Tab Key"



Substitutions
~~~~~~~~~~~~~

In this paragraph: |SubstitutionRef| Should be replaced...

.. |SubstitutionDef| replace:: ASUBSTITUTIONDEF

"""

from typing import Optional, Union


async def example1(
    pos: int,
    only: None,
    /,
    var: Union[float, bool],
    args=1,
    *,
    kwarg,
    also=None,
    **kwargs,
) -> Optional[str]:
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
    return "ok"


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


def example_3():
    """
    This show that we support *emph* and **strong**.

    This will be ``verbatim``, and this is `interpreted-text`.
    Interpreted text can have domain and role ``:domain:role:`interpreted text```.

        .. directive::

            This will be a directive

    .. math::

        This will be interpreted at math.

    .. note::

        This directive will be turned into a note admonition.

    .. warning::

        This directive will be turned into a warning admonition.

    """
