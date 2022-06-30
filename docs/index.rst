.. _index:
   
Papyri
======


.. toctree::

   crossrefs

Welcome to Papyri narrative docs. 

This show that papyri can support narrative docs as well as API docs, 
though for the time being API docs are better supported. 

If you find issues feel free to `Open and issue on GitHub <https://github.com/jupyter/papyri/>`__.

If you are working only on API docs, you can use ``papyri gen <conffile.toml> --no-api``, to skip the API generation
part. Though be aware that some of the cross linking features will not work. 

.. _examples:

Examples
--------

These are examples of various RST constructs, mostly as a visual test they render
correctly (or not yet). 


Lists
-----

Enumerated lists
~~~~~~~~~~~~~~~~

1. item 1
2. item 2
2. item 3



Bullet Lists
~~~~~~~~~~~~

- Item 
- other item
- another one


Headers
-------

Header 3
~~~~~~~~

Header 4
^^^^^^^^

Header 5
````````


Directives
-----------

Note that directive are not a thing in themselves, and convert to other kind of nodes, 
when they are not understood they are currently rendered raw, but will be an error



.. warning:: no title

   This is a warning

.. versionadded:: 1.2.3

   This was added in version 1.2.3

.. versionchanged:: 1.2.3

   This was changed in version 1.2.3

   You can now do::

      >>> This

   Also:

      - That
      - Or this
      - Or that. 

      

.. currentmodule:: not handled

.. seealso:: not handled

.. highlight:: not handled

.. code::

   Code block (to be done)


This will be math

   
.. math::

   \int_0^\infty f(x) dx

