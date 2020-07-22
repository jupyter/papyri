# Papyri

See the legendary [Villa of Papyri](https://en.wikipedia.org/wiki/Villa_of_the_Papyri), who get its name from it's
collection of many papyrus scrolls.

## What 

A set of tools to build better documentation for Python project. 
  - Opinionated therefore can understand more about the structure of your project. 
  - Allow automatic cross link (back and forth) between documentation across python packages. 
  - Use a documentation IR, to separate building the docs from rendering the docs in many contexts. 

This should hopefully allow a conda-forge-like model, where project upload their IR to a given repo, and a _single
website_ that contain multiple project documentation (without sub domains) can be build with better cross link between
project and _efficient_ page rebuild. 

This should also allow to reader documentation on _non html_ backend (think terminal), or provide documentation if
IDE (Spyder/Jupyterlab), without having to iframe it. 

## install 

You may need to get a modified version of numpydoc depending on the stage of development.

```
# clone this repo
# cd this repo
pip install flit
flit install --symlink
```

## Instructions / Overview

In the end there should be roughly 3 steps:

#### generation (gen.py module_name),

Which collect the documentation of a project into a doc-bundle; a number of
doc-blobs (currently json file), with a defined semantic structure, and
some metadata (version of the project this documentation refers to, and
potentially some other blobs)

During the generation a number of normalisation and inference can and should
happen, for example 

  - using type inference into the `Examples` sections of docstrings and storing
    those as pairs (token, reference), so that you can later decide that
    clicking on `np.array` in an example brings you to numpy array
    documentation; whether or not we are currently in the numpy doc. 
  - Parsing "See Also" into a well defined structure
  - running Example to generate images for docs with images (not implemented)
  - resolve package local references for example building numpy doc
    "`zeroes_like`" is non ambiguous and shoudl be Normalized to
    "`numpy.zeroes_like`", `~.pyplot.histogram`, normalized to
    `matplotlib.pyplot.histogram` as the **target** and `histogram` as the text
    ...etc.

The Generation step is likely project specific, as there might be import
conventions that are per-project and should not need to be repeated (`import
pandas as pd`, for example,)

#### Ingestion (crosslink.py)

The ingestion step take doc-bundle and/or doc-blobs and add them into a graph of
known items; the ingestion is critical to efficiently build the collection graph
metadata and understand which items refers to which; this allow the following: 

 - Update the list of backreferences to a docbundle
 - Update forward references metadata to know whether links are valid. 

Currently the ingestion loads all in memory and update all the bundle in place
but this can likely be done more efficiently. 

A lot more can likely be done at larger scale, like detecting if documentation
have changed in previous version so infer for which versions of a library this
documentation is valid. 

There is also likely some curating that might need to be done at that point, as
for example, numpy.array have an extremely large number of back-references.


#### Rendering (render.py)

Rendering can be done on on client side, which allows a lot of flexibility and
customisation. 

1) on a client IDE; the links can allow to navigate in the doc "Inspector" (for
example spyder) and will/can link only to already existing libraries of current
environment.


2) online experience can allow (back-)links to private doc-bundles to users. 





### Usage

Still quite hackish for now:

```bash
$ mkdir html
$ mkdir cache
$ rm htmls/*.html
$ python gen.py module_names
$ python crosslink.py
$ python render.py
```


