---
title: PaPyRi
separator: --- ?
verticalSeparator: \+\+\+
theme: solarized
revealOptions:
    transition: 'fade'
---

# Papyri 

+++

# What it is 

Papyri is a new python documentation paradigm and a set of tools from libraries and end users. 

- Flexible for libraries
- Richer representation for user in IDEs...
- Opinionated
- A direct attack on Mathemetica good documentation.
- A carrot to more "uniform" documentation conventions

+++

# What is is not

- A replacement for Sphinx.
- A replacement for writing docstring, tutorial, blogpost, any many other forms of docs.
- A tool that lints docs for you*

---

# Documentation Pain Points

+++

## For Users:

- Locally:
  - Have to read raw docstring, 
  - or read snippets not as good as online sphinx docs (no graph)

- Have issue finding actual website with docs for X.
  - and the right version.

- hard to discover new function/libs (e.g: numpy.array does not mention dask.array)

+++

## Maintainers

- Setting up sphinx 
    - plugins,. Often need to change code/config.py to change appearance and rebuild all (theme, copy-past button, ...)
- Poor inter-project connectivity. 
- Does not impact local version of docs for user (hence many hacks around dynamic __doc__ attributes)
- Hard to advertise (e.g: numpy.array does not mention dask.array)

+++

## Others:

- Sphinx can run arbitrary code, so security concern. 
- You already build and check the docs on CI, why rebuild on RTD ?
- Old docs website update and migration; (this version is deprecated, noindex-no-follow), color is out of fashion...
- difficult to customize for vendors, host for air gapped system.

---

# The idea



Separate "building" the docs from rendering the docs.

+++


Each project have project specific, compute-intensive, doc discovery and collection, generation.

Instead of producing HTML / PDF, produce an IR. 

+++

- Process this IR in a given context (which other projects you know of, the system you are on, the previous (or future)
  versions of this IR to get IR'.
  - Links and backreference links
  - which links are valid, 

- Get a graph of IR' 

(' because more informations than in basic IR).


+++

- Render IR' just in time, base on user desire.
   - ASCII/color rendering, 
   - Crosslink navigation don't need to be "URL" (see spyder/jupyterlab inspector).
   - can inject vendor specific information.
   - much _cheaper_ the dynamic `__doc__`, less performance impact, no arbitrary code exec.

--- 

# Frontends

+++ 

# Papyri local

- `papyri install matplotlib-doc`
or 
- dynamically lookup only on `hosted` (analytics?)

+++

- Allow IDEs, IPython to display rich environment specific docs with navigation; crosslinks with other installed project back and
  forth links, and example discovery. 

+++
- As good documentation of _current_ hosted sphinx docs like matplotlib.org, pandas, seaborn, numpy, with inline graphs... but
  with rendering that can respect user preference (color, theme, sections order...)

+++

- Can imagine a system of tagging and/or indexing users content to tell them in which projects they are already using
  functions/class/...

+++

# Papyri Open Source

+++

- Conda-Forge like model

+++

- Library author or community upload per-library-version doc-bundle built on CI.

+++

- Ingest and create a single-stop-shop website of _trusted_ quality docs, with crosslinks. 

+++

- Website can evolve features independently of libraries (re-building) docs

+++

- Can be _efficiently_ updated on new docs (unlike sphinx build-all)


+++

- Can have much smarter features than sphinx:
  - numpydoc.array : "This docstring has not changed between number 1.11 and numpy 1.23"
  - IPython.InputSpliter: "This parameter will be marked deprecated in IPython 7.18"

+++

- Can suggest new libraries:

  - sklearnGridSearch is use in `dask.distributed` examples.


--- 

# Papyri Pro

All of Papyri open-source plus, but if login with (institution) credentials

+++

- overwrite or extend give library docs with your own
    - e.g: instruction on how to setup Dash ssh cluster directly in dask docs.

+++

- Crosslink with your own private version of your docs & packages.

+++


- Directly open issues from docs, Links to experts,

+++

- Upload your env.yml and we narrow down the docs to only those packages.


--- 

# Papyri hosted

- Your own hosted version of that for internal use if you don't trust us
- Docs is JSON so no need "rebuild" can just download and audit the rendering code.
- Can still link _outward_ to public Papyri, or just proxy requests.
- Analytics of what you use look at.





Notes on 
- perf
- discoverability
- extend to non docstring.
- declarative config.





- Item 1 <!-- .element: class="fragment" data-fragment-index="1" -->
- Item 2 <!-- .element: class="fragment" data-fragment-index="2" -->





- http://localhost:5000/scipy.integrate.quadpack.quad  -> Math
- http://localhost:5000/scipy.integrate.quadrature.cumtrapz for plot.
- numpy.array for back references. 
- IPython Audio for fwd ad backref. 
- SPyder, 
- IPython 
