# Vélin

French for Vellum

> Vellum is prepared animal skin or "membrane", typically used as a material for writing on. Parchment is another term
> for this material, and if vellum is distinguished from this, it is by vellum being made from calfskin, as opposed to
> that from other animals,[1] or otherwise being of higher quality


## install 

You may need to get a modified version of numpydoc depending on the stage of development.

```
# clone this repo
# cd this repo
pip install flit
flit install --symlink
```

## Autoreformat docstrings in minirst/ref.py

```
velin [--write] path-to-file.py or path-to-dir
```

Without `--write` vélin will print the suggested diff, with `--write` it will _attempt_  to update the files.

## options

```
$ velin --help
usage: velin [-h] [--context context] [--unsafe] [--check] [--no-diff] [--no-color] [--write] path [path ...]

reformat the docstrigns of some file

positional arguments:
  files              Files or folder to reformat

optional arguments:
  -h, --help         show this help message and exit
  --context context  Number of context lines in the diff
  --unsafe           Lift some safety feature (don't fail if updating the docstring is not indempotent
  --check            Print the list of files/lines number and exit with a non-0 exit status, Use it for CI.
  --no-diff          Do not print the diff
  --no-color
  --write            Try to write the updated docstring to the files
```



## other things part of this repo that will need to be pulled out at some point

### Not Sphinx

A project which is not sphinx (for current lack of a better name), it is _not meant_ to be a Sphinx replacement either
but to explore a different approach; mainly:

- Be more Python Specific; by knowing more about the language you can usually be smarter and simpler. 
- Separate documentation gathering, and building from rendering. 
  - Go from source to IR
  - From IR to final HTML – without extension execution. 
- Potentially offer a docstring reformatter (!not a linter), that can reformat docstrings automatically to follow
  numpydoc conventions.

This should hopefully allow a conda-forge-like model, where project upload their IR to a given repo, and a _single
website_ that contain multiple project documentation (without sub domains) can be build with better cross link between
project and _efficient_ page rebuild. 

This should also allow to reder documentation on _non html_ backend (think terminal), or provide documentation if
IDE (Spyder/Jupyterlab), without having to iframe it. 

### Usage

Still quite hackish for now:

```bash
$ mkdir html
$ rm htmls/*.html
$ python gen.py
```


