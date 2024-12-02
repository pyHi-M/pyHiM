# How to document

*[Build documentation locally](./dev_installation.md#build-documentation-locally)*

## Philosophy
Our documentation is structured as follows:

1. Getting started: Step by step guide from installation to running an example.
2. User guide: Detailed explanation of what *pyHiM* can do and how to do it step-by-step.
3. Reference guide: Technical description of software, including API documentation and architecture.
4. Contributor's guide: Description of project conventions and development tips.

| Designed for  | Learn        | Work      |
| ------------- | ------------ | --------- |
| **User**      | Quickstart     | User guide    |
| **Developer** | Contributor's guide | Reference |

## Build documentation

To build *pyHiM* documentation locally, we use [Sphinx](https://www.sphinx-doc.org/en/master/) which generates static, local HTML pages (see *Sphinx* section below). The documentation is automatically rebuilt by `readthedocs` using the github *pyHiM* source documentation files. This build of the documentation is hosted at [Read the Docs](https://readthedocs.org/) and available online.

The documentation is placed in the `/docs` directory. Two types of documentations are provided by the developer:
- Pure text files (preferably written in [markdown](https://www.markdownguide.org/basic-syntax/)).
- [Docstring](https://www.python.org/dev/peps/pep-0257/) within the source code (using NumPy style). See section below for more information.

### Sphinx

*Sphinx* is a documentation generator. It's goal is to translate source documentation files into HTML static webpages. The default format for the documentation files is [reStructuredText (reST)](https://docutils.sourceforge.io/rst.html), however, *markdown* can also be used (see section below).

We use *Sphinx* with these main extensions:
- [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) to automatically include documentation from docstring
- [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) to support NumPy style docstring
- [MyST](https://myst-parser.readthedocs.io/en/latest/sphinx/intro.html) to support Markdown-based documentation


#### How use reST directives in markdown with MyST

- **ReST:**

```reStructuredText
.. toctree::
:maxdepth: 1
:parameter_name: parameter_value

quick_install
classic_run
param_file
personalise_run
```

- **MyST:**

````markdown
```{toctree}
:maxdepth: 1
:parameter_name: parameter_value

quick_install
classic_run
param_file
personalise_run
```
````

#### Link a section from an other page
```markdown
[Build documentation locally](./dev_installation.md#build-documentation-locally)
```

## Document features

When you develop a new feature, please explain what it does and how use it in the user guide.

## Comment the code

> "Code is more often read than written."
>
> -- <cite> Guido van Rossum </cite>

Comments in code, help to understand your code in different ways:
- Planning future development
- Describe code, explain goal of this section
- Algorithmic description
- Tagging issues or improvement areas

When you comment code, be brief and focused.

### Docstring

*[Docstring](https://www.python.org/dev/peps/pep-0257/) is documentation string that describes what a public object does.*

This comment should appear after the `def` line, so at the beginning of:

- module
- function
- class
- method

It's the `__doc__` special attribute of that object. To print it, you can use `help()`function.

Convention: always use `"""triple double quotes"""`.

There are two forms of docstrings: one-liners and multi-line docstrings.

#### One-line

One-liners are for really obvious cases. They should really fit on one line.

For example:

```python
def kos_root():
    """Return the pathname of the KOS root directory."""
    global _kos_root
    if _kos_root: return _kos_root
    ...
```

#### Multi-line

*For a stand-alone program, the docstring should document the script's function and command line syntax, environment variables, and files... To be sufficient for a new user to use the command properly (or yourself in 2 years !).*

- Structure:
    1. A summary line just like a one-line docstring
    2. Followed by a blank line
    3. Followed by a more elaborate description

- On Spyder IDE and VSCode you can generate easily docstring by typing 3*`"`+`Enter`

- [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring): useful vscode extension to comment easily (settings > docstring format > google/sphinx/numpy).

- [NumPy/SciPy Docstrings Example:](https://realpython.com/documenting-python-code)

```python
def extract_header(file_loc, print_cols=False):
    """Gets and prints the spreadsheet's header columns

    Parameters
    ----------
    file_loc: str
        The file location of the spreadsheet
    print_cols: bool, optional
        A flag used to print the columns to the console (default is False)

    Returns
    -------
    list
        a list of strings representing the header columns
    """
    headers = []
    do_something()
    return headers
```

### Type hinting

For python 3.5+ user, we can use type hinting.
It's a simple way to describe input and output, this is an example:

> ```python
> def hello_name(name: str) -> str:
>     return(f"Hello {name}")
> ```
