import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT/"doc"), str(ROOT/"src")])
from _conf import *
print(ROOT)

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Add any paths that contain templates here, relative to this directory.
templates_path = [str(ROOT/'doc/_templates')]

# load extensions
extensions = [
    # "ablog",
    "myst_nb",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_thebe",
    "sphinx_copybutton",
    "sphinx_comments",
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "sphinx_inline_tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_automodapi.automodapi",
    # "sphinx.ext.todo",
    # "sphinxcontrib.bibtex",
    # "sphinx_togglebutton",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    # "sphinx_design",
    # "sphinx.ext.ifconfig",
    # "sphinxext.opengraph",
]


# The full version, including alpha/beta/rc tags
release = '0.1'


# basic build settings
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ["default.css"]
# -- 国际化输出 ----------------------------------------------------------------
language = "en" #'zh_CN'
locale_dirs = ['locales/']  # path is example but recommended.
gettext_compact = False  # optional.

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

nb_execution_mode = 'off'
nb_merge_streams = True