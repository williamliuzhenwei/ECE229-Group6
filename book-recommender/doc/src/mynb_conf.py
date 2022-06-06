
# myst_nb settings
# 显示代码行数
nb_number_source_lines = True

# # Custom formats for reading notebook; suffix -> reader
# nb_custom_formats = {}

# Notebook level metadata key for config overrides
# nb_metadata_key = 'mystnb'

# Cell level metadata key for config overrides
# nb_cell_metadata_key = 'mystnb'

# Mapping of kernel name regex to replacement kernel name(applied before execution)
# nb_kernel_rgx_aliases = {}

# Execution mode for notebooks
# nb_execution_mode = 'off'

# Path to folder for caching notebooks (default: <outdir>)
# nb_execution_cache_path = ''

# Exclude (POSIX) glob patterns for notebooks
# nb_execution_excludepatterns = ()

# Execution timeout (seconds)
# nb_execution_timeout = 30

# Use temporary folder for the execution current working directory
# nb_execution_in_temp = False

# Allow errors during execution
nb_execution_allow_errors = True

# Raise an exception on failed execution, rather than emitting a warning
# nb_execution_raise_on_error = False

# Print traceback to stderr on execution error
# nb_execution_show_tb = False

# 合并 stderr 和 stdout 中的输出成单个流
nb_merge_streams = True

# The entry point for the execution output render class (in group `myst_nb.output_renderer`)
# nb_render_plugin = 'default'

# Remove code cell source
# nb_remove_code_source = False

# Remove code cell outputs
# nb_remove_code_outputs = False

# Number code cell source lines
# nb_number_source_lines = False

# Overrides for the base render priority of mime types: list of (builder name, mime type, priority)
nb_mime_priority_overrides = [
    ('html', 'text/plain', 0), # 最高级别
    ('latex', 'image/jpeg', None), # 禁用
    # ("gettext", 'customtype', None),
    # ('*', 'customtype', 20)
]

# Behaviour for stderr output
# nb_output_stderr = 'show'

# Pygments lexer applied to stdout/stderr and text/plain outputs
# nb_render_text_lexer = 'myst-ansi'

# Pygments lexer applied to error/traceback outputs
# nb_render_error_lexer = 'ipythontb'

# Options for image outputs (class|alt|height|width|scale|align)
# nb_render_image_options = {}

# Options for figure outputs (classes|name|caption|caption_before)
# nb_render_figure_options = {}

# The format to use for text/markdown rendering
# nb_render_markdown_format = 'commonmark'

# Javascript to be loaded on pages containing ipywidgets
# nb_ipywidgets_js = {'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js': {'integrity': 'sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=', 'crossorigin': 'anonymous'}, 'https://unpkg.com/@jupyter-widgets/html-manager@^0.20.0/dist/embed-amd.js': {'data-jupyter-widgets-cdn': 'https://cdn.jsdelivr.net/npm/', 'crossorigin': 'anonymous'}}
