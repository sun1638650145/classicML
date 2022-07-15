# 项目信息
project = 'classicML'
copyright = '2020-2022, Steve R. Sun'
author = 'Steve R. Sun'

# 版本号
release = '0.9.1a3'

# markdown插件
extensions = [
    'recommonmark',
    'sphinx_markdown_tables',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# 语言
language = 'zh'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# 主题
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
