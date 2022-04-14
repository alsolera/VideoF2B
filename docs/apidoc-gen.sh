# Auto-generate API documentation for the VideoF2B package.
export SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance
sphinx-apidoc -f -o ./source/api/ ../videof2b/ ../videof2b/version.py