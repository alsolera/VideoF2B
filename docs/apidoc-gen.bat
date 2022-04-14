REM Auto-generate API documentation for the VideoF2B package.
set SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance
sphinx-apidoc.exe -f -o .\source\api\ ..\videof2b\ ..\videof2b\version.py