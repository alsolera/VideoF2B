pyinstaller ^
--onefile ^
--clean ^
--hidden-import=platformdirs.windows ^
--exclude-module=matplotlib ^
--windowed ^
--add-data="resources;resources" ^
--version-file=version_win ^
VideoF2B.py
