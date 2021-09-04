pyinstaller ^
--onefile ^
--clean ^
--hidden-import=platformdirs.windows ^
--windowed ^
--add-data="resources;resources" ^
VideoF2B.py