pyinstaller --onefile --clean \
--hidden-import=platformdirs.unix \
--windowed \
--add-binary="/usr/lib/x86_64-linux-gnu/libOpenGL.so.0:." \
--add-binary="/usr/lib/x86_64-linux-gnu/libOpenGL.so:." \
VideoF2B.py