# -*- mode: python ; coding: utf-8 -*-

import platform
from importlib.metadata import metadata
from pathlib import Path

from videof2b.version import version_tuple


def make_version_win():
    '''
    Create the version metadata file for Windows EXE.
    '''
    meta = metadata('videof2b')
    product_name = meta['Name']
    template = f"""
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({version_tuple[0]}, {version_tuple[1]}, 0, 0),
    prodvers=({version_tuple[0]}, {version_tuple[1]}, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
    ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'FileDescription', u'{product_name}'),
        StringStruct(u'ProductName', u'{product_name}')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    Path('./version_win').open('w', encoding='utf8').write(template)


hidden_imports = []
proj_binaries = []
proj_datas = [
    ('resources', 'resources'),
]
proj_excludes = [
    'matplotlib',
]
win_version_file_name = ''


system_platform = platform.system()
if system_platform == 'Linux':
    hidden_imports = ['platformdirs.unix']
    proj_binaries = [
        ('/usr/lib/x86_64-linux-gnu/libOpenGL.so.0', '.'),
        ('/usr/lib/x86_64-linux-gnu/libOpenGL.so', '.'),
    ]
elif system_platform == 'Windows':
    hidden_imports = ['platformdirs.windows']
    make_version_win()
    win_version_file_name = 'version_win'


block_cipher = None

a = Analysis(['VideoF2B.py'],
             pathex=['.'],
             binaries=proj_binaries,
             datas=proj_datas,
             hiddenimports=hidden_imports,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=proj_excludes,
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure,
          a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='VideoF2B',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None,
          version=win_version_file_name)
