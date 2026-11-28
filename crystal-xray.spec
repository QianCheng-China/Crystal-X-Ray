# -*- mode: python ; coding: utf-8 -*-
a = Analysis(['xibao_app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['pyvista', 'pyvistaqt', 'vtk', 'matplotlib',
                   'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets'],
    hookspath=[], runtime_hooks=[],
    excludes=['tkinter', 'torch', 'torchvision', 'torchaudio',
              'pynvml', 'pandas', 'sympy', 'numba'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name='Crystal X-Ray', console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='Crystal X-Ray')
