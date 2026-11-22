# -*- coding: utf-8 -*-
"""
Crystal X-Ray v0.1 —— 晶胞演示软件
开发者: HighspeedG2304
依赖: numpy, pyvista, pyvistaqt, PySide6
运行: python xibao_app.py            检查依赖: python xibao_app.py --check
     软件渲染(虚拟机卡顿时): python xibao_app.py --sw
"""
import os, sys, json, math, traceback

APP_NAME = 'Crystal X-Ray'
APP_VERSION = '0.1'
APP_AUTHOR = 'HighspeedG2304'

os.environ.setdefault('QT_API', 'pyside6')
SW_MODE = '--sw' in sys.argv
if SW_MODE:
    os.environ['QT_OPENGL'] = 'software'
import numpy as np

# —— 启动诊断 ——
import time as _time
_T0 = _time.time()
def _diag(msg):
    print('[%6.1fs] %s' % (_time.time() - _T0, msg), flush=True)

def _die(msg):
    print('\n' + msg + '\n')
    try: input('按回车键退出...')
    except Exception: pass
    sys.exit(1)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_JSON = os.path.join(APP_DIR, '我的晶体.json')

print('=' * 58)
print('  %s v%s —— 晶胞演示软件' % (APP_NAME, APP_VERSION))
print('  开发者: %s' % APP_AUTHOR)
print('=' * 58)
_diag('模块导入开始（虚拟机/仿真环境下首次启动可能需要1~3分钟，请耐心等待）...')

try:
    import crystal_core as cc
except Exception as e:
    _die('无法导入 crystal_core.py（请与 xibao_app.py 放同一文件夹）\n' + repr(e))
_diag('crystal_core OK（%d 种晶体）' % len(cc.DB))

_diag('导入 PySide6...')
try:
    from PySide6.QtCore import Qt, QEvent
    from PySide6.QtWidgets import (QApplication, QMainWindow, QSplitter, QWidget,
        QVBoxLayout, QHBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem, QTabWidget,
        QGroupBox, QSlider, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
        QLineEdit, QPushButton, QTextBrowser, QTableWidget, QTableWidgetItem,
        QFileDialog, QMessageBox, QHeaderView, QFormLayout)
except Exception as e:
    _die('缺少 PySide6：pip install PySide6\n' + repr(e))
_diag('PySide6 OK')

_diag('导入 pyvista/vtk（较慢，属正常）...')
try:
    import pyvista as pv
    import vtk
except Exception as e:
    _die('缺少 pyvista/vtk：pip install pyvista pyvistaqt\n' + repr(e))
try:
    from pyvistaqt import QtInteractor
except Exception as e:
    _die('缺少 pyvistaqt：pip install pyvistaqt PySide6\n' + repr(e))
_diag('pyvista/vtk OK')

# 位置类型配色(按位置着色模式)
PTYPE_COLOR = {'corner': '#E53935', 'edge': '#FB8C00', 'face': '#43A047',
               'body': '#1E88E5',
               'prism_vertex': '#E53935', 'prism_face': '#43A047',
               'prism_body': '#1E88E5'}
PTYPE_CN = dict(cc.POS_TYPE)
PTYPE_CN.update({'prism_vertex': ('顶点', 6, 1/6), 'prism_face': ('面心', 2, .5),
                 'prism_body': ('体内', 1, 1.)})

def _frac_str(v):
    if abs(v) < 1e-6: return '0'
    for den, s in ((1,''),(2,'/2'),(3,'/3'),(4,'/4'),(6,'/6'),(8,'/8')):
        n = v * den
        if abs(n - round(n)) < 1e-6:
            return ('%d%s' % (round(n), s)) if den > 1 else '%d' % round(n)
    return '%.3f' % v

def _actor_opacity(ac, o):
    for fn in (lambda: setattr(ac, 'opacity', o),
               lambda: setattr(ac.prop, 'opacity', o),
               lambda: ac.GetProperty().SetOpacity(o)):
        try:
            fn(); return
        except Exception:
            continue

# ---------------- mesh 构建辅助 ----------------
def spheres_groups(atoms, radii, res, by_ptype=False):
    groups = {}
    for i, a in enumerate(atoms):
        key = a['ptype'] if by_ptype else a['elem']
        groups.setdefault(key, []).append(i)
    out = {}
    for key, idxs in groups.items():
        app = vtk.vtkAppendPolyData()
        for i in idxs:
            s = pv.Sphere(radius=float(radii[i]),
                          center=np.asarray(atoms[i]['cart'], float),
                          theta_resolution=res, phi_resolution=res)
            try: s.compute_normals(inplace=True)
            except Exception: pass
            s.cell_data['atom_idx'] = np.full(s.n_cells, i, dtype=np.int64)
            app.AddInputData(s)
        app.Update()
        out[key] = pv.wrap(app.GetOutput())
    return out

def bonds_mesh(atoms, pairs, rb):
    app = vtk.vtkAppendPolyData()
    n = 0
    for i, j in pairs:
        p1 = np.asarray(atoms[i]['cart'], float)
        p2 = np.asarray(atoms[j]['cart'], float)
        d = p2 - p1; L = float(np.linalg.norm(d))
        if L < 1e-6: continue
        app.AddInputData(pv.Cylinder(center=(p1 + p2) / 2, direction=d / L,
                                     height=L, radius=rb, resolution=12))
        n += 1
    if n == 0: return None
    app.Update()
    return pv.wrap(app.GetOutput())

def lines_polydata(seg_pairs):
    pts, seg = [], []
    for a, b in seg_pairs:
        pts.append(list(a)); pts.append(list(b))
        seg += [2, len(pts) - 2, len(pts) - 1]
    m = pv.PolyData(np.asarray(pts, float))
    m.lines = np.array(seg, np.int64)
    return m

CELL_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
              (0,4),(1,5),(2,6),(3,7)]
CELL_FACES = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]

def cell_corners(M, i, j, k):
    base = np.array([i, j, k], float) @ M
    out = []
    for u, v, w in ((0,0,0),(1,0,0),(1,1,0),(0,1,0),
                    (0,0,1),(1,0,1),(1,1,1),(0,1,1)):
        out.append(base + np.array([u, v, w], float) @ M)
    return np.array(out)

def quadfaces_mesh(corners):
    m = pv.PolyData(np.asarray(corners, float))
    conn = []
    for f in CELL_FACES:
        conn += [4] + f
    m.faces = np.array(conn, np.int64)
    return m

def hex_corners(a, center, z0, z1):
    vs = []
    for t in (0, 60, 120, 180, 240, 300):
        vs.append([center[0] + a * math.cos(math.radians(t)),
                   center[1] + a * math.sin(math.radians(t)), z0])
    for t in (0, 60, 120, 180, 240, 300):
        vs.append([center[0] + a * math.cos(math.radians(t)),
                   center[1] + a * math.sin(math.radians(t)), z1])
    return np.array(vs)

HEX_EDGES = [(i,(i+1)%6) for i in range(6)] + [(i,6+(i+1)%6) for i in range(6)] + \
            [(i, i+6) for i in range(6)]
HEX_FACES = [list(range(6)), list(range(6,12)), [0,1,7,6],[1,2,8,7],[2,3,9,8],
             [3,4,10,9],[4,5,11,10],[5,0,6,11]]

def hexfaces_mesh(corners):
    m = pv.PolyData(np.asarray(corners, float))
    conn = []
    for f in HEX_FACES:
        conn += [len(f)] + f
    m.faces = np.array(conn, np.int64)
    return m

# ---------------- 主窗口 ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('%s v%s —— 晶胞演示软件（%s）'
                            % (APP_NAME, APP_VERSION, APP_AUTHOR))
        self.resize(1420, 900)
        self.cr = None
        self.n = 1
        self.radius_scale = 1.0
        self.show_bonds = True
        self.show_frame = True
        self.show_labels = True
        self.show_faces = False
        self.show_poly = False
        self.show_holes = False
        self.color_by_ptype = False
        self.measure_mode = False
        self.mode = 'cell'
        self.variant = 'cell'
        self.slice_hkl = None
        self.slice_off = 0.0
        self._atoms = []; self._atom_actors = []
        self._hl = []; self._sel = None
        self._measure_a = None; self._measure_actors = []
        self._stack_atoms = []; self._stack_r = 1.3
        self.custom_db = self._load_custom()
        self._press_xy = None
        self._picker = None
        self._build_ui()
        self._setup_picker()
        self._populate_tree()
        self.load_crystal(cc.DB['nacl'])

    # ---------- UI ----------
    def _build_ui(self):
        mb = self.menuBar()
        mhelp = mb.addMenu('帮助')
        act_about = mhelp.addAction('关于 %s' % APP_NAME)
        act_about.triggered.connect(self._about)
        act_usage = mhelp.addAction('使用说明')
        act_usage.triggered.connect(self._usage)

        split = QSplitter(Qt.Horizontal)
        self.setCentralWidget(split)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel('晶体库（双击载入）')
        self.tree.setMinimumWidth(230)
        self.tree.itemDoubleClicked.connect(self._on_tree)
        split.addWidget(self.tree)

        frame = QWidget(); lay = QVBoxLayout(frame); lay.setContentsMargins(0,0,0,0)
        _diag('创建3D渲染窗口(这一步最容易因无GPU失败)...')
        try:
            self.plot = QtInteractor(frame)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, '3D初始化失败',
                'VTK渲染窗口创建失败（常见于虚拟机无GPU加速）。\n'
                '请关闭本程序，改用「启动兼容模式.bat」启动。\n\n错误: %r' % e)
            raise
        _diag('3D渲染窗口OK')
        try:   # 虚拟机下关闭多重采样，减少显卡驱动上下文问题
            rw = getattr(self.plot, 'render_window', None)
            if rw is None: rw = self.plot.ren_win
            rw.SetMultiSamples(0)
        except Exception:
            pass
        if not SW_MODE:
            try: self.plot.enable_depth_peeling()
            except Exception: pass
        try: self.plot.set_background('#F5F7FA')
        except Exception: pass
        lay.addWidget(self.plot)
        split.addWidget(frame)

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(390); self.tabs.setMaximumWidth(470)
        self.tabs.addTab(self._tab_display(), '显示')
        self.tabs.addTab(self._tab_stats(), '统计与均摊')
        self.tabs.addTab(self._tab_stack(), '堆积演示')
        self.tabs.addTab(self._tab_custom(), '自定义')
        split.addWidget(self.tabs)
        split.setSizes([240, 800, 420])
        self.statusBar().showMessage(
            '拖动=旋转 · 滚轮=缩放 · 单击原子=均摊/信息 · 双击左侧晶体库载入'
            '（虚拟机首次加载较慢，属正常）')

    def _about(self):
        QMessageBox.about(self, '关于',
            '<h3>%s v%s</h3><p>晶胞演示软件（化学教学）</p>'
            '<p>开发者：<b>%s</b></p>'
            '<p>功能：35种内置晶体 · 均摊演示 · 统计面板 · 金属密堆积 '
            '· 六棱柱/平行六面体 · 空隙 · 配位多面体 · 剖面 · 测距 '
            '· 自定义晶胞 · CIF导入</p>'
            '<p style="color:#666">基于 Python + PySide6 + PyVista/VTK'
            ' + 自研晶体学引擎</p>' % (APP_NAME, APP_VERSION, APP_AUTHOR))

    def _usage(self):
        QMessageBox.information(self, '使用说明',
            '<b>基本操作</b><br>'
            '· 左键拖动=旋转，滚轮=缩放，中键拖动=平移<br>'
            '· 单击任意原子 → 高亮并展开共享它的晶胞（均摊演示），'
            '右侧显示 8×1/8=1 等推导<br>'
            '· 再次点击空白处 → 取消选择<br>'
            '· 左侧晶体库<b>双击</b>载入晶体<br><br>'
            '<b>显示页</b>：晶胞数1~3、原子半径、键/框/标注/多面体/空隙开关、'
            '剖面(100)(110)(111)、测距模式（依次点两个原子）、按位置类型着色'
            '（顶点红/棱橙/面绿/体内蓝）<br><br>'
            '<b>统计页</b>：位置分解、化学式、配位数、密度、空间利用率、教学要点<br><br>'
            '<b>堆积页</b>：ABAB/ABCABC/自定义序列密置层堆积，自动画出'
            'FCC立方晶胞框或六棱柱框；点球显示配位数<br><br>'
            '<b>自定义页</b>：输入晶胞参数+原子分数坐标；导入CIF文件；'
            '保存到“我的晶体库”（下次启动自动载入）<br><br>'
            '<b>六方晶体（Mg等）</b>：显示形式可切换 平行六面体晶胞 ↔ 教材式六棱柱')

    def _tab_display(self):
        w = QWidget(); v = QVBoxLayout(w)
        g = QGroupBox('晶胞'); v.addWidget(g)
        f = QFormLayout(g)
        self.sl_n = QSlider(Qt.Horizontal); self.sl_n.setRange(1, 3)
        self.sl_n.setValue(1)
        self.lb_n = QLabel('1×1×1')
        self.sl_n.valueChanged.connect(
            lambda x: self.lb_n.setText('%d×%d×%d' % (x, x, x)))
        self.sl_n.sliderReleased.connect(self._chg)
        f.addRow(QLabel('晶胞数'), self.sl_n); f.addRow('', self.lb_n)
        self.sl_r = QSlider(Qt.Horizontal); self.sl_r.setRange(15, 100)
        self.sl_r.setValue(100)
        self.lb_r = QLabel('半径 100%')
        self.sl_r.valueChanged.connect(
            lambda x: self.lb_r.setText('半径 %d%%' % x))
        self.sl_r.sliderReleased.connect(lambda: (setattr(self, 'radius_scale',
            self.sl_r.value() / 100), self._chg()))
        f.addRow(QLabel('原子半径'), self.sl_r); f.addRow('', self.lb_r)
        self.cmb_variant = QComboBox()
        self.cmb_variant.addItems(['平行六面体晶胞', '教材式六棱柱'])
        self.cmb_variant.currentIndexChanged.connect(
            lambda i: (setattr(self, 'variant', 'cell' if i == 0 else 'prism'),
                       self._chg()))
        f.addRow(QLabel('显示形式'), self.cmb_variant)
        g2 = QGroupBox('显示开关'); v.addWidget(g2)
        f2 = QFormLayout(g2)
        for name, attr in [('显示化学键', 'show_bonds'), ('显示晶胞框', 'show_frame'),
                           ('元素标注', 'show_labels'),
                           ('半透明晶胞面', 'show_faces'),
                           ('配位多面体', 'show_poly'), ('空隙标记', 'show_holes'),
                           ('按位置类型着色', 'color_by_ptype')]:
            cb = QCheckBox(name)
            cb.setChecked(getattr(self, attr))
            cb.stateChanged.connect(
                lambda s, a=attr: (setattr(self, a, s == 2), self._chg()))
            f2.addRow(cb)
        g3 = QGroupBox('剖面'); v.addWidget(g3)
        f3 = QFormLayout(g3)
        self.cmb_slice = QComboBox()
        self.cmb_slice.addItems(['无', '(100)', '(110)', '(111)'])
        self.cmb_slice.currentIndexChanged.connect(self._chg_slice)
        f3.addRow(QLabel('切面'), self.cmb_slice)
        self.sl_slice = QSlider(Qt.Horizontal); self.sl_slice.setRange(-100, 100)
        self.sl_slice.setValue(0)
        self.lb_slice = QLabel('偏移 0%')
        self.sl_slice.valueChanged.connect(
            lambda x: self.lb_slice.setText('偏移 %d%%' % x))
        self.sl_slice.sliderReleased.connect(self._chg)
        f3.addRow(self.sl_slice); f3.addRow('', self.lb_slice)
        g4 = QGroupBox('工具'); v.addWidget(g4)
        f4 = QFormLayout(g4)
        self.btn_meas = QPushButton('测距模式（点两个原子）')
        self.btn_meas.setCheckable(True)
        self.btn_meas.clicked.connect(self._toggle_measure)
        f4.addRow(self.btn_meas)
        self.btn_clear = QPushButton('取消选择 / 清除测距')
        self.btn_clear.clicked.connect(self._clear_selection)
        f4.addRow(self.btn_clear)
        btn = QPushButton('复位视角')
        btn.clicked.connect(self._reset_cam)
        f4.addRow(btn)
        self.btn_bg = QPushButton('切换背景（浅/深）')
        self.btn_bg.clicked.connect(self._toggle_bg)
        f4.addRow(self.btn_bg)
        v.addStretch()
        return w

    def _tab_stats(self):
        w = QWidget(); v = QVBoxLayout(w)
        self.sel_browser = QTextBrowser()
        self.sel_browser.setMaximumHeight(130)
        self.sel_browser.setHtml('<p>单击 3D 视图中的任意原子，这里会显示'
                                 '<b>均摊分析</b>（该原子被几个晶胞共享）。</p>')
        v.addWidget(QLabel('<b>选中原子</b>')); v.addWidget(self.sel_browser)
        v.addWidget(QLabel('<b>晶胞统计</b>'))
        self.stats_browser = QTextBrowser()
        v.addWidget(self.stats_browser)
        return w

    def _tab_stack(self):
        w = QWidget(); v = QVBoxLayout(w)
        g = QGroupBox('密置层堆积'); v.addWidget(g)
        f = QFormLayout(g)
        self.cmb_seq = QComboBox()
        self.cmb_seq.addItems(['ABAB…（六方最密）', 'ABCABC…（面心最密）', '自定义'])
        self.cmb_seq.currentIndexChanged.connect(self._stack_chg)
        f.addRow(QLabel('序列'), self.cmb_seq)
        self.ed_seq = QLineEdit('ABCB')
        self.ed_seq.setToolTip('只允许字母A/B/C，例如 ABCBA')
        f.addRow(QLabel('自定义'), self.ed_seq)
        self.sp_layers = QSpinBox(); self.sp_layers.setRange(1, 8)
        self.sp_layers.setValue(4)
        self.sp_layers.valueChanged.connect(self._stack_chg)
        f.addRow(QLabel('层数'), self.sp_layers)
        self.cb_lcolor = QCheckBox('按层着色（A红 B绿 C蓝）')
        self.cb_lcolor.setChecked(True)
        self.cb_lcolor.stateChanged.connect(lambda s: self._stack_chg())
        f.addRow(self.cb_lcolor)
        self.cb_frame = QCheckBox('显示晶胞框（立方/六棱柱）')
        self.cb_frame.setChecked(True)
        self.cb_frame.stateChanged.connect(lambda s: self._stack_chg())
        f.addRow(self.cb_frame)
        btn = QPushButton('重建堆积（并切换到堆积模式）')
        btn.clicked.connect(self._stack_enter)
        f.addRow(btn)
        self.lb_seq = QLabel('')
        f.addRow(self.lb_seq)
        info = QTextBrowser()
        info.setHtml(
            '<h4>金属密堆积对比</h4><table border="1" cellspacing="0" cellpadding="4">'
            '<tr><th>堆积</th><th>格子</th><th>配位数</th><th>利用率</th><th>实例</th></tr>'
            '<tr><td>简单立方</td><td>P</td><td>6</td><td>52.36%</td><td>Po</td></tr>'
            '<tr><td>体心立方</td><td>I</td><td>8</td><td>68.02%</td>'
            '<td>Na K Fe Cr W</td></tr>'
            '<tr><td>六方最密</td><td>六方P</td><td>12</td><td>74.05%</td>'
            '<td>Mg Zn Ti</td></tr>'
            '<tr><td>面心最密</td><td>F</td><td>12</td><td>74.05%</td>'
            '<td>Cu Ag Au Al</td></tr></table>'
            '<p>• 密置层内每个球与 <b>6</b> 个球相切；层间错位形成空隙。<br>'
            '• <b>ABCABC…</b> → 面心立方（第4层回到A，可找出斜立的立方晶胞）。<br>'
            '• <b>ABAB…</b> → 六方最密（六棱柱晶胞：12顶点×1/6+2面心×1/2+3体内=6）。'
            '<br>• 四种堆积中，<b>面心/六方</b>空间利用率最高(74.05%)。</p>')
        v.addWidget(info)
        return w

    def _tab_custom(self):
        w = QWidget(); v = QVBoxLayout(w)
        g = QGroupBox('自定义晶胞'); v.addWidget(g)
        f = QFormLayout(g)
        self.ed_name = QLineEdit('我的晶体')
        f.addRow(QLabel('名称'), self.ed_name)
        self.sp_a = QDoubleSpinBox(); self.sp_b = QDoubleSpinBox()
        self.sp_c = QDoubleSpinBox()
        for sp in (self.sp_a, self.sp_b, self.sp_c):
            sp.setRange(1.0, 50.0); sp.setDecimals(3)
        self.sp_a.setValue(5.0); self.sp_b.setValue(5.0); self.sp_c.setValue(5.0)
        row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0)
        h.addWidget(self.sp_a); h.addWidget(self.sp_b); h.addWidget(self.sp_c)
        f.addRow('a / b / c (Å)', row)
        self.sp_al = QDoubleSpinBox(); self.sp_be = QDoubleSpinBox()
        self.sp_ga = QDoubleSpinBox()
        for sp in (self.sp_al, self.sp_be, self.sp_ga):
            sp.setRange(20.0, 160.0); sp.setDecimals(2); sp.setValue(90.0)
        row2 = QWidget(); h2 = QHBoxLayout(row2); h2.setContentsMargins(0,0,0,0)
        h2.addWidget(self.sp_al); h2.addWidget(self.sp_be); h2.addWidget(self.sp_ga)
        f.addRow('α β γ (°)', row2)
        self.cmb_cent = QComboBox()
        self.cmb_cent.addItems(['P', 'I', 'F', 'A', 'B', 'C', 'R'])
        f.addRow(QLabel('布拉维格子'), self.cmb_cent)
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(['元素', 'x', 'y', 'z'])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setMaximumHeight(190)
        v.addWidget(self.tbl)
        rowb = QWidget(); hb = QHBoxLayout(rowb); hb.setContentsMargins(0,0,0,0)
        for txt, fn in (('添加行', self._add_row), ('删除行', self._del_row),
                        ('以当前晶体为模板', self._fill_from_current)):
            b = QPushButton(txt); b.clicked.connect(fn); hb.addWidget(b)
        v.addWidget(rowb)
        rowb2 = QWidget(); hb2 = QHBoxLayout(rowb2); hb2.setContentsMargins(0,0,0,0)
        b1 = QPushButton('生成并显示'); b1.clicked.connect(self._make_custom)
        b2 = QPushButton('导入CIF…'); b2.clicked.connect(self._import_cif)
        b3 = QPushButton('保存到我的晶体库'); b3.clicked.connect(self._save_custom)
        hb2.addWidget(b1); hb2.addWidget(b2); hb2.addWidget(b3)
        v.addWidget(rowb2)
        self.lb_custom = QLabel('')
        v.addWidget(self.lb_custom)
        v.addStretch()
        self._add_row('Na', 0, 0, 0)
        self._add_row('Cl', 0.5, 0.5, 0.5)
        return w

    # ---------- 晶体库树 ----------
    def _populate_tree(self):
        self.tree.clear()
        cats = {}
        for cr in list(cc.CRYSTAL_DB) + list(self.custom_db.values()):
            cats.setdefault(cr['category'], []).append(cr)
        for cat in ['金属晶体', '离子晶体', '共价晶体', '分子晶体', '自定义']:
            if cat not in cats: continue
            top = QTreeWidgetItem([cat])
            for cr in cats[cat]:
                it = QTreeWidgetItem([cr['name']])
                it.setData(0, Qt.UserRole, cr['key'])
                top.addChild(it)
            self.tree.addTopLevelItem(top)
            top.setExpanded(cat in ('金属晶体', '离子晶体'))

    def _on_tree(self, item, col):
        key = item.data(0, Qt.UserRole)
        if not key: return
        cr = cc.DB.get(key) or self.custom_db.get(key)
        if cr: self.load_crystal(cr)

    # ---------- 载入 / 重建 ----------
    def load_crystal(self, cr):
        self.cr = cr
        self.mode = 'cell'
        self._clear_selection()
        self.cmb_variant.setEnabled(bool(cr.get('prism')))
        if not cr.get('prism'):
            self.cmb_variant.setCurrentIndex(0)
            self.variant = 'cell'
        self.rebuild()
        self._update_stats()
        self.statusBar().showMessage('已载入：%s（%s）' % (cr['name'], cr['lattice']), 8000)

    def _chg(self):
        if self.mode == 'cell': self.rebuild()

    def _chg_slice(self, i):
        self.slice_hkl = [None, (1,0,0), (1,1,0), (1,1,1)][i]
        self.sl_slice.setEnabled(i > 0)
        self._chg()

    def _reset_cam(self):
        try:
            self.plot.reset_camera(); self.plot.isometric_view()
        except Exception: pass

    def _toggle_bg(self):
        self._bg = not getattr(self, '_bg', False)
        try:
            self.plot.set_background('#20242B' if self._bg else '#F5F7FA')
        except Exception: pass

    def rebuild(self):
        try:
            if self.mode == 'stack':
                self._rebuild_stack()
            else:
                self._rebuild_cell()
        except Exception:
            traceback.print_exc()
            self.statusBar().showMessage('渲染出错：详见控制台 traceback', 10000)

    def _clear_scene(self):
        try: self.plot.clear()
        except Exception: pass
        self._atom_actors = []; self._hl = []; self._measure_actors = []
        self._sel = None; self._measure_a = None

    def _slice_clip(self, mesh):
        if self.slice_hkl is None or mesh is None or mesh.n_cells == 0:
            return mesh
        try:
            M = cc.lattice_matrix(self.cr)
            h, k, l = self.slice_hkl
            n = h * np.cross(M[1], M[2]) + k * np.cross(M[2], M[0]) + \
                l * np.cross(M[0], M[1])
            n = n / np.linalg.norm(n)
            center = np.array([self.n / 2.] * 3) @ M
            extent = np.abs(center).max() * 2
            org = center + n * (self.slice_off * extent)
            out = mesh.clip(normal=tuple(n), origin=tuple(org))
            return out if out is not None and out.n_cells > 0 else None
        except Exception:
            traceback.print_exc()
            return mesh

    def _rebuild_cell(self):
        cr = self.cr
        if cr is None: return
        self._clear_scene()
        prism = (self.variant == 'prism') and cr.get('prism')
        if prism:
            atoms = cc.prism_atoms(cr, self.n)
        else:
            atoms = cc.supercell_atoms(cr, self.n)
        self._atoms = atoms
        radii = [cc.display_radius(cr, a['elem']) * self.radius_scale for a in atoms]
        res = 24 if len(atoms) <= 120 else (18 if len(atoms) <= 320 else 12)
        if SW_MODE: res = max(10, res - 6)
        groups = spheres_groups(atoms, radii, res, self.color_by_ptype)
        for key, mesh in groups.items():
            color = PTYPE_COLOR.get(key, '#90A4AE') if self.color_by_ptype \
                    else cc.elem_color(key)
            mesh = self._slice_clip(mesh)
            if mesh is None: continue
            ac = self.plot.add_mesh(mesh, color=color, ambient=0.22,
                                    diffuse=0.85, specular=0.35, specular_power=25)
            self._atom_actors.append(ac)
        if self.show_bonds:
            try:
                pairs = cc.bonds_for(cr, atoms)
                bm = bonds_mesh(atoms, pairs, rb=max(0.05, 0.012 * cr['a']))
                if bm is not None:
                    bm = self._slice_clip(bm)
                    if bm is not None:
                        self.plot.add_mesh(bm, color='#546E7A')
            except Exception:
                traceback.print_exc()
        M = cc.lattice_matrix(cr)
        if self.show_frame:
            if prism:
                a, c = cr['a'], cr['c']
                segs = []
                for ctr in cc.prism_cells(cr, self.n):
                    for k in range(self.n):
                        co = hex_corners(a, ctr, k * c, (k + 1) * c)
                        for i, j in HEX_EDGES:
                            segs.append((co[i], co[j]))
                        if self.show_faces:
                            self.plot.add_mesh(hexfaces_mesh(co), color='#78909C',
                                               opacity=0.05)
                self.plot.add_mesh(lines_polydata(segs), color='#37474F',
                                   line_width=2)
            else:
                segs = []
                for i in range(self.n):
                    for j in range(self.n):
                        for k in range(self.n):
                            co = cell_corners(M, i, j, k)
                            for x, y in CELL_EDGES:
                                segs.append((co[x], co[y]))
                            if self.show_faces:
                                self.plot.add_mesh(quadfaces_mesh(co),
                                                   color='#78909C', opacity=0.05)
                self.plot.add_mesh(lines_polydata(segs), color='#37474F',
                                   line_width=2)
        if self.show_labels and 0 < len(atoms) <= 64:
            try:
                self.plot.add_point_labels(
                    [tuple(a['cart']) for a in atoms],
                    [a['elem'] for a in atoms], font_size=12, show_points=False,
                    always_visible=False, text_color='#263238')
            except Exception:
                try:
                    self.plot.add_point_labels(
                        [tuple(a['cart']) for a in atoms],
                        [a['elem'] for a in atoms])
                except Exception:
                    pass
        if self.show_holes and cr.get('holes'):
            try: self._add_holes(cr, M)
            except Exception: traceback.print_exc()
        if self.show_poly:
            try: self._add_polyhedra(cr, atoms)
            except Exception: traceback.print_exc()
        try:
            self.plot.add_text('%s %s | %s n=%d%s' % (APP_NAME, APP_VERSION,
                               cr['key'], self.n, ' prism' if prism else ''),
                               position='upper_left', font_size=10,
                               color='#37474F')
        except Exception:
            pass
        self._reset_cam()

    def _add_holes(self, cr, M):
        octa, tetra = cc.fcc_holes(cr)
        rm = 0.07 * cr['a']
        def draw(lst, empty_color):
            for f, occ in lst:
                for i in range(self.n):
                    for j in range(self.n):
                        for k in range(self.n):
                            p = np.array([i + f[0], j + f[1], k + f[2]]) @ M
                            col = cc.elem_color(occ) if occ else empty_color
                            s = pv.Sphere(radius=rm, center=p,
                                          theta_resolution=10, phi_resolution=10)
                            self.plot.add_mesh(s, color=col, opacity=0.85)
        draw(octa, '#FF8000')
        draw(tetra, '#00A6D6')

    def _add_polyhedra(self, cr, atoms):
        pos = np.array([a['cart'] for a in atoms])
        for ph in cc.coordination_polyhedra(cr, atoms):
            vs = pos[ph['verts']]
            cen = vs.mean(axis=0)
            vs2 = cen + 0.8 * (vs - cen)
            faces = cc.convex_hull_faces(vs2)
            if not faces: continue
            m = pv.PolyData(vs2)
            conn = []
            for fa in faces:
                conn += [3] + list(fa)
            m.faces = np.array(conn, np.int64)
            self.plot.add_mesh(m, color=cc.elem_color(atoms[ph['center']]['elem']),
                               opacity=0.28, show_edges=True, edge_color='#263238')

    # ---------- 堆积模式 ----------
    def _stack_enter(self):
        self.mode = 'stack'
        self._rebuild_stack()

    def _stack_seq_string(self):
        i = self.cmb_seq.currentIndex()
        L = self.sp_layers.value()
        if i == 0: pat = 'AB'
        elif i == 1: pat = 'ABC'
        else:
            pat = ''.join(c for c in self.ed_seq.text().upper() if c in 'ABC') or 'AB'
        return (pat * 20)[:L]

    def _stack_chg(self):
        if self.mode == 'stack':
            self._rebuild_stack()

    def _rebuild_stack(self):
        self._clear_scene()
        seq = self._stack_seq_string()
        L = self.sp_layers.value()
        self.lb_seq.setText('当前序列：%s（%d层）' % (seq, L))
        atoms = cc.build_stack(seq, r=self._stack_r, extent=4)
        self._stack_atoms = atoms
        r = self._stack_r
        res = 10 if SW_MODE else 16
        groups = {}
        for a in atoms:
            key = a['letter'] if self.cb_lcolor.isChecked() else 'X'
            groups.setdefault(key, []).append(a['cart'])
        colmap = {'A': '#E53935', 'B': '#43A047', 'C': '#1E88E5', 'X': '#78909C'}
        for key, pts in groups.items():
            app = vtk.vtkAppendPolyData()
            for p in pts:
                s = pv.Sphere(radius=r, center=p, theta_resolution=res,
                              phi_resolution=res)
                try: s.compute_normals(inplace=True)
                except Exception: pass
                app.AddInputData(s)
            app.Update()
            m = pv.wrap(app.GetOutput())
            m.cell_data['atom_idx'] = np.full(m.n_cells, -1, np.int64)
            self.plot.add_mesh(m, color=colmap.get(key, '#78909C'),
                               ambient=0.22, diffuse=0.85, specular=0.35)
        if self.cb_frame.isChecked():
            self._stack_frames(seq, L, r)
        try:
            self.plot.add_text('%s %s | Stack %s (%d layers)'
                               % (APP_NAME, APP_VERSION, seq, L),
                               position='upper_left', font_size=10,
                               color='#37474F')
        except Exception:
            pass
        self._reset_cam()

    def _stack_frames(self, seq, L, r):
        for k in range(0, max(0, L - 3)):
            corners, ok = cc.fcc_cube_in_stack(seq, r, k0=k)
            if ok:
                segs = []
                for x, y in [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),
                             (3,6),(4,7),(5,7),(6,7)]:
                    segs.append((np.array(corners[x]), np.array(corners[y])))
                self.plot.add_mesh(lines_polydata(segs), color='#0277BD',
                                   line_width=3)
                break
        for k in range(0, max(0, L - 2)):
            hp, ok = cc.hcp_prism_in_stack(seq, r, k0=k)
            if ok:
                hexv, z0, z1 = hp
                co = hex_corners(2 * r, (0, 0), z0, z1)
                segs = [(co[i], co[j]) for i, j in HEX_EDGES]
                self.plot.add_mesh(lines_polydata(segs), color='#C2185B',
                                   line_width=3)
                break

    # ---------- 拾取(Qt事件过滤器实现, 跨版本可靠) ----------
    def _setup_picker(self):
        try:
            self._picker = vtk.vtkCellPicker()
            self._picker.SetTolerance(0.02)
        except Exception:
            self._picker = None
        try:
            self.plot.installEventFilter(self)
        except Exception:
            traceback.print_exc()
        _diag('拾取器已安装(Qt事件过滤器)')

    def eventFilter(self, obj, event):
        if obj is self.plot:
            t = event.type()
            if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                try:
                    p = event.position()
                    self._press_xy = (p.x(), p.y())
                except Exception:
                    self._press_xy = None
            elif t == QEvent.MouseButtonRelease and \
                    event.button() == Qt.LeftButton and \
                    self._press_xy is not None:
                try:
                    p = event.position()
                    dx = abs(p.x() - self._press_xy[0])
                    dy = abs(p.y() - self._press_xy[1])
                except Exception:
                    dx = dy = 99
                self._press_xy = None
                if dx + dy < 6:            # 位移很小 → 视为点击而非旋转拖拽
                    self._do_pick(p.x(), p.y())
        return super().eventFilter(obj, event)

    def _vtk_xy(self, x_qt, y_qt):
        """Qt坐标(左上原点,逻辑像素) -> VTK坐标(左下原点,设备像素)"""
        try:
            rw = getattr(self.plot, 'render_window', None)
            if rw is None: rw = self.plot.ren_win
            _, h = rw.GetSize()
            base = self.plot.height()
            scale = (h / base) if (h and base) else 1.0
            return int(round(x_qt * scale)), int(round((base - y_qt) * scale))
        except Exception:
            return int(x_qt), int(y_qt)

    def _picked_ds(self, p):
        for g in (lambda: p.GetMapper().GetInput(),
                  lambda: p.GetMapper().GetInputAsDataSet(),
                  lambda: p.GetProp3D().GetMapper().GetInput(),
                  lambda: p.GetProp3D().GetMapper().GetInputAsDataSet(),
                  lambda: p.GetDataSet()):
            try:
                ds = g()
                if ds is not None: return ds
            except Exception:
                continue
        return None

    def _do_pick(self, x_qt, y_qt):
        if self._picker is None: return
        x, y = self._vtk_xy(x_qt, y_qt)
        try:
            self._picker.Pick(x, y, 0, self.plot.renderer)
        except Exception:
            return
        cid = self._picker.GetCellId()
        if cid < 0:
            if self.mode == 'cell' and not self.measure_mode:
                self._clear_selection()
            return
        ds = self._picked_ds(self._picker)
        if ds is None: return
        try:
            arr = ds.GetCellData().GetArray('atom_idx')
            if arr is None: return
            idx = int(arr.GetTuple1(cid))
        except Exception:
            return
        if self.mode == 'stack':
            self._stack_pick()
            return
        if 0 <= idx < len(self._atoms):
            if self.measure_mode:
                self._measure_pick(idx)
            else:
                self._show_selection(idx)

    def _stack_pick(self):
        try:
            pp = np.array(self._picker.GetPickPosition())
        except Exception:
            return
        pts = np.array([a['cart'] for a in self._stack_atoms])
        if len(pts) == 0: return
        i = int(np.argmin(np.linalg.norm(pts - pp, axis=1)))
        r = self._stack_r
        d = np.linalg.norm(pts - pts[i], axis=1)
        ntouch = int(np.sum((d > 1e-6) & (d <= 2 * r * 1.03)))
        a = self._stack_atoms[i]
        self.statusBar().showMessage(
            '第%d层(%s层)的球：与 %d 个球相切（配位数%d）'
            % (a['layer'] + 1, a['letter'], ntouch, ntouch), 12000)

    # 均摊选择
    def _show_selection(self, idx):
        atom = self._atoms[idx]
        self._sel = idx
        for ac in self._hl:
            try: self.plot.remove_actor(ac)
            except Exception: pass
        self._hl = []
        for ac in self._atom_actors:
            _actor_opacity(ac, 0.22)
        cr = self.cr
        r = cc.display_radius(cr, atom['elem']) * self.radius_scale
        shell = pv.Sphere(radius=r * 1.15, center=atom['cart'],
                          theta_resolution=24, phi_resolution=24)
        ac = self.plot.add_mesh(shell, color='#FFD54F', opacity=0.55)
        self._hl.append(ac)
        prism = (self.variant == 'prism') and cr.get('prism')
        try:
            if prism:
                self._hl_prism_frames(atom)
            else:
                self._hl_cell_frames(atom)
        except Exception:
            traceback.print_exc()
        self._update_sel_html(atom)

    def _hl_cell_frames(self, atom):
        M = cc.lattice_matrix(self.cr)
        i, j, k = atom['cell']
        u, v, w = atom['frac']
        axes = []
        for comp, cidx in ((u, 0), (v, 1), (w, 2)):
            if abs(comp) < 1e-6:
                axes.append([cidx, -1])
        cells = [(i, j, k)]
        import itertools
        for combo in itertools.product(*([[0, -1]] * len(axes))):
            ci, cj, ck = i, j, k
            for (axis, _), off in zip(axes, combo):
                if axis == 0: ci += off
                elif axis == 1: cj += off
                else: ck += off
            if (ci, cj, ck) != (i, j, k):
                cells.append((ci, cj, ck))
        for ci, cj, ck in cells:
            co = cell_corners(M, ci, cj, ck)
            segs = [(co[x], co[y]) for x, y in CELL_EDGES]
            ac = self.plot.add_mesh(lines_polydata(segs), color='#E91E63',
                                    line_width=3)
            self._hl.append(ac)
            fm = self.plot.add_mesh(quadfaces_mesh(co), color='#E91E63',
                                    opacity=0.06)
            self._hl.append(fm)

    def _hl_prism_frames(self, atom):
        cr = self.cr
        a, c = cr['a'], cr['c']
        w1 = np.array([1.5 * a, a * math.sqrt(3) / 2])
        w2 = np.array([1.5 * a, -a * math.sqrt(3) / 2])
        q = np.array(atom['cart'][:2], float)
        z = atom['cart'][2]
        layer = int(round(z / c))
        nodes = []
        R = 4 * self.n * a
        for m in range(-int(R / (1.5 * a)) - 2, int(R / (1.5 * a)) + 3):
            for n_ in range(-int(R / (1.5 * a)) - 2, int(R / (1.5 * a)) + 3):
                p = m * w1 + n_ * w2
                if np.linalg.norm(p - q) <= a + 1e-3:
                    nodes.append(p)
        if atom['ptype'] == 'prism_body':
            nodes = [min(nodes, key=lambda p: np.linalg.norm(p - q))]
            layers = [layer]
        elif atom['ptype'] == 'prism_face':
            nodes = [min(nodes, key=lambda p: np.linalg.norm(p - q))]
            layers = [layer - 1, layer]
        else:
            layers = [layer - 1, layer]
        for nd in nodes:
            for kk in layers:
                co = hex_corners(a, nd, kk * c, (kk + 1) * c)
                segs = [(co[x], co[y]) for x, y in HEX_EDGES]
                ac = self.plot.add_mesh(lines_polydata(segs), color='#E91E63',
                                        line_width=3)
                self._hl.append(ac)
                fm = self.plot.add_mesh(hexfaces_mesh(co), color='#E91E63',
                                        opacity=0.06)
                self._hl.append(fm)

    def _clear_selection(self):
        for ac in self._hl:
            try: self.plot.remove_actor(ac)
            except Exception: pass
        self._hl = []
        for ac in self._measure_actors:
            try: self.plot.remove_actor(ac)
            except Exception: pass
        self._measure_actors = []
        self._measure_a = None
        if self._atom_actors:
            for ac in self._atom_actors:
                _actor_opacity(ac, 1.0)
        self._sel = None
        self.sel_browser.setHtml('<p>单击 3D 视图中的任意原子，这里会显示'
                                 '<b>均摊分析</b>（该原子被几个晶胞共享）。</p>')

    # 测距
    def _toggle_measure(self):
        self.measure_mode = not self.measure_mode
        self.btn_meas.setText('测距模式：已开启（依次点两个原子）'
                              if self.measure_mode else '测距模式（点两个原子）')

    def _measure_pick(self, idx):
        a = self._atoms[idx]
        mk = pv.Sphere(radius=cc.display_radius(self.cr, a['elem']) *
                       self.radius_scale * 0.35, center=a['cart'],
                       theta_resolution=12, phi_resolution=12)
        ac = self.plot.add_mesh(mk, color='#00E5FF', opacity=0.9)
        self._measure_actors.append(ac)
        if self._measure_a is None:
            self._measure_a = idx
            self.statusBar().showMessage(
                '测距：已选第1个原子(%s)，请点第2个' % a['elem'], 10000)
            return
        b = self._measure_a
        if b == idx:
            self.statusBar().showMessage('两次点击了同一原子', 6000)
            return
        p1 = np.asarray(self._atoms[b]['cart'], float)
        p2 = np.asarray(a['cart'], float)
        d = float(np.linalg.norm(p2 - p1))
        ac2 = self.plot.add_mesh(lines_polydata([(p1, p2)]), color='#00E5FF',
                                 line_width=3)
        self._measure_actors.append(ac2)
        try:
            mid = (p1 + p2) / 2
            self.plot.add_point_labels([tuple(mid)], ['d=%.3f A' % d],
                                       font_size=14, show_points=False,
                                       text_color='#006064')
        except Exception:
            pass
        self.statusBar().showMessage(
            '%s — %s 距离 = %.3f Å' % (self._atoms[b]['elem'], a['elem'], d), 15000)
        self._measure_a = None

    # ---------- 信息面板 ----------
    def _update_sel_html(self, atom):
        e = atom['elem']
        ptype = atom['ptype']
        name, nshare, share = PTYPE_CN[ptype]
        prism = (self.variant == 'prism') and self.cr.get('prism')
        frac = '(%s, %s, %s)' % tuple(_frac_str(v) for v in
                                      (atom['frac'] or (0, 0, 0)))
        word = '个六棱柱' if prism else '个晶胞'
        html = ('<p><b>%s（%s）</b> 位于晶胞<b>%s</b>位置 %s</p>'
                '<p>被 <b>%d</b> %s共享 → 每个晶胞分摊 <b>1/%d</b>'
                '；合计 %d × 1/%d = <b>1</b> 个</p>') % (
                e, cc.elem_name(e), name, frac if not prism else '',
                nshare, word, nshare, nshare, nshare)
        self.sel_browser.setHtml(html)

    def _update_stats(self):
        cr = self.cr
        if cr is None: return
        prism = (self.variant == 'prism') and cr.get('prism')
        try:
            counts = cc.prism_apportioned(cr) if prism else cc.apportioned_counts(cr)
        except Exception:
            counts = {}
        try: f = cc.formula(cr)
        except Exception: f = '?'
        Z = int(round(sum(counts.values()))) if counts else 0
        html = ['<h3>%s</h3>' % cr['name'],
                '<p>%s · %s · a=%.3f b=%.3f c=%.3f Å · α=%.1f β=%.1f γ=%.1f</p>'
                % (cr['system'], cr['lattice'], cr['a'], cr['b'], cr['c'],
                   cr['alpha'], cr['beta'], cr['gamma'])]
        try:
            summ = cc.prism_summary(cr) if prism else cc.cell_atom_summary(cr)
            rows = ''
            for e, d in summ.items():
                parts = []
                for pt, (num, sh) in sorted(d.items(), key=lambda kv: -kv[1][1]):
                    parts.append('%s %d×%s' % (PTYPE_CN[pt][0], num, _frac_str(sh)))
                rows += ('<tr><td>%s（%s）</td><td>%s</td><td>%d</td></tr>'
                         % (e, cc.elem_name(e), ' + '.join(parts),
                            int(round(counts.get(e, 0)))))
            if rows:
                html.append('<table border="1" cellspacing="0" cellpadding="3" '
                            'width="100%"><tr><th>元素</th><th>位置分解'
                            '</th><th>合计</th></tr>' + rows + '</table>')
        except Exception:
            traceback.print_exc()
        html.append('<p><b>化学式：</b><span style="font-size:20px">%s</span>'
                    '　Z = %d</p>' % (f, Z))
        try:
            cn = cc.coordination(cr)
            s = ' · '.join('%s: %d (%s)' % (e, v[0], ''.join(sorted(v[1])))
                           for e, v in cn.items())
            html.append('<p><b>配位数：</b>%s</p>' % s)
        except Exception:
            pass
        try:
            rho = cc.density(cr)
            V = cc.cell_volume(cr)
            html.append('<p><b>密度：</b>ρ = Z·M / (N<sub>A</sub>·V) = '
                        '<b>%.2f g/cm³</b>（V=%.2f Å³）</p>' % (rho, V))
        except Exception:
            pass
        try:
            eta = cc.packing_fraction(cr, scale=self.radius_scale, prism=prism)
            html.append('<p><b>空间利用率：</b>%.1f%%（按当前显示半径）</p>'
                        % (eta * 100))
        except Exception:
            pass
        try:
            md = cc.min_distances(cr)
            s = ' · '.join('%s–%s %.3f Å' % (k[0], k[1], v)
                           for k, v in sorted(md.items())[:4])
            html.append('<p><b>最近接触：</b>%s</p>' % s)
        except Exception:
            pass
        if cr.get('holes'):
            html.append('<p><b>空隙：</b>勾选“空隙标记”——橙色=八面体空隙，'
                        '蓝色=四面体空隙；实心=已被占据。</p>')
        html.append('<hr><p><b>教学要点</b>：%s</p>' % cr.get('notes', ''))
        self.stats_browser.setHtml('\n'.join(html))

    # ---------- 自定义 / CIF / 持久化 ----------
    def _add_row(self, e='C', x=0, y=0, z=0):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        for c, v in enumerate([e, str(x), str(y), str(z)]):
            self.tbl.setItem(r, c, QTableWidgetItem(v))

    def _del_row(self):
        r = self.tbl.currentRow()
        if r >= 0: self.tbl.removeRow(r)

    def _fill_from_current(self):
        if not self.cr: return
        cr = self.cr
        self.ed_name.setText(cr['name'])
        self.sp_a.setValue(cr['a']); self.sp_b.setValue(cr['b'])
        self.sp_c.setValue(cr['c'])
        self.sp_al.setValue(cr['alpha']); self.sp_be.setValue(cr['beta'])
        self.sp_ga.setValue(cr['gamma'])
        self.cmb_cent.setCurrentText(cr['centering'])
        self.tbl.setRowCount(0)
        for e, (x, y, z) in cr['basis']:
            self._add_row(e, x, y, z)

    def _make_custom(self):
        basis = []
        for r in range(self.tbl.rowCount()):
            items = [self.tbl.item(r, c) for c in range(4)]
            if not items[0]: continue
            try:
                e = items[0].text().strip()
                x, y, z = (float(items[c].text()) for c in (1, 2, 3))
                basis.append((e, x, y, z))
            except Exception:
                QMessageBox.warning(self, '输入有误',
                                    '第%d行的原子坐标无法解析，请检查。' % (r + 1))
                return
        if not basis:
            QMessageBox.warning(self, '缺少原子', '请至少添加一个原子。')
            return
        cr = cc.make_custom('usr_%d' % (len(self.custom_db) + 1),
                            self.ed_name.text().strip() or '我的晶体',
                            self.sp_a.value(), self.sp_b.value(),
                            self.sp_c.value(), self.sp_al.value(),
                            self.sp_be.value(), self.sp_ga.value(),
                            self.cmb_cent.currentText(), basis)
        self.custom_db[cr['key']] = cr
        self._populate_tree()
        self.load_crystal(cr)
        self._save_custom(silent=True)
        self.statusBar().showMessage('自定义晶体已生成并保存到我的晶体库', 8000)

    def _import_cif(self):
        fn, _ = QFileDialog.getOpenFileName(self, '选择CIF文件', APP_DIR,
                                            'CIF (*.cif *.CIF);;所有文件 (*)')
        if not fn: return
        try:
            text = open(fn, encoding='utf-8', errors='ignore').read()
            cr = cc.parse_cif(text)
        except Exception as e:
            QMessageBox.critical(self, 'CIF解析失败', str(e))
            return
        cr['key'] = 'cif_%d' % (len(self.custom_db) + 1)
        self.custom_db[cr['key']] = cr
        self._populate_tree()
        self.load_crystal(cr)
        self._save_custom(silent=True)
        self.statusBar().showMessage('CIF 导入成功：%s' % cr['name'], 8000)

    def _save_custom(self, silent=False):
        try:
            data = {}
            for k, cr in self.custom_db.items():
                data[k] = dict(name=cr['name'], a=cr['a'], b=cr['b'], c=cr['c'],
                               alpha=cr['alpha'], beta=cr['beta'],
                               gamma=cr['gamma'], centering=cr['centering'],
                               basis=[[e, x, y, z] for e, (x, y, z)
                                      in cr['basis']],
                               notes=cr.get('notes', ''))
            with open(CUSTOM_JSON, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            if not silent:
                QMessageBox.information(self, '已保存',
                                        '已保存到 我的晶体.json（下次启动自动载入）')
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, '保存失败', str(e))

    def _load_custom(self):
        out = {}
        try:
            if os.path.exists(CUSTOM_JSON):
                data = json.load(open(CUSTOM_JSON, encoding='utf-8'))
                for k, d in data.items():
                    out[k] = cc.make_custom(k, d.get('name', k),
                        d['a'], d['b'], d['c'], d.get('alpha', 90),
                        d.get('beta', 90), d.get('gamma', 90),
                        d.get('centering', 'P'),
                        [(b[0], b[1], b[2], b[3]) for b in d.get('basis', [])],
                        notes=d.get('notes', ''))
        except Exception:
            traceback.print_exc()
        return out

# ---------------- 启动 ----------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName('%s v%s' % (APP_NAME, APP_VERSION))
    def hook(t, v, tb):
        traceback.print_exception(t, v, tb)
        try:
            QMessageBox.critical(None, '程序出错',
                '发生异常，详情见控制台。\n%s: %s' % (t.__name__, v))
        except Exception:
            pass
    sys.excepthook = hook
    _diag('创建主窗口...')
    win = MainWindow()
    win.show()
    _diag('%s v%s 启动完成，总耗时 %.1f 秒'
          % (APP_NAME, APP_VERSION, _time.time() - _T0))
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
