# -*- coding: utf-8 -*-
import os, sys, json, math, traceback, time

APP_NAME = 'Crystal X-Ray'
APP_VERSION = '1.0'
APP_AUTHOR = 'HighspeedG2304'

os.environ.setdefault('QT_API', 'pyside6')
SW_MODE = '--sw' in sys.argv
if SW_MODE:
    os.environ['QT_OPENGL'] = 'software'

import numpy as np

_T0 = time.time()
def _diag(m):
    print('[%6.1fs] %s' % (time.time() - _T0, m), flush=True)
def _die(m):
    print('\n' + m + '\n')
    try: input('按回车退出...')
    except Exception: pass
    sys.exit(1)

if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_JSON = os.path.join(APP_DIR, '我的晶体.json')

_diag('导入 PySide6...')
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QPainter, QColor, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTreeWidget, QTreeWidgetItem, QTabWidget, QGroupBox, QSlider,
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPushButton,
    QTextBrowser, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QHeaderView, QFormLayout, QAbstractItemView, QSizePolicy, QColorDialog,
    QSplashScreen, QRadioButton, QProgressBar, QDialog)
_diag('PySide6 OK')

cc = None; pv = None; vtk = None; QtInteractor = None

PTYPE_COLOR = {'corner': '#E53935', 'edge': '#FB8C00', 'face': '#43A047',
               'body': '#1E88E5', 'prism_vertex': '#E53935',
               'prism_face': '#43A047', 'prism_body': '#1E88E5'}
PTYPE_CN = {'corner': ('顶点', 8, .125), 'edge': ('棱', 4, .25),
            'face': ('面', 2, .5), 'body': ('体内', 1, 1.),
            'prism_vertex': ('顶点', 6, 1/6), 'prism_face': ('面心', 2, .5),
            'prism_body': ('体内', 1, 1.)}
STYLES = ('球棍', '空间填充', '骨棒', '多面体')

def _frac_str(v):
    if abs(v) < 1e-6: return '0'
    for den, s in ((1, ''), (2, '/2'), (3, '/3'), (4, '/4'), (6, '/6'), (8, '/8')):
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

def glyphs_mesh(points, radius, res):
    pts = pv.PolyData(np.asarray(points, float))
    geom = pv.Sphere(radius=float(radius), theta_resolution=res,
                     phi_resolution=res)
    return pts.glyph(geom=geom, scale=False, orient=False)

def lines_polydata(segs):
    pts, seg = [], []
    for a, b in segs:
        pts.append(list(a)); pts.append(list(b))
        seg += [2, len(pts) - 2, len(pts) - 1]
    m = pv.PolyData(np.asarray(pts, float))
    m.lines = np.array(seg, np.int64)
    return m

def cut_sphere(center, radius, planes):
    pd = pv.Sphere(radius=float(radius), center=tuple(center),
                   theta_resolution=18, phi_resolution=18)
    for n, pt in planes:
        plane = vtk.vtkPlane()
        plane.SetNormal(*[float(x) for x in n])
        plane.SetOrigin(*[float(x) for x in pt])
        cl = vtk.vtkClipPolyData()
        cl.SetInputData(pd)
        cl.SetClipFunction(plane)
        cl.InsideOutOff()
        cl.Update()
        out = pv.wrap(cl.GetOutput())
        okside = False
        if out.n_cells and len(out.points):
            c = np.asarray(out.center, float)
            okside = np.dot(c - np.asarray(pt, float),
                            np.asarray(n, float)) < 0
        if not okside:
            cl.InsideOutOn()
            cl.Update()
            out = pv.wrap(cl.GetOutput())
            if out.n_cells == 0:
                return None
        pd = out
    return pd if pd.n_cells else None

CELL_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]
CELL_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5],
              [2, 3, 7, 6], [3, 0, 4, 7]]

def cell_corners(M, i, j, k):
    base = np.array([i, j, k], float) @ M
    return np.array([base + np.array([u, v, w], float) @ M
                     for u, v, w in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))])

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

HEX_EDGES = [(i, (i + 1) % 6) for i in range(6)] + \
            [(i, 6 + (i + 1) % 6) for i in range(6)] + \
            [(i, i + 6) for i in range(6)]
HEX_FACES = [list(range(6)), list(range(6, 12)), [0, 1, 7, 6], [1, 2, 8, 7],
             [2, 3, 9, 8], [3, 4, 10, 9], [4, 5, 11, 10], [5, 0, 6, 11]]

def hexfaces_mesh(corners):
    m = pv.PolyData(np.asarray(corners, float))
    conn = []
    for f in HEX_FACES:
        conn += [len(f)] + f
    m.faces = np.array(conn, np.int64)
    return m

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('%s %s' % (APP_NAME, APP_VERSION))
        self.resize(1420, 900)
        self.cr = None
        self.variant = 'cell'
        self.radius_scale = 1.0
        self.style = '空间填充'
        self.cut_mode = False
        self.show_bonds = False
        self.show_frame = True
        self.show_labels = True
        self.show_faces = False
        self.show_poly = False
        self.show_holes = False
        self.color_by_ptype = False
        self.measure_mode = False
        self.mode = 'cell'
        self.proj = None
        self.slice_hkl = None
        self.slice_off = 0.0
        self._atoms = []; self._atom_actors = []
        self._rmax = 1.0
        self._flat_pos = []
        self._hl = []; self._sel = None
        self._mpts = []; self._m_actors = []
        self._stack_atoms = []; self._stack_r = 1.3
        self.elem_colors = {}
        self.custom_db = self._load_custom()
        self._press_xy = None
        self._build_ui()
        self._setup_pick()
        self._populate_tree()
        self.load_crystal(cc.DB['nacl'])

    # ---------- UI ----------
    def _build_ui(self):
        mb = self.menuBar()
        m = mb.addMenu('帮助')
        m.addAction('关于').triggered.connect(self._about)
        m.addAction('使用说明').triggered.connect(self._usage)
        split = QSplitter(Qt.Horizontal)
        self.setCentralWidget(split)

        left = QWidget(); lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 0, 4)
        self.search = QLineEdit()
        self.search.setPlaceholderText('搜索')
        self.search.textChanged.connect(self._filter_tree)
        lv.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel('晶体资源库')
        self.tree.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tree.header().setSectionResizeMode(QHeaderView.Stretch)
        self.tree.setUniformRowHeights(True)
        self.tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tree.itemDoubleClicked.connect(self._on_tree)
        lv.addWidget(self.tree, 1)
        split.addWidget(left)

        frame = QWidget(); lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        _diag('创建3D窗口...')
        try:
            self.plot = QtInteractor(frame)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(None, '3D初始化失败',
                '渲染窗口创建失败。请用「启动兼容模式.bat」。\n\n%r' % e)
            raise
        _diag('3D窗口OK')
        try:
            rw = getattr(self.plot, 'render_window', None) or self.plot.ren_win
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
        self.tabs.setMinimumWidth(400); self.tabs.setMaximumWidth(480)
        self.tabs.addTab(self._tab_display(), '显示')
        self.tabs.addTab(self._tab_stats(), '统计')
        self.tabs.addTab(self._tab_proj(), '投影')
        self.tabs.addTab(self._tab_tools(), '工具')
        self.tabs.addTab(self._tab_stack(), '堆积')
        self.tabs.addTab(self._tab_custom(), '自定义')
        split.addWidget(self.tabs)
        split.setSizes([250, 790, 410])
        split.setStretchFactor(1, 1)

        bar = QWidget(); hb = QHBoxLayout(bar)
        hb.setContentsMargins(4, 0, 4, 0)
        self.lb_task = QLabel('')
        self.pb = QProgressBar(); self.pb.setFixedWidth(200)
        self.pb.setRange(0, 100); self.pb.setValue(0)
        self.pb.setTextVisible(False)
        hb.addWidget(self.lb_task); hb.addWidget(self.pb)
        self.statusBar().addPermanentWidget(bar)
        self.statusBar().showMessage('左键旋转 滚轮缩放 单击选中')

    def _about(self):
        QMessageBox.about(self, '关于',
            '<h3>%s v%s</h3><p>晶胞演示软件</p><p>开发者：<b>%s</b></p>'
            '<p>93种晶体 · 均摊 · 投影 · 密堆积 · 空隙 · 键角 · d间距 · CIF'
            '</p>' % (APP_NAME, APP_VERSION, APP_AUTHOR))

    def _usage(self):
        QMessageBox.information(self, '使用说明',
            '<b>操作</b><br>左键旋转 · 滚轮缩放 · 中键平移 · 单击原子选中<br>'
            '<b>显示</b>：风格切换/切割分摊(单胞)/剖面/测距(点2个原子出距离，'
            '第3个出键角)<br>'
            '<b>投影</b>：正交视角或压平投影，沿晶向观察；含预设<br>'
            '<b>工具</b>：键长键角表 · 画晶面 · d间距 · 密度互算<br>'
            '<b>堆积</b>：ABAB/ABCABC，自动画晶胞框<br>'
            '<b>自定义</b>：分数坐标建胞 · 导入CIF')

    def _tab_display(self):
        w = QWidget(); v = QVBoxLayout(w)
        g = QGroupBox('晶胞'); v.addWidget(g); f = QFormLayout(g)
        row = QWidget(); h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.sp_nx = QSpinBox(); self.sp_ny = QSpinBox(); self.sp_nz = QSpinBox()
        for sp in (self.sp_nx, self.sp_ny, self.sp_nz):
            sp.setRange(1, 3); sp.setValue(1)
            sp.valueChanged.connect(self._chg)
            h.addWidget(sp)
        f.addRow('n×n×n', row)
        self.sl_r = QSlider(Qt.Horizontal); self.sl_r.setRange(15, 100)
        self.sl_r.setValue(100)
        self.lb_r = QLabel('半径 100%')
        self.sl_r.valueChanged.connect(
            lambda x: self.lb_r.setText('半径 %d%%' % x))
        self.sl_r.sliderReleased.connect(self._chg)
        f.addRow('半径', self.sl_r); f.addRow('', self.lb_r)
        self.cmb_style = QComboBox(); self.cmb_style.addItems(STYLES)
        self.cmb_style.currentTextChanged.connect(self._set_style)
        f.addRow('风格', self.cmb_style)
        self.cmb_var = QComboBox()
        self.cmb_var.addItems(['平行六面体', '六棱柱'])
        self.cmb_var.currentIndexChanged.connect(self._set_variant)
        f.addRow('形式', self.cmb_var)
        g2 = QGroupBox('开关'); v.addWidget(g2); f2 = QFormLayout(g2)
        self._cb_map = {}
        for name, attr in [('化学键', 'show_bonds'), ('晶胞框', 'show_frame'),
                           ('标注', 'show_labels'), ('晶胞面', 'show_faces'),
                           ('多面体', 'show_poly'), ('空隙', 'show_holes'),
                           ('位置着色', 'color_by_ptype')]:
            cb = QCheckBox(name)
            cb.setChecked(getattr(self, attr))
            cb.stateChanged.connect(
                lambda s, a=attr: (setattr(self, a, s == 2), self._chg()))
            self._cb_map[attr] = cb
            f2.addRow(cb)
        self.cb_cut = QCheckBox('切割分摊')
        self.cb_cut.stateChanged.connect(self._set_cut)
        f2.addRow(self.cb_cut)
        g3 = QGroupBox('视图'); v.addWidget(g3); f3 = QFormLayout(g3)
        self.cmb_slice = QComboBox()
        self.cmb_slice.addItems(['无', '(100)', '(110)', '(111)'])
        self.cmb_slice.currentIndexChanged.connect(self._set_slice)
        f3.addRow('切面', self.cmb_slice)
        self.sl_sl = QSlider(Qt.Horizontal); self.sl_sl.setRange(-100, 100)
        self.sl_sl.sliderReleased.connect(self._chg)
        f3.addRow('偏移', self.sl_sl)
        self.btn_meas = QPushButton('测距')
        self.btn_meas.setCheckable(True)
        self.btn_meas.clicked.connect(self._toggle_measure)
        f3.addRow(self.btn_meas)
        rowb = QWidget(); hb2 = QHBoxLayout(rowb)
        hb2.setContentsMargins(0, 0, 0, 0)
        for txt, fn in (('清除', self._clear_all), ('复位', self._reset_cam),
                        ('深色', self._toggle_bg)):
            b = QPushButton(txt); b.clicked.connect(fn); hb2.addWidget(b)
        f3.addRow(rowb)
        self.btn_col = QPushButton('元素颜色…')
        self.btn_col.clicked.connect(self._edit_colors)
        v.addWidget(self.btn_col)
        v.addStretch()
        return w

    def _tab_stats(self):
        w = QWidget(); v = QVBoxLayout(w)
        self.sel_browser = QTextBrowser()
        self.sel_browser.setMaximumHeight(140)
        self.sel_browser.setHtml('<p>点击原子查看均摊</p>')
        v.addWidget(self.sel_browser)
        self.stats_browser = QTextBrowser()
        v.addWidget(self.stats_browser)
        return w

    def _tab_proj(self):
        w = QWidget(); v = QVBoxLayout(w)
        g = QGroupBox('视向'); v.addWidget(g); f = QFormLayout(g)
        self.cmb_dir = QComboBox()
        self.cmb_dir.addItems(['[100]', '[010]', '[001]', '[110]', '[101]',
                               '[111]', '自定义'])
        f.addRow('晶向', self.cmb_dir)
        self.le_uvw = QLineEdit('1 0 0')
        f.addRow('uvw', self.le_uvw)
        rowr = QWidget(); hr = QHBoxLayout(rowr)
        hr.setContentsMargins(0, 0, 0, 0)
        self.rb_ortho = QRadioButton('正交视角')
        self.rb_flat = QRadioButton('压平投影')
        self.rb_ortho.setChecked(True)
        hr.addWidget(self.rb_ortho); hr.addWidget(self.rb_flat)
        f.addRow(rowr)
        rowb = QWidget(); hb = QHBoxLayout(rowb)
        hb.setContentsMargins(0, 0, 0, 0)
        b1 = QPushButton('应用'); b2 = QPushButton('退出投影')
        b1.clicked.connect(self._proj_apply)
        b2.clicked.connect(self._proj_exit)
        hb.addWidget(b1); hb.addWidget(b2)
        f.addRow(rowb)
        g2 = QGroupBox('预设'); v.addWidget(g2); f2 = QFormLayout(g2)
        for txt, key, uvw in (('NaCl [111]', 'nacl', '1 1 1'),
                              ('金刚石 [110]', 'diamond', '1 1 0'),
                              ('CaF₂ [110]', 'caf2', '1 1 0'),
                              ('CsCl [100]', 'cscl', '1 0 0')):
            b = QPushButton(txt)
            b.clicked.connect(lambda _, k=key, u=uvw: self._preset(k, u))
            f2.addRow(b)
        self.lb_proj = QLabel('')
        v.addWidget(self.lb_proj)
        v.addStretch()
        return w

    def _tab_tools(self):
        w = QWidget(); v = QVBoxLayout(w)
        g = QGroupBox('键长键角（当前晶体）'); v.addWidget(g)
        gv = QVBoxLayout(g)
        self.tool_browser = QTextBrowser()
        self.tool_browser.setMaximumHeight(170)
        gv.addWidget(self.tool_browser)
        g2 = QGroupBox('晶面'); v.addWidget(g2); f2 = QFormLayout(g2)
        row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
        self.sp_fh = QSpinBox(); self.sp_fk = QSpinBox(); self.sp_fl = QSpinBox()
        for sp in (self.sp_fh, self.sp_fk, self.sp_fl):
            sp.setRange(-5, 5); h.addWidget(sp)
        self.sp_fh.setValue(1)
        f2.addRow('h k l', row)
        b = QPushButton('画晶面')
        b.clicked.connect(self._draw_plane)
        f2.addRow(b)
        g3 = QGroupBox('d 间距'); v.addWidget(g3); f3 = QFormLayout(g3)
        self.dsp = {}
        for lbl in ('a', 'b', 'c', 'α', 'β', 'γ'):
            sp = QDoubleSpinBox(); sp.setRange(0.1, 99); sp.setDecimals(3)
            sp.setValue(5.640 if lbl in ('a', 'b', 'c') else 90.0)
            self.dsp[lbl] = sp
        rowa = QWidget(); ha = QHBoxLayout(rowa); ha.setContentsMargins(0, 0, 0, 0)
        for lbl in ('a', 'b', 'c'):
            ha.addWidget(self.dsp[lbl])
        rowb = QWidget(); hb = QHBoxLayout(rowb); hb.setContentsMargins(0, 0, 0, 0)
        for lbl in ('α', 'β', 'γ'):
            hb.addWidget(self.dsp[lbl])
        f3.addRow('abc', rowa); f3.addRow('角度', rowb)
        rowh = QWidget(); hh = QHBoxLayout(rowh); hh.setContentsMargins(0, 0, 0, 0)
        self.sp_dh = QSpinBox(); self.sp_dk = QSpinBox(); self.sp_dl = QSpinBox()
        for sp in (self.sp_dh, self.sp_dk, self.sp_dl):
            sp.setRange(-5, 5); hh.addWidget(sp)
        self.sp_dh.setValue(1)
        f3.addRow('h k l', rowh)
        rowd = QWidget(); hd = QHBoxLayout(rowd); hd.setContentsMargins(0, 0, 0, 0)
        bd = QPushButton('计算 d')
        bd.clicked.connect(self._calc_d)
        self.lb_d = QLabel('')
        hd.addWidget(bd); hd.addWidget(self.lb_d)
        f3.addRow(rowd)
        g4 = QGroupBox('密度（立方）'); v.addWidget(g4); f4 = QFormLayout(g4)
        self.sp_da = QDoubleSpinBox(); self.sp_da.setRange(0.1, 99)
        self.sp_da.setDecimals(3); self.sp_da.setValue(5.0)
        self.sp_dm = QDoubleSpinBox(); self.sp_dm.setRange(1, 9999)
        self.sp_dm.setValue(58.44)
        self.sp_dz = QSpinBox(); self.sp_dz.setRange(1, 99); self.sp_dz.setValue(4)
        f4.addRow('a (Å)', self.sp_da); f4.addRow('M (g/mol)', self.sp_dm)
        f4.addRow('Z', self.sp_dz)
        rowe = QWidget(); he = QHBoxLayout(rowe); he.setContentsMargins(0, 0, 0, 0)
        b1 = QPushButton('算 ρ'); b2 = QPushButton('算 a')
        b1.clicked.connect(self._calc_rho)
        b2.clicked.connect(self._calc_a)
        self.lb_rho = QLabel('')
        he.addWidget(b1); he.addWidget(b2)
        f4.addRow(rowe); f4.addRow('', self.lb_rho)
        return w

    def _tab_stack(self):
        w = QWidget(); v = QVBoxLayout(w)
        g = QGroupBox('密置层堆积'); v.addWidget(g); f = QFormLayout(g)
        self.cmb_seq = QComboBox()
        self.cmb_seq.addItems(['ABAB…', 'ABCABC…', '自定义'])
        self.cmb_seq.currentIndexChanged.connect(self._stack_chg)
        f.addRow('序列', self.cmb_seq)
        self.ed_seq = QLineEdit('ABCB')
        f.addRow('自定义', self.ed_seq)
        self.sp_layers = QSpinBox(); self.sp_layers.setRange(1, 8)
        self.sp_layers.setValue(4)
        self.sp_layers.valueChanged.connect(self._stack_chg)
        f.addRow('层数', self.sp_layers)
        self.cb_lcolor = QCheckBox('按层着色')
        self.cb_lcolor.setChecked(True)
        self.cb_lcolor.stateChanged.connect(lambda s: self._stack_chg())
        f.addRow(self.cb_lcolor)
        self.cb_sframe = QCheckBox('晶胞框')
        self.cb_sframe.setChecked(True)
        self.cb_sframe.stateChanged.connect(lambda s: self._stack_chg())
        f.addRow(self.cb_sframe)
        b = QPushButton('生成堆积')
        b.clicked.connect(self._stack_enter)
        f.addRow(b)
        self.lb_seq = QLabel('')
        f.addRow(self.lb_seq)
        info = QTextBrowser()
        info.setHtml('<table border="1" cellspacing="0" cellpadding="3">'
            '<tr><th>堆积</th><th>格子</th><th>配位</th><th>η</th>'
            '<th>实例</th></tr>'
            '<tr><td>简单立方</td><td>P</td><td>6</td><td>52.36%</td><td>Po</td></tr>'
            '<tr><td>体心立方</td><td>I</td><td>8</td><td>68.02%</td>'
            '<td>Na K Fe</td></tr>'
            '<tr><td>六方最密</td><td>六方P</td><td>12</td><td>74.05%</td>'
            '<td>Mg Zn Ti</td></tr>'
            '<tr><td>面心最密</td><td>F</td><td>12</td><td>74.05%</td>'
            '<td>Cu Ag Au</td></tr></table>'
            '<p>点球显示配位数</p>')
        v.addWidget(info)
        return w

    def _tab_custom(self):
        w = QWidget(); v = QVBoxLayout(w)
        g = QGroupBox('自定义晶胞'); v.addWidget(g); f = QFormLayout(g)
        self.ed_name = QLineEdit('我的晶体')
        f.addRow('名称', self.ed_name)
        self.sp_a = QDoubleSpinBox(); self.sp_b = QDoubleSpinBox()
        self.sp_c = QDoubleSpinBox()
        for sp in (self.sp_a, self.sp_b, self.sp_c):
            sp.setRange(1.0, 50.0); sp.setDecimals(3)
        self.sp_a.setValue(5.0); self.sp_b.setValue(5.0); self.sp_c.setValue(5.0)
        row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.sp_a); h.addWidget(self.sp_b); h.addWidget(self.sp_c)
        f.addRow('a b c (Å)', row)
        self.sp_al = QDoubleSpinBox(); self.sp_be = QDoubleSpinBox()
        self.sp_ga = QDoubleSpinBox()
        for sp in (self.sp_al, self.sp_be, self.sp_ga):
            sp.setRange(20.0, 160.0); sp.setDecimals(2); sp.setValue(90.0)
        row2 = QWidget(); h2 = QHBoxLayout(row2); h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(self.sp_al); h2.addWidget(self.sp_be); h2.addWidget(self.sp_ga)
        f.addRow('α β γ (°)', row2)
        self.cmb_cent = QComboBox()
        self.cmb_cent.addItems(['P', 'I', 'F', 'A', 'B', 'C', 'R'])
        f.addRow('格子', self.cmb_cent)
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(['元素', 'x', 'y', 'z'])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setMaximumHeight(190)
        v.addWidget(self.tbl)
        rowb = QWidget(); hb = QHBoxLayout(rowb); hb.setContentsMargins(0, 0, 0, 0)
        for txt, fn in (('添加', self._add_row), ('删除', self._del_row),
                        ('模板', self._fill_from_current)):
            b = QPushButton(txt); b.clicked.connect(fn); hb.addWidget(b)
        v.addWidget(rowb)
        rowb2 = QWidget(); hb2 = QHBoxLayout(rowb2); hb2.setContentsMargins(0, 0, 0, 0)
        for txt, fn in (('生成', self._make_custom), ('导入CIF…', self._import_cif),
                        ('保存', self._save_custom)):
            b = QPushButton(txt); b.clicked.connect(fn); hb2.addWidget(b)
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
        items = list(cc.CRYSTAL_DB) + list(self.custom_db.values())
        cats = {}
        for cr in items:
            cats.setdefault(cr['category'], []).append(cr)
        for cat in ('金属晶体', '离子晶体', '共价晶体', '分子晶体', '自定义'):
            if cat not in cats:
                continue
            top = QTreeWidgetItem([cat])
            protos = {}
            for cr in cats[cat]:
                protos.setdefault(cr.get('prototype') or '其他', []).append(cr)
            for proto, lst in sorted(protos.items()):
                if len(protos) == 1:
                    holders = [top]
                else:
                    holders = [QTreeWidgetItem([proto])]
                    top.addChild(holders[0])
                for cr in lst:
                    it = QTreeWidgetItem([cr['name']])
                    it.setData(0, Qt.UserRole, cr['key'])
                    fs = cc.formula(cr)
                    it.setData(0, Qt.UserRole + 1,
                               (cr['name'] + ' ' + fs + ' ' +
                                ' '.join(e for e, _ in cr['basis'])).lower())
                    it.setToolTip(0, fs)
                    holders[0].addChild(it)
            self.tree.addTopLevelItem(top)
            top.setExpanded(cat in ('金属晶体', '离子晶体'))

    def _filter_tree(self, text):
        t = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            any_cat = False
            for pi in range(top.childCount()):
                node = top.childItem(pi) if False else top.child(pi)
                any_node = False
                for ci in range(node.childCount()):
                    it = node.child(ci)
                    vis = (not t) or (t in (it.data(0, Qt.UserRole + 1) or ''))
                    it.setHidden(not vis)
                    any_node = any_node or vis
                if top.childCount() and node.childCount() == 0:
                    vis = (not t) or (t in (node.data(0, Qt.UserRole + 1) or ''))
                    node.setHidden(not vis)
                    any_node = vis
                else:
                    node.setHidden(not any_node)
                any_cat = any_cat or any_node
            top.setHidden(not any_cat)

    def _on_tree(self, item, col):
        key = item.data(0, Qt.UserRole)
        if not key:
            return
        cr = cc.DB.get(key) or self.custom_db.get(key)
        if cr:
            self.load_crystal(cr)

    # ---------- 载入 / 任务进度 ----------
    def _task(self, pct, msg):
        self.pb.setValue(pct)
        self.lb_task.setText(msg)
        QApplication.processEvents()

    def _task_end(self):
        self.pb.setValue(0)
        self.lb_task.setText('')

    def load_crystal(self, cr):
        self.cr = cr
        self.mode = 'cell'
        self.proj = None
        self._clear_all()
        self.cmb_var.setEnabled(bool(cr.get('prism')) and not self.cut_mode)
        if not cr.get('prism'):
            self.cmb_var.setCurrentIndex(0)
            self.variant = 'cell'
        self.rebuild()
        self.statusBar().showMessage('已载入 %s' % cr['name'], 5000)

    def _chg(self):
        if self.mode == 'cell' and not self.cut_mode:
            self.rebuild()

    def _set_style(self, s):
        self.style = s
        auto_bonds = s in ('球棍', '骨棒')
        cb = self._cb_map.get('show_bonds')
        if cb is not None:
            cb.blockSignals(True)
            cb.setChecked(auto_bonds)
            cb.blockSignals(False)
        self.show_bonds = auto_bonds
        if self.mode == 'cell' and not self.cut_mode:
            self.rebuild()


    def _find_cbs(self):
        return []

    def _set_variant(self, i):
        self.variant = 'cell' if i == 0 else 'prism'
        if self.variant == 'prism' and self.cut_mode:
            self.cb_cut.setChecked(False)
        self.rebuild()

    def _set_slice(self, i):
        self.slice_hkl = [None, (1, 0, 0), (1, 1, 0), (1, 1, 1)][i]
        self._chg()

    def _set_cut(self, s):
        self.cut_mode = (s == 2)
        for sp in (self.sp_nx, self.sp_ny, self.sp_nz):
            sp.blockSignals(True)
            if self.cut_mode:
                sp.setValue(1)
            sp.setEnabled(not self.cut_mode)
            sp.blockSignals(False)
        self.cmb_var.setEnabled(bool(self.cr.get('prism')) and not self.cut_mode)
        self.rebuild()

    def _reset_cam(self):
        try:
            self.plot.reset_camera()
            self.plot.isometric_view()
        except Exception:
            pass

    def _toggle_bg(self):
        self._bg = not getattr(self, '_bg', False)
        try:
            self.plot.set_background('#20242B' if self._bg else '#F5F7FA')
        except Exception:
            pass

    def _toggle_measure(self):
        self.measure_mode = self.btn_meas.isChecked()
        self._mpts = []
        if self.measure_mode:
            self.statusBar().showMessage('点两个原子出距离，第3个出键角', 8000)

    def _clear_all(self):
        for ac in self._hl + self._m_actors:
            try: self.plot.remove_actor(ac)
            except Exception: pass
        self._hl = []; self._m_actors = []
        self._sel = None; self._mpts = []
        for ac in self._atom_actors:
            _actor_opacity(ac, 1.0)
        self.sel_browser.setHtml('<p>点击原子查看均摊</p>')

    def rebuild(self):
        try:
            if self.mode == 'stack':
                self._rebuild_stack()
            else:
                self._rebuild_cell()
        except Exception:
            traceback.print_exc()
            self.statusBar().showMessage('渲染出错，见控制台', 8000)

    def _clear_scene(self):
        try: self.plot.clear()
        except Exception: pass
        self._atom_actors = []
        self._hl = []; self._m_actors = []
        self._sel = None; self._mpts = []
        self._flat_pos = []

    def _slice_clip(self, mesh):
        if self.slice_hkl is None or mesh is None or mesh.n_cells == 0:
            return mesh
        try:
            M = cc.lattice_matrix(self.cr)
            h, k, l = self.slice_hkl
            n = h * np.cross(M[1], M[2]) + k * np.cross(M[2], M[0]) + \
                l * np.cross(M[0], M[1])
            n = n / np.linalg.norm(n)
            center = np.array([max(1, self.sp_nx.value()) / 2.,
                               max(1, self.sp_ny.value()) / 2.,
                               max(1, self.sp_nz.value()) / 2.]) @ M
            extent = np.abs(center).max() * 2
            org = center + n * (self.slice_off * extent)
            out = mesh.clip(normal=tuple(n), origin=tuple(org))
            return out if out is not None and out.n_cells > 0 else None
        except Exception:
            traceback.print_exc()
            return mesh

    def _supercell(self, cr, nx, ny, nz):
        M = cc.lattice_matrix(cr)
        atoms, seen = [], set()
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for e, f in cc.unit_sites(cr):
                        for img in cc.closed_cell_images(f):
                            s = (i + img[0], j + img[1], k + img[2])
                            cart = M.T @ np.array(s)
                            key = tuple(np.round(cart, 4))
                            if key in seen:
                                continue
                            seen.add(key)
                            pt = cc.pos_type(f)
                            atoms.append(dict(
                                elem=e, xyz=s, cart=cart, cell=(i, j, k),
                                frac=f, ptype=pt, share=cc.POS_TYPE[pt][2],
                                nshare=cc.POS_TYPE[pt][1]))
        return atoms

    def _rebuild_cell(self):
        cr = self.cr
        if cr is None:
            return
        self._clear_scene()
        self._task(5, '计算原子')
        prism = (self.variant == 'prism') and cr.get('prism')
        if prism:
            nc = max(self.sp_nx.value(), self.sp_ny.value(),
                     self.sp_nz.value())
            atoms = cc.prism_atoms(cr, nc)
        else:
            atoms = self._supercell(cr, self.sp_nx.value(),
                                    self.sp_ny.value(), self.sp_nz.value())
        self._atoms = atoms
        rf = {'球棍': .30, '空间填充': 1.0, '骨棒': .12, '多面体': .30}[self.style]
        if self.cut_mode:
            rf = 1.0
        radii = [cc.display_radius(cr, a['elem']) * rf * self.radius_scale
                 for a in atoms]
        self._rmax = max(radii) if radii else 1.0
        na = len(atoms)
        res = 20 if na <= 120 else (14 if na <= 320 else 10)
        if SW_MODE:
            res = max(8, res - 6)
        self._task(30, '构建模型')
        if self.cut_mode and not prism:
            planes_all = cc.atom_cut_planes(atoms, 1, cr)
            pieces = {}
            for i, a in enumerate(atoms):
                pc = cut_sphere(a['cart'], radii[i], planes_all[i])
                if pc is not None:
                    pieces.setdefault(a['elem'], []).append(pc)
            for e, lst in pieces.items():
                app = vtk.vtkAppendPolyData()
                for pc in lst:
                    app.AddInputData(pc)
                app.Update()
                mesh = self._slice_clip(pv.wrap(app.GetOutput()))
                if mesh is None or mesh.n_cells == 0:
                    continue
                col = self.elem_colors.get(e) or cc.elem_color(e)
                ac = self.plot.add_mesh(mesh, color=col, ambient=.22,
                                        diffuse=.85, specular=.3)
                self._atom_actors.append(ac)
        else:
            groups = {}
            for i, a in enumerate(atoms):
                key = a['ptype'] if self.color_by_ptype else a['elem']
                groups.setdefault(key, []).append(i)
            for key, idxs in groups.items():
                pts = np.array([atoms[i]['cart'] for i in idxs], float)
                r0 = float(np.mean([radii[i] for i in idxs]))
                mesh = glyphs_mesh(pts, r0, res)
                mesh = self._slice_clip(mesh)
                if mesh is None or mesh.n_cells == 0:
                    continue
                if self.color_by_ptype:
                    col = PTYPE_COLOR.get(key, '#90A4AE')
                else:
                    col = self.elem_colors.get(key) or cc.elem_color(key)
                ac = self.plot.add_mesh(mesh, color=col, ambient=.22,
                                        diffuse=.85, specular=.35)
                self._atom_actors.append(ac)
        self._task(55, '化学键')
        if self.show_bonds and not self.cut_mode:
            try:
                pairs = cc.bonds_for(cr, atoms)
                if pairs:
                    a0 = cr['a']
                    rb = (0.075 if self.style == '骨棒' else 0.05) * a0
                    segs = [(atoms[i]['cart'], atoms[j]['cart'])
                            for i, j in pairs]
                    tm = self._slice_clip(
                        lines_polydata(segs).tube(radius=rb, n_sides=8))
                    if tm is not None and tm.n_cells:
                        self.plot.add_mesh(tm, color='#546E7A')
            except Exception:
                traceback.print_exc()
        self._task(70, '晶胞框')
        M = cc.lattice_matrix(cr)
        if self.show_frame:
            segs = []
            if prism:
                a, c0 = cr['a'], cr['c']
                for ctr in cc.prism_cells(cr, nc):
                    for kk in range(nc):
                        co = hex_corners(a, ctr, kk * c0, (kk + 1) * c0)
                        segs += [(co[x], co[y]) for x, y in HEX_EDGES]
                        if self.show_faces:
                            self.plot.add_mesh(hexfaces_mesh(co),
                                               color='#78909C', opacity=.05)
            else:
                for i in range(self.sp_nx.value()):
                    for j in range(self.sp_ny.value()):
                        for k in range(self.sp_nz.value()):
                            co = cell_corners(M, i, j, k)
                            segs += [(co[x], co[y]) for x, y in CELL_EDGES]
                            if self.show_faces:
                                self.plot.add_mesh(quadfaces_mesh(co),
                                                   color='#78909C', opacity=.05)
            if segs:
                self.plot.add_mesh(lines_polydata(segs), color='#37474F',
                                   line_width=2)
        self._task(80, '标注')
        if self.show_labels and 0 < na <= 64:
            try:
                self.plot.add_point_labels(
                    [tuple(a['cart']) for a in atoms],
                    [a['elem'] for a in atoms], font_size=12, show_points=False)
            except Exception:
                pass
        if self.show_holes and cr.get('holes'):
            self._add_holes(cr, M)
        if self.show_poly and not self.cut_mode:
            self._add_polyhedra(cr, atoms)
        try:
            self.plot.add_text('%s v%s | %s | %d×%d×%d%s' % (
                APP_NAME, APP_VERSION, cr['key'], self.sp_nx.value(),
                self.sp_ny.value(), self.sp_nz.value(),
                ' 六棱柱' if prism else ''), position='upper_left', font_size=10)
        except Exception:
            pass
        self._proj_exit_cam()
        self._task(92, '统计')
        self._update_stats()
        self._update_tools()
        self._task(100, '完成')
        self._task_end()

    def _add_holes(self, cr, M):
        octa, tetra = cc.fcc_holes(cr)
        rm = .07 * cr['a']
        for lst, empty_col in ((octa, '#FF8000'), (tetra, '#00A6D6')):
            occ_pts, emp_pts = [], []
            for f, occ in lst:
                for i in range(self.sp_nx.value()):
                    for j in range(self.sp_ny.value()):
                        for k in range(self.sp_nz.value()):
                            p = np.array([i + f[0], j + f[1], k + f[2]]) @ M
                            (occ_pts if occ else emp_pts).append(p)
            for pts, col in ((occ_pts, None), (emp_pts, empty_col)):
                if not pts:
                    continue
                m = glyphs_mesh(pts, rm, 10)
                if occ and col is None:
                    col = self.elem_colors.get(occ) or cc.elem_color(occ)
                self.plot.add_mesh(m, color=col, opacity=.85)

    def _add_polyhedra(self, cr, atoms):
        pos = np.array([a['cart'] for a in atoms])
        for ph in cc.coordination_polyhedra(cr, atoms):
            vs = pos[ph['verts']]
            cen = vs.mean(axis=0)
            vs2 = cen + .8 * (vs - cen)
            faces = cc.convex_hull_faces(vs2)
            if not faces:
                continue
            m = pv.PolyData(vs2)
            conn = []
            for fa in faces:
                conn += [3] + list(fa)
            m.faces = np.array(conn, np.int64)
            e = atoms[ph['center']]['elem']
            self.plot.add_mesh(m, color=self.elem_colors.get(e)
                               or cc.elem_color(e), opacity=.28,
                               show_edges=True, edge_color='#263238')

    # ---------- 堆积 ----------
    def _stack_enter(self):
        self.mode = 'stack'
        self._rebuild_stack()

    def _stack_seq_string(self):
        i = self.cmb_seq.currentIndex()
        L = self.sp_layers.value()
        if i == 0:
            pat = 'AB'
        elif i == 1:
            pat = 'ABC'
        else:
            pat = ''.join(c for c in self.ed_seq.text().upper()
                          if c in 'ABC') or 'AB'
        return (pat * 20)[:L]

    def _stack_chg(self):
        if self.mode == 'stack':
            self._rebuild_stack()

    def _rebuild_stack(self):
        self._clear_scene()
        self._task(20, '堆积')
        seq = self._stack_seq_string()
        L = self.sp_layers.value()
        self.lb_seq.setText('%s · %d层' % (seq, L))
        atoms = cc.build_stack(seq, r=self._stack_r, extent=4)
        self._stack_atoms = atoms
        r = self._stack_r
        res = 10 if SW_MODE else 16
        groups = {}
        for ai, a in enumerate(atoms):
            key = a['letter'] if self.cb_lcolor.isChecked() else 'X'
            groups.setdefault(key, []).append(ai)
        colmap = {'A': '#E53935', 'B': '#43A047', 'C': '#1E88E5', 'X': '#78909C'}
        for key, idxs in groups.items():
            pts = np.array([atoms[i]['cart'] for i in idxs], float)
            m = glyphs_mesh(pts, r, res)
            self.plot.add_mesh(m, color=colmap.get(key, '#78909C'),
                               ambient=.22, diffuse=.85, specular=.35)
        self._task(70, '晶胞框')
        if self.cb_sframe.isChecked():
            for k in range(0, max(0, L - 3)):
                corners, ok = cc.fcc_cube_in_stack(seq, r, k0=k)
                if ok:
                    segs = [(corners[x], corners[y]) for x, y in
                            ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
                             (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7))]
                    self.plot.add_mesh(lines_polydata(segs), color='#0277BD',
                                       line_width=3)
                    break
            for k in range(0, max(0, L - 2)):
                hp, ok = cc.hcp_prism_in_stack(seq, r, k0=k)
                if ok:
                    hexv, z0, z1 = hp
                    co = hex_corners(2 * r, (0, 0), z0, z1)
                    segs = [(co[x], co[y]) for x, y in HEX_EDGES]
                    self.plot.add_mesh(lines_polydata(segs), color='#C2185B',
                                       line_width=3)
                    break
        try:
            self.plot.add_text('%s | %s %d层' % (APP_NAME, seq, L),
                               position='upper_left', font_size=10)
        except Exception:
            pass
        self._proj_exit_cam()
        self._task(100, '完成')
        self._task_end()

    # ---------- 投影 ----------
    def _uvw(self):
        t = self.cmb_dir.currentText()
        if t == '自定义':
            try:
                v = [int(x) for x in self.le_uvw.text().replace(',', ' ').split()]
                if len(v) == 3 and any(v):
                    return tuple(v)
            except Exception:
                pass
            return None
        return {('[100]'): (1, 0, 0), ('[010]'): (0, 1, 0),
                ('[001]'): (0, 0, 1), ('[110]'): (1, 1, 0),
                ('[101]'): (1, 0, 1), ('[111]'): (1, 1, 1)}[t]

    def _proj_apply(self):
        uvw = self._uvw()
        if uvw is None:
            self.statusBar().showMessage('晶向无效', 5000)
            return
        if self.rb_ortho.isChecked():
            self._ortho_view(uvw)
        else:
            self._rebuild_flat(uvw)

    def _proj_exit(self):
        self.proj = None
        if self.mode == 'cell':
            self.rebuild()
        else:
            self._rebuild_stack()

    def _proj_exit_cam(self):
        self.proj = None
        try:
            self.plot.disable_parallel_projection()
        except Exception:
            pass

    def _view_frame(self, cr, uvw):
        n = cc.view_direction(cr, uvw)
        tmp = np.array([1.0, 0, 0]) if abs(n[0]) < .9 else np.array([0, 1.0, 0])
        e1 = np.cross(n, tmp); e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        return n, e1, e2

    def _ortho_view(self, uvw):
        cr = self.cr
        n, e1, e2 = self._view_frame(cr, uvw)
        b = self.plot.bounds
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2,
                           (b[4] + b[5]) / 2])
        dist = max(b[1] - b[0], b[3] - b[2], b[5] - b[4]) * 2.2
        up = np.array([0, 0, 1.0]) if abs(n[2]) < .9 else np.array([1.0, 0, 0])
        up = up - np.dot(up, n) * n
        up = up / np.linalg.norm(up)
        try:
            self.plot.enable_parallel_projection()
        except Exception:
            pass
        self.plot.camera_position = [tuple(center - n * dist), tuple(center),
                                     tuple(up)]
        self.statusBar().showMessage('沿 [%d%d%d] 正交观察' % uvw, 5000)

    def _rebuild_flat(self, uvw):
        cr = self.cr
        if cr is None or self.mode != 'cell':
            return
        self._clear_scene()
        self._task(20, '投影')
        n, e1, e2 = self._view_frame(cr, uvw)
        atoms = self._atoms
        pos, depth = [], []
        for a in atoms:
            p = np.asarray(a['cart'], float)
            x, y, z = float(p @ e1), float(p @ e2), float(p @ n)
            pos.append(x * e1 + y * e2)
            depth.append(z)
        pos = np.array(pos); depth = np.array(depth)
        rr = .38 * min(cc.display_radius(cr, e)
                       for e, _ in cr['basis']) * self.radius_scale
        res = 12 if SW_MODE else 16
        m = glyphs_mesh(pos, rr, res)
        m['depth'] = np.repeat(depth, m.n_points // max(1, len(depth)))
        self._task(60, '渲染')
        if 'depth' in m.point_data:
            self.plot.add_mesh(m, scalars='depth', cmap='viridis',
                               show_scalar_bar=True,
                               scalar_bar_args=dict(title='z / Å'))
        else:
            self.plot.add_mesh(m, color='#78909C')
        agg = {}
        for i, a in enumerate(atoms):
            key = (round(float(pos[i] @ e1), 2), round(float(pos[i] @ e2), 2),
                   a['elem'])
            if key not in agg:
                agg[key] = [0, depth[i], pos[i]]
            agg[key][0] += 1
        labels, lpts = [], []
        for (x, y, e), (cnt, d, p) in agg.items():
            labels.append('%s%s z=%.2f' % (e, ('×%d' % cnt) if cnt > 1 else '', d))
            lpts.append(tuple(p + n * (rr * 2.5)))
        try:
            self.plot.add_point_labels(lpts, labels, font_size=11,
                                       show_points=False, always_visible=True)
        except Exception:
            pass
        self._flat_pos = list(pos)
        b = self.plot.bounds
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2,
                           (b[4] + b[5]) / 2])
        dist = max(b[1] - b[0], b[3] - b[2], b[5] - b[4]) * 2.5
        try:
            self.plot.enable_parallel_projection()
        except Exception:
            pass
        self.plot.camera_position = [tuple(center - n * dist), tuple(center),
                                     tuple(e2)]
        self.proj = dict(uvw=uvw, flat=True)
        self.lb_proj.setText('[%d%d%d] 压平 · %d原子' % (uvw + (len(atoms),)))
        try:
            self.plot.add_text('[%d%d%d] 投影' % uvw, position='upper_left',
                               font_size=10)
        except Exception:
            pass
        self._task(100, '完成')
        self._task_end()

    def _preset(self, key, uvw):
        cr = cc.DB.get(key)
        if not cr:
            return
        self.load_crystal(cr)
        self.rb_flat.setChecked(True)
        self.le_uvw.setText(' '.join(uvw.split()))
        self.cmb_dir.setCurrentText(''.join(('[%s]' % uvw.replace(' ', ''))))
        self._rebuild_flat(tuple(int(x) for x in uvw.split()))

    # ---------- 拾取 ----------
    def _setup_pick(self):
        try:
            self._wpp = vtk.vtkWorldPointPicker()
        except Exception:
            self._wpp = None
        try:
            self.plot.installEventFilter(self)
        except Exception:
            traceback.print_exc()
        _diag('拾取就绪')

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
                if dx + dy < 6:
                    self._do_pick(p.x(), p.y())
        return super().eventFilter(obj, event)

    def _vtk_xy(self, x_qt, y_qt):
        try:
            rw = getattr(self.plot, 'render_window', None) or self.plot.ren_win
            _, h = rw.GetSize()
            base = self.plot.height()
            scale = (h / base) if (h and base) else 1.0
            return int(round(x_qt * scale)), int(round((base - y_qt) * scale))
        except Exception:
            return int(x_qt), int(y_qt)

    def _do_pick(self, x_qt, y_qt):
        if self._wpp is None or self.proj is not None:
            return
        x, y = self._vtk_xy(x_qt, y_qt)
        try:
            self._wpp.Pick(x, y, 0, self.plot.renderer)
            p = np.asarray(self._wpp.GetPickPosition(), float)
        except Exception:
            return
        if self.mode == 'stack':
            pts = np.array([a['cart'] for a in self._stack_atoms], float)
            if not len(pts):
                return
            d = np.linalg.norm(pts - p, axis=1)
            i = int(np.argmin(d))
            if d[i] > self._stack_r * 1.4:
                return
            r = self._stack_r
            dd = np.linalg.norm(pts - pts[i], axis=1)
            ntouch = int(np.sum((dd > 1e-6) & (dd <= 2 * r * 1.03)))
            a = self._stack_atoms[i]
            self.statusBar().showMessage(
                '第%d层(%s)：配位数 %d' % (a['layer'] + 1, a['letter'], ntouch),
                10000)
            return
        if not self._atoms:
            return
        pts = np.array([a['cart'] for a in self._atoms], float)
        d = np.linalg.norm(pts - p, axis=1)
        i = int(np.argmin(d))
        if d[i] > self._rmax * 1.4 + .05:
            if not self.measure_mode:
                self._clear_all()
            return
        if self.measure_mode:
            self._measure_pick(i)
        else:
            self._show_selection(i)

    # ---------- 选中 / 均摊 ----------
    def _show_selection(self, idx):
        atom = self._atoms[idx]
        self._sel = idx
        for ac in self._hl:
            try: self.plot.remove_actor(ac)
            except Exception: pass
        self._hl = []
        for ac in self._atom_actors:
            _actor_opacity(ac, .22)
        cr = self.cr
        rf = 1.0 if self.cut_mode else \
            {'球棍': .30, '空间填充': 1.0, '骨棒': .12, '多面体': .30}[self.style]
        r = cc.display_radius(cr, atom['elem']) * rf * self.radius_scale
        shell = pv.Sphere(radius=r * 1.15, center=atom['cart'],
                          theta_resolution=24, phi_resolution=24)
        self._hl.append(self.plot.add_mesh(shell, color='#FFD54F', opacity=.55))
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
        import itertools
        cells = [(i, j, k)]
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
            self._hl.append(self.plot.add_mesh(
                lines_polydata([(co[x], co[y]) for x, y in CELL_EDGES]),
                color='#E91E63', line_width=3))
            self._hl.append(self.plot.add_mesh(
                quadfaces_mesh(co), color='#E91E63', opacity=.06))

    def _hl_prism_frames(self, atom):
        cr = self.cr
        a, c = cr['a'], cr['c']
        w1 = np.array([1.5 * a, a * math.sqrt(3) / 2])
        w2 = np.array([1.5 * a, -a * math.sqrt(3) / 2])
        q = np.array(atom['cart'][:2], float)
        z = atom['cart'][2]
        layer = int(round(z / c))
        nodes = []
        K = 3 * max(self.sp_nx.value(), self.sp_nz.value()) + 3
        for m in range(-K, K + 1):
            for n_ in range(-K, K + 1):
                p = m * w1 + n_ * w2
                if np.linalg.norm(p - q) <= a + 1e-3:
                    nodes.append(p)
        if not nodes:
            return
        if atom['ptype'] == 'prism_body':
            nodes = [min(nodes, key=lambda p: np.linalg.norm(p - q))]
            layers = [layer]
        else:
            nodes = [min(nodes, key=lambda p: np.linalg.norm(p - q))]
            layers = [layer - 1, layer]
        for nd in nodes:
            for kk in layers:
                co = hex_corners(a, nd, kk * c, (kk + 1) * c)
                self._hl.append(self.plot.add_mesh(
                    lines_polydata([(co[x], co[y]) for x, y in HEX_EDGES]),
                    color='#E91E63', line_width=3))
                self._hl.append(self.plot.add_mesh(
                    hexfaces_mesh(co), color='#E91E63', opacity=.06))

    def _update_sel_html(self, atom):
        e = atom['elem']
        name, nshare, share = PTYPE_CN[atom['ptype']]
        prism = (self.variant == 'prism') and self.cr.get('prism')
        if prism or atom['xyz'] is None:
            html = ('<p><b>%s</b> · 六棱柱%s位置</p>'
                    '<p>被 <b>%d</b> 个六棱柱共享 → <b>%d×1/%d=1</b></p>') % (
                e, name, nshare, nshare, nshare)
        else:
            frac = atom['frac'] or (0, 0, 0)
            xyz = atom['xyz']
            html = ('<p><b>%s</b> · %s位置 · 分数坐标 (%s, %s, %s)</p>'
                    '<p>晶胞序号 (%d, %d, %d)</p>'
                    '<p>被 <b>%d</b> 个晶胞共享 → <b>%d×1/%d=1</b></p>') % (
                e, name,
                _frac_str(frac[0]), _frac_str(frac[1]), _frac_str(frac[2]),
                xyz[0], xyz[1], xyz[2],
                nshare, nshare, nshare)
        self.sel_browser.setHtml(html)


    # ---------- 测距/测角 ----------
    def _measure_pick(self, idx):
        a = self._atoms[idx]
        self._mpts.append(idx)
        mk = pv.Sphere(radius=self._rmax * .35, center=a['cart'],
                       theta_resolution=12, phi_resolution=12)
        self._m_actors.append(self.plot.add_mesh(mk, color='#00E5FF',
                                                 opacity=.9))
        n = len(self._mpts)
        if n == 1:
            self.statusBar().showMessage('已选1个原子', 5000)
            return
        p2 = np.asarray(self._atoms[self._mpts[-2]]['cart'], float)
        p3 = np.asarray(a['cart'], float)
        seg = lines_polydata([(p2, p3)])
        self._m_actors.append(self.plot.add_mesh(seg, color='#00E5FF',
                                                 line_width=3))
        d = float(np.linalg.norm(p3 - p2))
        if n == 2:
            self.statusBar().showMessage('距离 = %.3f Å' % d, 10000)
        elif n >= 3:
            p1 = np.asarray(self._atoms[self._mpts[-3]]['cart'], float)
            v1 = p1 - p2; v2 = p3 - p2
            c = float(np.dot(v1, v2) / (np.linalg.norm(v1) *
                                        np.linalg.norm(v2) + 1e-12))
            ang = math.degrees(math.acos(max(-1., min(1., c))))
            try:
                self.plot.add_point_labels([tuple(p2)], ['%.1f°' % ang],
                                           font_size=14, show_points=False,
                                           text_color='#006064')
            except Exception:
                pass
            self.statusBar().showMessage(
                '%s-%s-%s 键角 = %.1f°' % (self._atoms[self._mpts[-3]]['elem'],
                self._atoms[self._mpts[-2]]['elem'], a['elem'], ang), 12000)
            self._mpts = []
        else:
            self.statusBar().showMessage('距离 = %.3f Å' % d, 10000)

    # ---------- 统计 ----------
    def _update_stats(self):
        cr = self.cr
        if cr is None:
            return
        prism = (self.variant == 'prism') and cr.get('prism')
        try:
            counts = cc.prism_apportioned(cr) if prism \
                else cc.apportioned_counts(cr)
        except Exception:
            counts = {}
        try:
            f = cc.formula(cr)
        except Exception:
            f = '?'
        Z = int(round(sum(counts.values()))) if counts else 0
        html = ['<h3>%s</h3>' % cr['name'],
                '<p>%s · %s · a=%.3f b=%.3f c=%.3f Å · α=%.0f β=%.0f γ=%.0f</p>'
                % (cr['system'], cr['lattice'], cr['a'], cr['b'], cr['c'],
                   cr['alpha'], cr['beta'], cr['gamma'])]
        try:
            summ = cc.prism_summary(cr) if prism else cc.cell_atom_summary(cr)
            rows = ''
            for e, dd in summ.items():
                parts = ['%s %d×%s' % (PTYPE_CN[pt][0], num, _frac_str(sh))
                         for pt, (num, sh) in sorted(dd.items(),
                                                     key=lambda kv: -kv[1][1])]
                rows += ('<tr><td>%s</td><td>%s</td><td>%d</td></tr>'
                         % (e, ' + '.join(parts), int(round(
                             counts.get(e, 0)))))
            if rows:
                html.append('<table border="1" cellspacing="0" cellpadding="3" '
                            'width="100%"><tr><th>元素</th><th>位置分解</th>'
                            '<th>合计</th></tr>' + rows + '</table>')
        except Exception:
            traceback.print_exc()
        html.append('<p><b>化学式</b> %s · <b>Z</b> = %d</p>' % (f, Z))
        try:
            cn = cc.coordination(cr)
            html.append('<p><b>配位数</b> %s</p>' % ' · '.join(
                '%s:%d(%s)' % (e, v[0], ''.join(sorted(v[1])))
                for e, v in cn.items()))
        except Exception:
            pass
        try:
            html.append('<p><b>密度</b> %.2f g/cm³ · V=%.2f Å³</p>'
                        % (cc.density(cr), cc.cell_volume(cr)))
        except Exception:
            pass
        try:
            eta = cc.packing_fraction(cr, scale=self.radius_scale, prism=prism)
            html.append('<p><b>空间利用率</b> %.1f%%</p>' % (eta * 100))
        except Exception:
            pass
        try:
            md = cc.min_distances(cr)
            html.append('<p><b>键长</b> %s</p>' % ' · '.join(
                '%s–%s %.3f' % (k[0], k[1], v)
                for k, v in sorted(md.items())[:5]))
        except Exception:
            pass
        self.stats_browser.setHtml('\n'.join(html))

    def _update_tools(self):
        cr = self.cr
        if cr is None:
            return
        try:
            ba = cc.bond_angles(cr)
        except Exception:
            ba = {}
        try:
            md = cc.min_distances(cr)
        except Exception:
            md = {}
        rows = ''
        for k, v in sorted(md.items()):
            rows += '<tr><td>键长 %s–%s</td><td>%.3f Å</td></tr>' % (k[0], k[1], v)
        for e, lst in ba.items():
            for a, cnt in lst:
                rows += '<tr><td>键角 %s</td><td>%.1f° × %d</td></tr>' % (
                    e, a, cnt)
        self.tool_browser.setHtml(
            '<table border="1" cellspacing="0" cellpadding="3" width="100%">'
            + rows + '</table>' if rows else '<p>无</p>')
        for lbl, val in (('a', cr['a']), ('b', cr['b']), ('c', cr['c']),
                         ('α', cr['alpha']), ('β', cr['beta']),
                         ('γ', cr['gamma'])):
            self.dsp[lbl].setValue(val)

    # ---------- 工具 ----------
    def _draw_plane(self):
        cr = self.cr
        if cr is None:
            return
        h, k, l = self.sp_fh.value(), self.sp_fk.value(), self.sp_fl.value()
        if not any((h, k, l)):
            self.statusBar().showMessage('hkl不能全为0', 5000)
            return
        try:
            nv, d = cc.hkl_plane(cr, h, k, l)
            M = cc.lattice_matrix(cr)
            center = np.array([self.sp_nx.value() / 2., self.sp_ny.value() / 2.,
                               self.sp_nz.value() / 2.]) @ M
            t = float(np.dot(center, nv) - d)
            org = center - nv * t
            tmp = np.array([1., 0, 0]) if abs(nv[0]) < .9 else np.array([0, 1., 0])
            e1 = np.cross(nv, tmp); e1 /= np.linalg.norm(e1)
            e2 = np.cross(nv, e1)
            L = 1.6 * float(np.linalg.norm(
                np.array([self.sp_nx.value() * cr['a'],
                          self.sp_ny.value() * cr['b'],
                          self.sp_nz.value() * cr['c']])))
            quad = np.array([org - e1 * L / 2 - e2 * L / 2,
                             org + e1 * L / 2 - e2 * L / 2,
                             org + e1 * L / 2 + e2 * L / 2,
                             org - e1 * L / 2 + e2 * L / 2])
            m = pv.PolyData(quad)
            m.faces = np.array([4, 0, 1, 2, 3], np.int64)
            self.plot.add_mesh(m, color='#80DEEA', opacity=.35)
            self.statusBar().showMessage('(%d%d%d) d=%.3f Å' % (h, k, l, d),
                                         10000)
        except Exception:
            traceback.print_exc()

    def _mini_cr(self):
        return dict(a=self.dsp['a'].value(), b=self.dsp['b'].value(),
                    c=self.dsp['c'].value(), alpha=self.dsp['α'].value(),
                    beta=self.dsp['β'].value(), gamma=self.dsp['γ'].value())

    def _calc_d(self):
        try:
            d = cc.d_spacing(self._mini_cr(), self.sp_dh.value(),
                             self.sp_dk.value(), self.sp_dl.value())
            self.lb_d.setText('%.4f Å' % d)
        except Exception:
            self.lb_d.setText('错误')

    def _calc_rho(self):
        try:
            a = self.sp_da.value()
            rho = self.sp_dz.value() * self.sp_dm.value() / \
                (6.02214076e23 * a ** 3 * 1e-24)
            self.lb_rho.setText('ρ = %.3f g/cm³' % rho)
        except Exception:
            self.lb_rho.setText('错误')

    def _calc_a(self):
        try:
            pass
        except Exception:
            pass
        from PySide6.QtWidgets import QInputDialog
        rho, ok = QInputDialog.getDouble(self, '输入ρ', 'ρ (g/cm³)',
                                         2.16, 0.01, 30, 3)
        if not ok:
            return
        a = (self.sp_dz.value() * self.sp_dm.value() /
             (6.02214076e23 * rho * 1e-24)) ** (1 / 3)
        self.lb_rho.setText('a = %.4f Å' % a)

    # ---------- 元素颜色 ----------
    def _edit_colors(self):
        if not self._atoms:
            return
        elems = sorted(set(a['elem'] for a in self._atoms))
        dlg = QDialog(self)
        dlg.setWindowTitle('元素颜色')
        v = QVBoxLayout(dlg)
        form = QFormLayout()
        btns = {}
        for e in elems:
            b = QPushButton(self.elem_colors.get(e) or cc.elem_color(e))
            b.setStyleSheet('background:%s' % b.text())
            def _pick(ee=e, btn_ref=[b]):
                c = QColorDialog.getColor(QColor(btn_ref[0].text()), dlg)
                if c.isValid():
                    self.elem_colors[ee] = c.name()
                    btn_ref[0].setText(c.name())
                    btn_ref[0].setStyleSheet('background:%s' % c.name())
            b.clicked.connect(_pick)
            btns[e] = b
            form.addRow(e, b)
        v.addLayout(form)
        hb = QHBoxLayout()
        breset = QPushButton('重置')
        breset.clicked.connect(lambda: (self.elem_colors.clear(),
                                        dlg.reject(), self.rebuild()))
        bok = QPushButton('确定')
        bok.clicked.connect(lambda: (dlg.accept(), self.rebuild()))
        hb.addWidget(breset); hb.addWidget(bok)
        v.addLayout(hb)
        dlg.exec()

    # ---------- 自定义 / CIF ----------
    def _add_row(self, e='C', x=0, y=0, z=0):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        for c, val in enumerate([e, str(x), str(y), str(z)]):
            self.tbl.setItem(r, c, QTableWidgetItem(val))

    def _del_row(self):
        r = self.tbl.currentRow()
        if r >= 0:
            self.tbl.removeRow(r)

    def _fill_from_current(self):
        if not self.cr:
            return
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
            if not items[0]:
                continue
            try:
                e = items[0].text().strip()
                x, y, z = (float(items[c].text()) for c in (1, 2, 3))
                basis.append((e, x, y, z))
            except Exception:
                QMessageBox.warning(self, '输入有误', '第%d行无法解析' % (r + 1))
                return
        if not basis:
            QMessageBox.warning(self, '缺少原子', '请至少添加一个原子')
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

    def _import_cif(self):
        fn, _ = QFileDialog.getOpenFileName(self, 'CIF', APP_DIR,
                                            'CIF (*.cif *.CIF);;所有 (*)')
        if not fn:
            return
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
        self.statusBar().showMessage('已导入 %s' % cr['name'], 8000)

    def _save_custom(self, silent=False):
        try:
            data = {}
            for k, cr in self.custom_db.items():
                data[k] = dict(name=cr['name'], a=cr['a'], b=cr['b'],
                               c=cr['c'], alpha=cr['alpha'], beta=cr['beta'],
                               gamma=cr['gamma'], centering=cr['centering'],
                               basis=[[e, x, y, z] for e, (x, y, z)
                                      in cr['basis']],
                               notes=cr.get('notes', ''))
            with open(CUSTOM_JSON, 'w', encoding='utf-8') as fp:
                json.dump(data, fp, ensure_ascii=False, indent=1)
            if not silent:
                QMessageBox.information(self, '已保存', '我的晶体.json')
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

def _make_splash():
    pm = QPixmap(460, 280)
    pm.fill(QColor('#10151C'))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QColor('#8AB4F8'))
    f = QFont('Microsoft YaHei', 26)
    f.setBold(True)
    p.setFont(f)
    p.drawText(pm.rect().adjusted(0, -40, 0, -40), Qt.AlignCenter, APP_NAME)
    p.setPen(QColor('#9AA0A6'))
    p.setFont(QFont('Microsoft YaHei', 11))
    p.drawText(pm.rect().adjusted(0, 30, 0, 30), Qt.AlignCenter,
               'v%s · %s' % (APP_VERSION, APP_AUTHOR))
    p.setPen(QColor('#43A047'))
    for i, (x, y, r) in enumerate(((190, 200, 12), (230, 185, 12),
                                   (270, 200, 12), (210, 232, 12),
                                   (250, 232, 12), (230, 214, 12))):
        p.setBrush(QColor(['#E53935', '#43A047', '#1E88E5', '#FB8C00',
                           '#1E88E5', '#FFD54F'][i]))
        p.drawEllipse(x - r, y - r, 2 * r, 2 * r)
    p.end()
    sp = QSplashScreen(pm)
    return sp

def main():
    global cc, pv, vtk, QtInteractor
    app = QApplication(sys.argv)
    app.setApplicationName('%s v%s' % (APP_NAME, APP_VERSION))
    def hook(t, v, tb):
        traceback.print_exception(t, v, tb)
        try:
            QMessageBox.critical(None, '程序出错', '%s: %s' % (t.__name__, v))
        except Exception:
            pass
    sys.excepthook = hook
    sp = _make_splash()
    sp.show()
    app.processEvents()
    def smsg(t):
        try:
            sp.showMessage(t, Qt.AlignBottom, QColor('#8AB4F8'))
        except Exception:
            pass
        app.processEvents()

    smsg('计算引擎…')
    try:
        import crystal_core as cc
    except Exception as e:
        sp.close()
        _die('无法导入 crystal_core.py\n' + repr(e))
    _diag('crystal_core OK（%d种）' % len(cc.DB))
    smsg('渲染引擎…')
    try:
        import pyvista as pv
        import vtk
    except Exception as e:
        sp.close()
        _die('缺少 pyvista/vtk\n' + repr(e))
    try:
        from pyvistaqt import QtInteractor
    except Exception as e:
        sp.close()
        _die('缺少 pyvistaqt\n' + repr(e))
    _diag('pyvista/vtk OK')
    smsg('构建界面…')
    try:
        win = MainWindow()
    except Exception as e:
        sp.close()
        traceback.print_exc()
        _die('界面初始化失败\n' + repr(e))
    win.show()
    sp.finish(win)
    _diag('%s v%s 启动完成 %.1fs' % (APP_NAME, APP_VERSION,
                                    time.time() - _T0))
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
