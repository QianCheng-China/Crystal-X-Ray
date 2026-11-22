# -*- coding: utf-8 -*-
"""
crystal_core.py —— 晶胞演示软件 · 晶体学计算引擎（纯 Python+NumPy，无 GUI 依赖）
自测: python crystal_core.py --test
"""
import math
import numpy as np

AVOGADRO = 6.02214076e23

# ============ 1. 元素表: 符号 -> (中文名, 摩尔质量, 颜色, 默认显示半径Å) ============
ELEMENTS = {
 'H': ('氢', 1.008, '#F9F9F9', 0.32), 'He': ('氦', 4.003, '#D9FFFF', 1.20),
 'Li': ('锂', 6.94, '#CC80FF', 1.52), 'Be': ('铍', 9.012, '#C2FF00', 1.12),
 'B': ('硼', 10.81, '#FFB5B5', 0.87), 'C': ('碳', 12.011, '#4A4A4A', 0.77),
 'N': ('氮', 14.007, '#3050F8', 0.75), 'O': ('氧', 15.999, '#FF0D0D', 0.73),
 'F': ('氟', 18.998, '#90E050', 0.71), 'Ne': ('氖', 20.18, '#B3E3F5', 1.12),
 'Na': ('钠', 22.990, '#AB5CF2', 1.86), 'Mg': ('镁', 24.305, '#8AFF00', 1.60),
 'Al': ('铝', 26.982, '#BFA6A6', 1.43), 'Si': ('硅', 28.086, '#F0C8A0', 1.11),
 'P': ('磷', 30.974, '#FF8000', 1.07), 'S': ('硫', 32.06, '#FFE22A', 1.02),
 'Cl': ('氯', 35.45, '#1FF01F', 1.81), 'Ar': ('氩', 39.948, '#80D1E3', 1.88),
 'K': ('钾', 39.098, '#8F40D4', 2.27), 'Ca': ('钙', 40.078, '#3DFF00', 1.97),
 'Ti': ('钛', 47.867, '#BFC2C7', 1.47), 'Cr': ('铬', 51.996, '#8A99C7', 1.28),
 'Mn': ('锰', 54.938, '#9C7AC7', 1.27), 'Fe': ('铁', 55.845, '#E06633', 1.24),
 'Co': ('钴', 58.933, '#F090A0', 1.25), 'Ni': ('镍', 58.693, '#50D050', 1.24),
 'Cu': ('铜', 63.546, '#C88033', 1.28), 'Zn': ('锌', 65.38, '#7D80B0', 1.33),
 'Ge': ('锗', 72.63, '#668F8F', 1.22), 'As': ('砷', 74.922, '#BD80E3', 1.19),
 'Se': ('硒', 78.971, '#FFA100', 1.20), 'Br': ('溴', 79.904, '#A62929', 1.20),
 'Ag': ('银', 107.87, '#C0C0C0', 1.44), 'I': ('碘', 126.90, '#940094', 1.35),
 'Cs': ('铯', 132.91, '#57178F', 1.76), 'Ba': ('钡', 137.33, '#00C900', 2.17),
 'W': ('钨', 183.84, '#21994D', 1.39), 'Pt': ('铂', 195.08, '#D6D6D6', 1.39),
 'Au': ('金', 196.97, '#FFD123', 1.44), 'Hg': ('汞', 200.59, '#B8B8D0', 1.49),
 'Pb': ('铅', 207.2, '#575961', 1.75), 'Po': ('钋', 209.0, '#AB5C00', 1.68),
}

def elem_info(sym):
    sym = str(sym).strip()
    sym2 = sym[:1].upper() + sym[1:2].lower()
    if sym2 in ELEMENTS:
        return ELEMENTS[sym2]
    return (sym, 0.0, '#DD8899', 1.30)

def elem_name(sym):  return elem_info(sym)[0]
def elem_mass(sym):  return elem_info(sym)[1]
def elem_color(sym): return elem_info(sym)[2]
def elem_radius(sym):return elem_info(sym)[3]

# ============ 2. 晶体库（35种） ============
def _cr(key, name, cat, sysm, abc, angles=(90, 90, 90), centering='P',
        basis=(), radii=None, bonds=None, poly=(), holes=None,
        prism=False, lattice='', notes=''):
    a, b, c = (list(abc) + [None, None, None])[:3]
    if b is None: b = a
    if c is None: c = a
    return dict(key=key, name=name, category=cat, system=sysm, a=float(a),
                b=float(b), c=float(c), alpha=angles[0], beta=angles[1],
                gamma=angles[2], centering=centering,
                basis=[(e, (float(x), float(y), float(z))) for e, (x, y, z) in basis],
                radii=radii or {}, bonds=bonds, poly=list(poly), holes=holes,
                prism=prism, lattice=lattice, notes=notes, custom=False)

F_BASIS = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
I_BASIS = [(0, 0, 0), (.5, .5, .5)]
A_BASIS = [(0, 0, 0), (0, .5, .5)]
B_BASIS = [(0, 0, 0), (.5, 0, .5)]
C_BASIS = [(0, 0, 0), (.5, .5, 0)]
R_BASIS = [(0, 0, 0), (2/3, 1/3, 1/3), (1/3, 2/3, 2/3)]
CENTERING_BASIS = {'P': [(0, 0, 0)], 'I': I_BASIS, 'F': F_BASIS,
                   'A': A_BASIS, 'B': B_BASIS, 'C': C_BASIS, 'R': R_BASIS}

CRYSTAL_DB = [
 _cr('po', '钋 Po（简单立方）', '金属晶体', '立方', (3.35,), centering='P',
     basis=[('Po', (0, 0, 0))], radii={'Po': 1.68}, bonds=[],
     lattice='简单立方(P)',
     notes='顶点8×1/8 = 1个原子；配位数6；空间利用率52.36%——所有堆积中最低，仅Po采用。'),
 _cr('na', '钠 Na（体心立方）', '金属晶体', '立方', (4.29,), centering='I',
     basis=[('Na', (0, 0, 0))], radii={'Na': 1.86}, bonds=[],
     lattice='体心立方(I)',
     notes='8顶点×1/8 + 体心1 = 2；配位数8；空间利用率68.02%。碱金属与Fe/Cr/W属此型。'),
 _cr('k', '钾 K（体心立方）', '金属晶体', '立方', (5.33,), centering='I',
     basis=[('K', (0, 0, 0))], radii={'K': 2.31}, bonds=[],
     lattice='体心立方(I)', notes='同Na：Z=2，CN=8，68.02%。'),
 _cr('fe', '铁 Fe（体心立方）', '金属晶体', '立方', (2.866,), centering='I',
     basis=[('Fe', (0, 0, 0))], radii={'Fe': 1.24}, bonds=[],
     lattice='体心立方(I)', notes='α-铁：Z=2，CN=8，68.02%；密度7.87 g/cm³。'),
 _cr('cr', '铬 Cr（体心立方）', '金属晶体', '立方', (2.884,), centering='I',
     basis=[('Cr', (0, 0, 0))], radii={'Cr': 1.25}, bonds=[],
     lattice='体心立方(I)', notes='Z=2，CN=8，68.02%。'),
 _cr('w', '钨 W（体心立方）', '金属晶体', '立方', (3.165,), centering='I',
     basis=[('W', (0, 0, 0))], radii={'W': 1.37}, bonds=[],
     lattice='体心立方(I)', notes='Z=2，CN=8，68.02%；熔点最高的金属。'),
 _cr('cu', '铜 Cu（面心立方）', '金属晶体', '立方', (3.615,), centering='F',
     basis=[('Cu', (0, 0, 0))], radii={'Cu': 1.28}, bonds=[],
     lattice='面心立方(F)',
     notes='8×1/8 + 6×1/2 = 4；配位数12；空间利用率74.05%（ABCABC…最密堆积）。'),
 _cr('ag', '银 Ag（面心立方）', '金属晶体', '立方', (4.086,), centering='F',
     basis=[('Ag', (0, 0, 0))], radii={'Ag': 1.44}, bonds=[], lattice='面心立方(F)',
     notes='Z=4，CN=12，74.05%。'),
 _cr('au', '金 Au（面心立方）', '金属晶体', '立方', (4.078,), centering='F',
     basis=[('Au', (0, 0, 0))], radii={'Au': 1.44}, bonds=[], lattice='面心立方(F)',
     notes='Z=4，CN=12，74.05%。'),
 _cr('al', '铝 Al（面心立方）', '金属晶体', '立方', (4.049,), centering='F',
     basis=[('Al', (0, 0, 0))], radii={'Al': 1.43}, bonds=[], lattice='面心立方(F)',
     notes='Z=4，CN=12，74.05%。'),
 _cr('ni', '镍 Ni（面心立方）', '金属晶体', '立方', (3.524,), centering='F',
     basis=[('Ni', (0, 0, 0))], radii={'Ni': 1.24}, bonds=[], lattice='面心立方(F)',
     notes='Z=4，CN=12，74.05%。'),
 _cr('pb', '铅 Pb（面心立方）', '金属晶体', '立方', (4.95,), centering='F',
     basis=[('Pb', (0, 0, 0))], radii={'Pb': 1.75}, bonds=[], lattice='面心立方(F)',
     notes='Z=4，CN=12，74.05%。'),
 _cr('pt', '铂 Pt（面心立方）', '金属晶体', '立方', (3.924,), centering='F',
     basis=[('Pt', (0, 0, 0))], radii={'Pt': 1.39}, bonds=[], lattice='面心立方(F)',
     notes='Z=4，CN=12，74.05%。'),
 _cr('ca', '钙 Ca（面心立方）', '金属晶体', '立方', (5.58,), centering='F',
     basis=[('Ca', (0, 0, 0))], radii={'Ca': 1.97}, bonds=[], lattice='面心立方(F)',
     notes='Z=4，CN=12，74.05%。'),
 _cr('mg', '镁 Mg（六方最密）', '金属晶体', '六方', (3.21, None, 5.21), (90, 90, 120),
     centering='P', basis=[('Mg', (0, 0, 0)), ('Mg', (2/3, 1/3, 1/2))],
     radii={'Mg': 1.60}, bonds=[], prism=True, lattice='简单六方格子',
     notes='平行六面体晶胞：8顶点×1/8 + 内部1 = 2个原子。'
           '教材式六棱柱：12顶点×1/6 + 2面心×1/2 + 3内部 = 6。'
           'CN=12，74.05%（ABAB…堆积）。Mg/Zn/Ti属此型。'),
 _cr('zn', '锌 Zn（六方最密）', '金属晶体', '六方', (2.66, None, 4.95), (90, 90, 120),
     centering='P', basis=[('Zn', (0, 0, 0)), ('Zn', (2/3, 1/3, 1/2))],
     radii={'Zn': 1.33}, bonds=[], prism=True, lattice='简单六方格子',
     notes='Z=2；理想c/a=1.633，锌为1.86（畸变六方，面内6+层间6）。'),
 _cr('ti', '钛 Ti（六方最密）', '金属晶体', '六方', (2.95, None, 4.68), (90, 90, 120),
     centering='P', basis=[('Ti', (0, 0, 0)), ('Ti', (2/3, 1/3, 1/2))],
     radii={'Ti': 1.47}, bonds=[], prism=True, lattice='简单六方格子',
     notes='Z=2；CN=12；74.05%。'),
 _cr('be', '铍 Be（六方最密）', '金属晶体', '六方', (2.29, None, 3.58), (90, 90, 120),
     centering='P', basis=[('Be', (0, 0, 0)), ('Be', (2/3, 1/3, 1/2))],
     radii={'Be': 1.12}, bonds=[], prism=True, lattice='简单六方格子', notes='Z=2；CN=12。'),
 _cr('co', '钴 Co（六方最密）', '金属晶体', '六方', (2.51, None, 4.07), (90, 90, 120),
     centering='P', basis=[('Co', (0, 0, 0)), ('Co', (2/3, 1/3, 1/2))],
     radii={'Co': 1.25}, bonds=[], prism=True, lattice='简单六方格子', notes='Z=2；CN=12。'),
 _cr('cscl', '氯化铯 CsCl', '离子晶体', '立方', (4.123,), centering='P',
     basis=[('Cl', (0, 0, 0)), ('Cs', (.5, .5, .5))],
     radii={'Cs': 1.76, 'Cl': 1.81}, bonds=[('Cs', 'Cl')], poly=[('Cs', 'Cl')],
     lattice='简单立方(P)',
     notes='Cs⁺占体心、Cl⁻占顶点：8×1/8=1个Cl⁻ + 体心1个Cs⁺ → CsCl。配位数8:8。'
           'r(Cs⁺)+r(Cl⁻)=a√3/2。注意：这不是体心立方格子，是两套简单立方穿插。'),
 _cr('nacl', '氯化钠 NaCl（岩盐）', '离子晶体', '立方', (5.640,), centering='F',
     basis=[('Cl', (0, 0, 0)), ('Na', (.5, .5, .5))],
     radii={'Na': 1.02, 'Cl': 1.81}, bonds=[('Na', 'Cl')], poly=[('Na', 'Cl')],
     holes='Cl', lattice='面心立方(F)',
     notes='Cl⁻：8×1/8+6×1/2=4；Na⁺：12×1/4+1=4 → NaCl。配位数6:6。'
           'Na⁺填满Cl⁻面心立方骨架的全部八面体空隙；r(Na⁺)+r(Cl⁻)=a/2。'),
 _cr('mgo', '氧化镁 MgO', '离子晶体', '立方', (4.212,), centering='F',
     basis=[('O', (0, 0, 0)), ('Mg', (.5, .5, .5))],
     radii={'Mg': 0.72, 'O': 1.40}, bonds=[('Mg', 'O')], poly=[('Mg', 'O')],
     holes='O', lattice='面心立方(F)',
     notes='与NaCl同型（岩盐型）：Mg²⁺4个 + O²⁻4个；配位数6:6。'),
 _cr('caf2', '氟化钙 CaF₂（萤石）', '离子晶体', '立方', (5.463,), centering='F',
     basis=[('Ca', (0, 0, 0)), ('F', (.25, .25, .25)), ('F', (.75, .75, .75))],
     radii={'Ca': 1.09, 'F': 1.28}, bonds=[('Ca', 'F')], poly=[('F', 'Ca')],
     holes='Ca', lattice='面心立方(F)',
     notes='Ca²⁺：8×1/8+6×1/2=4；F⁻：8（内部，填满全部四面体空隙）→ CaF₂。'
           '配位数：Ca²⁺为8、F⁻为4；Ca-F=a√3/4。'),
 _cr('znsw', '硫化锌 ZnS（闪锌矿）', '离子晶体', '立方', (5.41,), centering='F',
     basis=[('S', (0, 0, 0)), ('Zn', (.25, .25, .25))],
     radii={'Zn': 0.90, 'S': 1.45}, bonds=None, poly=[('Zn', 'S')], holes='S',
     lattice='面心立方(F)',
     notes='S²⁻面心立方(4个)，Zn²⁺占一半四面体空隙(4个)→ ZnS。配位数4:4。'
           '与金刚石同型（把C换成Zn、S）。'),
 _cr('znsz', '硫化锌 ZnS（纤锌矿）', '离子晶体', '六方', (3.82, None, 6.26), (90, 90, 120),
     centering='P',
     basis=[('Zn', (1/3, 2/3, 0)), ('Zn', (2/3, 1/3, .5)),
            ('S', (1/3, 2/3, .375)), ('S', (2/3, 1/3, .875))],
     radii={'Zn': 0.90, 'S': 1.45}, bonds=[('Zn', 'S')], poly=[('Zn', 'S')],
     lattice='简单六方格子',
     notes='六方ZnS：Zn²⁺、S²⁻各2个（Z=2），配位数4:4，是闪锌矿的六方变体。'),
 _cr('na2o', '氧化钠 Na₂O（反萤石）', '离子晶体', '立方', (5.56,), centering='F',
     basis=[('O', (0, 0, 0)), ('Na', (.25, .25, .25)), ('Na', (.75, .75, .75))],
     radii={'Na': 1.02, 'O': 1.40}, bonds=[('Na', 'O')], holes='O',
     lattice='面心立方(F)',
     notes='反萤石结构：O²⁻面心立方(4个)，Na⁺填满全部四面体空隙(8个)→ Na₂O。'
           '配位数：Na⁺为4、O²⁻为8（与CaF₂恰好互换）。'),
 _cr('bn', '氮化硼 BN（立方）', '共价晶体', '立方', (3.615,), centering='F',
     basis=[('N', (0, 0, 0)), ('B', (.25, .25, .25))],
     radii={'B': 0.82, 'N': 0.75}, bonds=None, poly=[('B', 'N')],
     lattice='面心立方(F)',
     notes='闪锌矿型：B、N均sp³杂化，配位数4:4，Z=4。立方氮化硼硬度接近金刚石。'),
 _cr('rutile', '二氧化钛 TiO₂（金红石）', '离子晶体', '四方', (4.594, None, 2.958),
     centering='P',
     basis=[('Ti', (0, 0, 0)), ('Ti', (.5, .5, .5)),
            ('O', (.305, .305, 0)), ('O', (.695, .695, 0)),
            ('O', (.805, .195, .5)), ('O', (.195, .805, .5))],
     radii={'Ti': 0.68, 'O': 1.32}, bonds=[('Ti', 'O')], poly=[('Ti', 'O')],
     lattice='简单四方(P)',
     notes='Ti⁴⁺2个（顶点+体心），O²⁻4个 → TiO₂。Ti⁴⁺的配位数为6（TiO₆八面体），'
           'O²⁻为3。金红石是常见白色颜料。'),
 _cr('perov', '钙钛矿 CaTiO₃', '离子晶体', '立方', (3.84,), centering='P',
     basis=[('Ca', (0, 0, 0)), ('Ti', (.5, .5, .5)),
            ('O', (.5, .5, 0)), ('O', (.5, 0, .5)), ('O', (0, .5, .5))],
     radii={'Ca': 1.35, 'Ti': 0.68, 'O': 1.32}, bonds=[('Ti', 'O')], poly=[('Ti', 'O')],
     lattice='简单立方(P)',
     notes='Ca²⁺：8×1/8=1；Ti⁴⁺：体心1；O²⁻：6×1/2=3 → CaTiO₃。TiO₆八面体共顶点'
           '连接，Ca²⁺处于十二配位。BaTiO₃同型（压电材料）。'),
 _cr('diamond', '金刚石 C', '共价晶体', '立方', (3.567,), centering='F',
     basis=[('C', (0, 0, 0)), ('C', (.25, .25, .25))],
     radii={'C': 0.77}, bonds=None, poly=[('C', 'C')], holes='C',
     lattice='面心立方(F)',
     notes='碳原子：面心立方8个 + 内部4个（占一半四面体空隙）= 8。sp³杂化，'
           '配位数4，键长1.55Å=a√3/4。空间利用率仅34%，但键能大→硬度最大。密度3.51。'),
 _cr('si', '硅 Si', '共价晶体', '立方', (5.431,), centering='F',
     basis=[('Si', (0, 0, 0)), ('Si', (.25, .25, .25))],
     radii={'Si': 1.11}, bonds=None, poly=[('Si', 'Si')], holes='Si',
     lattice='面心立方(F)', notes='金刚石型：Z=8，CN=4，键长2.35Å；半导体基石。'),
 _cr('ge', '锗 Ge', '共价晶体', '立方', (5.658,), centering='F',
     basis=[('Ge', (0, 0, 0)), ('Ge', (.25, .25, .25))],
     radii={'Ge': 1.22}, bonds=None, poly=[('Ge', 'Ge')], holes='Ge',
     lattice='面心立方(F)', notes='金刚石型：Z=8，CN=4。'),
 _cr('graphite', '石墨 C', '共价晶体', '六方', (2.46, None, 6.71), (90, 90, 120),
     centering='P',
     basis=[('C', (0, 0, 0)), ('C', (1/3, 2/3, 0)),
            ('C', (0, 0, .5)), ('C', (2/3, 1/3, .5))],
     radii={'C': 0.77}, bonds=[('C', 'C')],
     lattice='简单六方格子',
     notes='层状结构：每晶胞4个C。层内sp²杂化，C-C 1.42Å；层间3.35Å靠范德华力'
           '→质软、层间可滑动、能导电。'),
 _cr('co2', '干冰 CO₂', '分子晶体', '立方', (5.62,), centering='P',
     basis=(('C', (0, 0, 0)), ('C', (0, .5, .5)), ('C', (.5, 0, .5)), ('C', (.5, .5, 0)),
            ('O', (.119, .119, .119)), ('O', (-.119, -.119, -.119)),
            ('O', (.119, .5 - .119, .5 - .119)), ('O', (-.119, .5 + .119, .5 + .119)),
            ('O', (.5 - .119, .119, .5 - .119)), ('O', (.5 + .119, -.119, .5 + .119)),
            ('O', (.5 - .119, .5 - .119, .119)), ('O', (.5 + .119, .5 + .119, -.119))),
     radii={'C': 0.77, 'O': 0.73}, bonds=[('C', 'O')],
     lattice='面心立方排列的分子',
     notes='CO₂分子（直线形，C=O 1.16Å）占据面心立方位置，分子轴沿体对角线方向；'
           '每晶胞4个CO₂分子。分子间仅范德华力→熔点-78.5°C、易升华。'),
 _cr('ar', '固态氩 Ar', '分子晶体', '立方', (5.31,), centering='F',
     basis=[('Ar', (0, 0, 0))], radii={'Ar': 1.88}, bonds=[],
     lattice='面心立方(F)',
     notes='稀有气体单原子分子呈面心立方密堆积：Z=4，“配位数”12（仅范德华接触）。'),
]

def make_custom(key, name, a, b, c, alpha, beta, gamma, centering, basis,
                bonds=None, notes=''):
    """兼容两种原子格式: (元素,(x,y,z)) 与 (元素,x,y,z)"""
    norm = []
    for item in basis:
        if len(item) == 2:
            e, xyz = item
            x, y, z = xyz
        else:
            e, x, y, z = item
        norm.append((str(e), (float(x), float(y), float(z))))
    return dict(key=key, name=name, category='自定义', system='自定义',
                a=float(a), b=float(b), c=float(c), alpha=float(alpha),
                beta=float(beta), gamma=float(gamma), centering=centering,
                basis=norm,
                radii={}, bonds=bonds, poly=[], holes=None, prism=False,
                lattice='自定义', notes=notes or '自定义晶体', custom=True)

DB = {c['key']: c for c in CRYSTAL_DB}

# ============ 3. 晶格 / 位置 / 均摊 ============
def lattice_matrix(cr):
    a, b, c = cr['a'], cr['b'], cr['c']
    al, be, ga = (math.radians(x) for x in (cr['alpha'], cr['beta'], cr['gamma']))
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(ga), b * math.sin(ga), 0.0])
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.vstack([va, vb, np.array([cx, cy, cz])])

def cell_volume(cr):
    return abs(float(np.linalg.det(lattice_matrix(cr))))

def _wrap01(x):
    x = x - math.floor(x)
    if abs(x - 1.0) < 1e-9: x = 0.0
    return x

def unit_sites(cr):
    """晶胞内独立原子位置(site)：基元×格子心化, 折叠[0,1), 去重"""
    out, seen = [], set()
    cent = CENTERING_BASIS[cr['centering']]
    for e, (x, y, z) in cr['basis']:
        for t in cent:
            u, v, w = _wrap01(x + t[0]), _wrap01(y + t[1]), _wrap01(z + t[2])
            key = (e, round(u, 5), round(v, 5), round(w, 5))
            if key not in seen:
                seen.add(key)
                out.append((e, (u, v, w)))
    return out

POS_TYPE = {'corner': ('顶点', 8, 0.125), 'edge': ('棱', 4, 0.25),
            'face': ('面', 2, 0.5), 'body': ('体内', 1, 1.0)}

def pos_type(frac, tol=1e-6):
    z = min(sum(1 for v in frac if abs(v) < tol), 3)
    return ['body', 'face', 'edge', 'corner'][z]

def closed_cell_images(frac):
    """闭晶胞边界镜像: 分量为0的位置在对面(=1)还有一个镜像"""
    u, v, w = frac
    us = [u, 1.0] if abs(u) < 1e-9 else [u]
    vs = [v, 1.0] if abs(v) < 1e-9 else [v]
    ws = [w, 1.0] if abs(w) < 1e-9 else [w]
    return [(x, y, z) for x in us for y in vs for z in ws]

def supercell_atoms(cr, n=1):
    """n×n×n 超晶胞原子(含闭胞边界镜像、去重)。每个原子带均摊信息"""
    M = lattice_matrix(cr)
    atoms, seen = [], set()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for e, f in unit_sites(cr):
                    for img in closed_cell_images(f):
                        s = (i + img[0], j + img[1], k + img[2])
                        cart = M.T @ np.array(s)
                        key = tuple(np.round(cart, 4))
                        if key in seen:
                            continue
                        seen.add(key)
                        pt = pos_type(f)
                        atoms.append(dict(
                            elem=e, xyz=s, cart=cart, cell=(i, j, k), frac=f,
                            ptype=pt, share=POS_TYPE[pt][2], nshare=POS_TYPE[pt][1]))
    return atoms

def apportioned_counts(cr):
    """均摊后每元素原子数 = site数（教材“顶点×1/8+…”公式是它的直观分解）"""
    cnt = {}
    for e, f in unit_sites(cr):
        cnt[e] = cnt.get(e, 0) + 1
    return cnt

def cell_atom_summary(cr):
    """教材式位置分解: {elem: {ptype: [可见原子数, 每个分摊]}}"""
    out = {}
    for e, f in unit_sites(cr):
        pt = pos_type(f)
        out.setdefault(e, {}).setdefault(pt, [0, POS_TYPE[pt][2]])
        out[e][pt][0] += len(closed_cell_images(f))
    return out

# ============ 4. 六棱柱(教材式) ============
def _hex_normals():
    return [(math.cos(math.radians(30 + 60 * k)), math.sin(math.radians(30 + 60 * k)))
            for k in range(6)]

def _in_hex(q, center, a, eps=1e-6):
    lim = a * math.sqrt(3) / 2 + eps
    dx, dy = q[0] - center[0], q[1] - center[1]
    for nx, ny in _hex_normals():
        if dx * nx + dy * ny > lim:
            return False
    return True

def prism_cells(cr, nc=1):
    """要画框的六棱柱中心(honeycomb)"""
    a = cr['a']
    w1 = np.array([1.5 * a, a * math.sqrt(3) / 2])
    w2 = np.array([1.5 * a, -a * math.sqrt(3) / 2])
    rmax = (nc - 1) * 1.75 * a + 1e-6
    cells = []
    for m in range(-2 * nc, 2 * nc + 1):
        for n_ in range(-2 * nc, 2 * nc + 1):
            p = m * w1 + n_ * w2
            if np.linalg.norm(p) <= rmax:
                cells.append(p)
    return cells

def prism_atoms(cr, nc=1):
    """教材式六棱柱原子: 顶点(1/6) 面心(1/2) 体内(1)"""
    a, c = cr['a'], cr['c']
    v1 = np.array([a, 0.0]); v2 = np.array([a / 2, a * math.sqrt(3) / 2])
    hole = (v1 + v2) / 3.0
    elem = cr['basis'][0][0]
    drawn = prism_cells(cr, nc)
    atoms = []
    K = 3 * nc + 2
    for m in range(-K, K + 1):
        for n_ in range(-K, K + 1):
            p = m * v1 + n_ * v2
            if not any(_in_hex(p, cc, a) for cc in drawn):
                continue
            coset = (m - n_) % 3
            pt = 'prism_face' if coset == 0 else 'prism_vertex'
            sh = 0.5 if coset == 0 else 1 / 6
            ns = 2 if coset == 0 else 6
            for k in range(nc + 1):
                atoms.append(dict(elem=elem, xyz=None,
                                  cart=np.array([p[0], p[1], k * c]), cell=None,
                                  frac=None, ptype=pt, share=sh, nshare=ns))
    for cc in drawn:
        for k in range(nc):
            z = k * c + c / 2
            for off in (hole, hole - v1, hole - v2):
                q = cc + off
                atoms.append(dict(elem=elem, xyz=None,
                                  cart=np.array([q[0], q[1], z]), cell=None,
                                  frac=None, ptype='prism_body', share=1.0, nshare=1))
    return atoms

_prism_atoms = prism_atoms   # 兼容旧名

PRISM_POS = {'prism_vertex': ('顶点', 6, 1/6), 'prism_face': ('面心', 2, 0.5),
             'prism_body': ('体内', 1, 1.0)}

def prism_apportioned(cr):
    return {cr['basis'][0][0]: 6.0}   # 12×1/6 + 2×1/2 + 3

def prism_summary(cr):
    return {cr['basis'][0][0]:
            {'prism_vertex': [12, 1/6], 'prism_face': [2, 0.5],
             'prism_body': [3, 1.0]}}

# ============ 5. 化学式 / 密度 / 利用率 / 配位 / 键 ============
_METALS = set('Li Be Na Mg Al K Ca Ti Cr Mn Fe Co Ni Cu Zn Ag Cs Ba W Pt Au Hg Pb Po'
              .split())

def formula(cr, counts=None):
    if counts is None:
        counts = apportioned_counts(cr)
    cnt = {e: v for e, v in counts.items() if v > 1e-6}
    if len(cnt) == 1:
        return next(iter(cnt))
    # 约成最简整数比: 4Na+4Cl->NaCl, 4Ca+8F->CaF2
    try:
        if all(abs(v - round(v)) < 1e-6 for v in cnt.values()):
            g = 0
            for v in cnt.values():
                g = math.gcd(g, int(round(v)))
            if g > 1:
                cnt = {e: round(v) / g for e, v in cnt.items()}
    except Exception:
        pass
    if any(e in _METALS for e in cnt):
        order = sorted(e for e in cnt if e in _METALS) + \
                sorted(e for e in cnt if e not in _METALS)
    else:
        order = sorted(cnt)
        if 'C' in cnt:
            order = ['C'] + (['H'] if 'H' in cnt else []) + \
                    sorted(e for e in cnt if e not in ('C', 'H'))
    sub = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    def s(v):
        v = round(v)
        return str(v).translate(sub) if v > 1 else ''
    return ''.join(f'{e}{s(cnt[e])}' for e in order)

def density(cr):
    counts = apportioned_counts(cr)
    m = sum(elem_mass(e) * v for e, v in counts.items())
    V = cell_volume(cr) * 1e-24
    return m / (AVOGADRO * V)

def display_radius(cr, e):
    return cr.get('radii', {}).get(e, elem_radius(e))

def packing_fraction(cr, scale=1.0, prism=False):
    counts = prism_apportioned(cr) if prism else apportioned_counts(cr)
    V = cell_volume(cr) * (3 if prism else 1)
    sph = sum(4 / 3 * math.pi * (display_radius(cr, e) * scale) ** 3 * v
              for e, v in counts.items())
    return sph / V

def _trans_vectors(M):
    return [np.array([i, j, k]) @ M for i in (-1, 0, 1) for j in (-1, 0, 1)
            for k in (-1, 0, 1)]

def min_distances(cr):
    """元素对 -> 最短周期距离(Å)（含同位点的纯格子平移）"""
    M = lattice_matrix(cr)
    ua = unit_sites(cr)
    ts = _trans_vectors(M)
    out = {}
    for e1, f1 in ua:
        for e2, f2 in ua:
            same = (e1 == e2 and f1 == f2)
            d1 = np.array(f1) @ M
            d2 = np.array(f2) @ M
            cands = [np.linalg.norm(d1 - d2 - t) for t in ts
                     if not (same and np.allclose(t, 0))]
            if not cands:
                continue
            key = tuple(sorted((e1, e2)))
            out[key] = min(out.get(key, 1e9), min(cands))
    return out

def coordination(cr):
    """{elem: (配位数, 邻居元素集合)}，自动计算（含周期镜像）"""
    M = lattice_matrix(cr)
    ua = unit_sites(cr)
    ts = _trans_vectors(M)
    res = {}
    for e1, f1 in ua:
        d1 = np.array(f1) @ M
        pairs = []
        for e2, f2 in ua:
            same = (e1 == e2 and f1 == f2)
            d2 = np.array(f2) @ M
            for t in ts:
                if same and np.allclose(t, 0):
                    continue
                d = np.linalg.norm(d1 - d2 - t)
                if d > 1e-4:
                    pairs.append((d, e2))
        if not pairs:
            continue
        dmin = min(p[0] for p in pairs)
        nb = [p for p in pairs if p[0] <= dmin * 1.03 + 1e-4]
        res[e1] = (len(nb), set(e for _, e in nb))
    return res

def bonds_for(cr, atoms, scale=1.0):
    """显示原子之间的键(去重)。bonds=[]无键; None自动; 显式列表按元素对"""
    spec = cr.get('bonds', None)
    md = min_distances(cr)
    if spec == []:
        return []
    if spec is None:
        allowed = {k for k, v in md.items()
                   if v <= 1.25 * (display_radius(cr, k[0]) + display_radius(cr, k[1]))}
    else:
        allowed = {tuple(sorted(p)) for p in spec}
    pos = np.array([a['cart'] for a in atoms])
    out = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            e = tuple(sorted((atoms[i]['elem'], atoms[j]['elem'])))
            if e not in allowed:
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            if d <= md.get(e, 1e9) * 1.15 + 0.02:
                out.append((i, j))
    return out

def fcc_holes(cr):
    """面心立方骨架空隙: (八面体列表, 四面体列表), 每项(闭胞坐标, 占据元素|None)"""
    if not cr.get('holes'):
        return [], []
    ua = unit_sites(cr)
    octa_sites = [(0.5, 0.5, 0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)]
    tetra_sites = [(x, y, z) for x in (0.25, 0.75) for y in (0.25, 0.75)
                   for z in (0.25, 0.75)]
    def occ(f):
        q = np.array(f, float)
        for e, g in ua:
            d = q - np.array(g)
            d = d - np.round(d)          # 周期最小镜像判定
            if np.linalg.norm(d) < 0.05:
                return e
        return None
    oct_out = [(img, occ(f)) for f in octa_sites for img in closed_cell_images(f)]
    tet_out = [(img, occ(f)) for f in tetra_sites for img in closed_cell_images(f)]
    return oct_out, tet_out

def convex_hull_faces(points, eps=1e-7):
    """≤12点的暴力凸包(配位多面体用), 返回三角形顶点索引"""
    n = len(points)
    P = np.asarray(points, float)
    if n < 4:
        return []
    c = P.mean(axis=0)
    faces = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                nrm = np.cross(P[j] - P[i], P[k] - P[i])
                if np.linalg.norm(nrm) < eps:
                    continue
                vals = [np.dot(nrm, P[m] - P[i]) for m in range(n)
                        if m not in (i, j, k)]
                if all(v <= 1e-7 for v in vals) or all(v >= -1e-7 for v in vals):
                    faces.append((i, j, k) if np.dot(nrm, c - P[i]) > 0
                                 else (k, j, i))
    return faces

def coordination_polyhedra(cr, atoms):
    """[{center, verts}] 只对配位完整的中心原子作多面体"""
    out = []
    if not cr.get('poly'):
        return out
    try:
        cn_map = coordination(cr)
    except Exception:
        cn_map = {}
    pos = np.array([a['cart'] for a in atoms])
    for center_e, nb_e in cr['poly']:
        expect = cn_map.get(center_e, (0,))[0]
        for i, a in enumerate(atoms):
            if a['elem'] != center_e:
                continue
            others = [(j, np.linalg.norm(pos[j] - pos[i])) for j in range(len(atoms))
                      if j != i and atoms[j]['elem'] == nb_e]
            if len(others) < 4:
                continue
            dmin = min(x[1] for x in others)
            verts = [j for j, dj in others if dj <= dmin * 1.15 + 0.05]
            if expect and len(verts) != expect:
                continue          # 显示边界处配位不完整 → 跳过
            if 4 <= len(verts) <= 12 and \
               len(convex_hull_faces([pos[j] for j in verts])) >= 4:
                out.append(dict(center=i, verts=verts))
    return out

# ============ 6. 金属密堆积演示 ============
def build_stack(sequence, r=1.3, extent=4):
    """密置层堆叠: 'ABCA...' 每字符一层"""
    v1 = np.array([2 * r, 0.0]); v2 = np.array([r, r * math.sqrt(3)])
    hole1 = (v1 + v2) / 3.0
    hole2 = 2 * hole1
    off = {'A': np.zeros(2), 'B': hole2, 'C': hole1}
    h = stack_layer_height(r)
    atoms = []
    R = extent * 2 * r * 0.87
    for L, ch in enumerate(sequence):
        if ch not in off:
            continue
        base = off[ch]
        for m in range(-extent - 1, extent + 2):
            for n in range(-extent - 1, extent + 2):
                p = base + m * v1 + n * v2
                if np.linalg.norm(p) <= R:
                    atoms.append(dict(cart=np.array([p[0], p[1], L * h]),
                                      layer=L, letter=ch))
    return atoms

def stack_layer_height(r):
    return r * 2 * math.sqrt(2) / math.sqrt(3)

def _rot(v, deg):
    t = math.radians(deg)
    return np.array([v[0] * math.cos(t) - v[1] * math.sin(t),
                     v[0] * math.sin(t) + v[1] * math.cos(t)])

def fcc_cube_in_stack(sequence, r=1.3, k0=0):
    """ABC堆积中的FCC立方晶胞8个角(需 sequence[k0:k0+4]='ABCA')"""
    if k0 + 3 >= len(sequence) or sequence[k0:k0 + 4] != 'ABCA':
        return None, False
    h = stack_layer_height(r)
    v1 = np.array([2 * r, 0.0]); v2 = np.array([r, r * math.sqrt(3)])
    hole1 = (v1 + v2) / 3.0
    d = [2 * hole1, 2 * _rot(hole1, 120), 2 * _rot(hole1, 240)]
    P0 = np.array([0.0, 0.0, k0 * h])
    corners = [P0]
    for i in range(3):
        corners.append(np.array([d[i][0], d[i][1], (k0 + 1) * h]))
    for i in range(3):
        for j in range(i + 1, 3):
            corners.append(np.array([d[i][0] + d[j][0], d[i][1] + d[j][1],
                                     (k0 + 2) * h]))
    corners.append(np.array([0.0, 0.0, (k0 + 3) * h]))
    return corners, True

def hcp_prism_in_stack(sequence, r=1.3, k0=0):
    """AB堆积中的教材式六棱柱(需 sequence[k0:k0+3]='ABA')"""
    if k0 + 2 >= len(sequence) or sequence[k0:k0 + 3] != 'ABA':
        return None, False
    a = 2 * r
    hexv = [(a * math.cos(math.radians(t)), a * math.sin(math.radians(t)))
            for t in (0, 60, 120, 180, 240, 300)]
    z0 = k0 * stack_layer_height(r)
    z1 = (k0 + 2) * stack_layer_height(r)
    return (hexv, z0, z1), True

# ============ 7. CIF 解析 ============
def _cif_tokens(line):
    out, i, n = [], 0, len(line)
    while i < n:
        while i < n and line[i] in ' \t':
            i += 1
        if i >= n:
            break
        if line[i] in "'\"":
            q = line[i]; i += 1; st = i
            while i < n and line[i] != q:
                i += 1
            out.append(line[st:i]); i += 1
        else:
            st = i
            while i < n and line[i] not in ' \t':
                i += 1
            out.append(line[st:i])
    return out

def _parse_symop(s):
    import re
    s = s.replace(' ', '').lower()
    if s.count(',') != 2:
        return None
    Mx = np.zeros((3, 3)); T = np.zeros(3)
    for r, p in enumerate(s.split(',')):
        p2 = p
        for var, idx in (('x', 0), ('y', 1), ('z', 2)):
            p2 = re.sub(r'(?<![0-9.])' + var + r'(?![0-9.])', f'V[{idx}]', p2)
        try:
            T[r] = float(eval(p2, {'__builtins__': {}}, {'V': np.zeros(3)}))
            for idx in range(3):
                e = {'V': np.zeros(3)}
                e['V'][idx] = 1.0
                Mx[r][idx] = float(eval(p2, {'__builtins__': {}}, e)) - T[r]
        except Exception:
            return None
    return Mx, T

def parse_cif(text, name=''):
    lines = text.splitlines()
    vals, loops = {}, []
    i = 0
    cur_name = name
    while i < len(lines):
        ln = lines[i].strip()
        if not ln or ln.startswith('#'):
            i += 1; continue
        if ln.lower().startswith('data_'):
            if not cur_name:
                cur_name = ln[5:].replace('_', ' ')
            i += 1; continue
        if ln.lower() == 'loop_':
            i += 1
            headers = []
            while i < len(lines) and lines[i].strip().startswith('_'):
                headers.append(lines[i].strip().lower()); i += 1
            rows = []
            while i < len(lines):
                s = lines[i].strip()
                if not s or s.startswith('#') or s.lower() == 'loop_' \
                   or s.startswith('_') or s.lower().startswith('data_'):
                    break
                toks = _cif_tokens(s)
                if len(toks) == len(headers) and len(headers) > 0:
                    rows.append(toks); i += 1
                elif len(toks) > len(headers) and len(headers) > 0:
                    while len(toks) >= len(headers):
                        rows.append(toks[:len(headers)])
                        toks = toks[len(headers):]
                    i += 1
                else:
                    break
            loops.append((headers, rows)); continue
        if ln.startswith('_'):
            toks = _cif_tokens(ln)
            if len(toks) >= 2:
                vals[toks[0].lower()] = ' '.join(toks[1:])
                i += 1
            else:
                key = toks[0].lower(); i += 1
                if i < len(lines):
                    vals[key] = lines[i].strip(); i += 1
            continue
        i += 1

    def num(key, default):
        v = vals.get(key)
        if v is None:
            return default
        try:
            return float(str(v).split('(')[0])
        except Exception:
            return default

    a = num('_cell_length_a', None)
    if a is None:
        raise ValueError('CIF 缺少晶胞参数 _cell_length_a')
    b = num('_cell_length_b', a); c = num('_cell_length_c', a)
    al = num('_cell_angle_alpha', 90.0); be = num('_cell_angle_beta', 90.0)
    ga = num('_cell_angle_gamma', 90.0)
    ops = []
    for headers, rows in loops:
        for hkey in ('_space_group_symop_operation_xyz',
                     '_symmetry_equiv_pos_as_xyz'):
            if hkey in headers:
                col = headers.index(hkey)
                for row in rows:
                    if col < len(row):
                        op = _parse_symop(row[col])
                        if op is not None and not any(
                                np.allclose(op[0], o[0]) and np.allclose(op[1], o[1])
                                for o in ops):
                            ops.append(op)
    sites = []
    for headers, rows in loops:
        if not any(h.startswith('_atom_site') for h in headers):
            continue
        idx = {h: headers.index(h) for h in headers}
        for row in rows:
            try:
                fx = float(row[idx['_atom_site_fract_x']].split('(')[0])
                fy = float(row[idx['_atom_site_fract_y']].split('(')[0])
                fz = float(row[idx['_atom_site_fract_z']].split('(')[0])
            except Exception:
                continue
            sym = row[idx['_atom_site_type_symbol']] \
                if '_atom_site_type_symbol' in idx else ''
            if not sym or sym in ('.', '?'):
                sym = row[idx['_atom_site_label']] \
                    if '_atom_site_label' in idx else 'X'
            sym = ''.join(ch for ch in sym if ch.isalpha())[:2]
            sym = sym[0].upper() + (sym[1:].lower() if len(sym) > 1 else '')
            sites.append((sym, (fx, fy, fz)))
    if not sites:
        raise ValueError('CIF 中没有可识别的原子坐标')
    basis, seen = [], set()
    for e, f in sites:
        p = np.array(f)
        for Mx, T in (ops or [(np.eye(3), np.zeros(3))]):
            q = Mx @ p + T
            q = q - np.floor(q + 1e-9)
            q = np.where(np.abs(q - 1) < 1e-9, 0.0, q)
            key = (e, round(q[0], 4), round(q[1], 4), round(q[2], 4))
            if key in seen:
                continue
            seen.add(key)
            basis.append((e, (float(q[0]), float(q[1]), float(q[2]))))
    return make_custom('cif_%d' % (abs(hash(cur_name)) % 99999),
                       cur_name or 'CIF结构', a, b, c, al, be, ga, 'P', basis,
                       notes='CIF导入：每晶胞%d个原子' % len(basis))

# ============ 8. 自测 ============
def _close(a, b, tol=0.02):
    return abs(a - b) <= tol

def run_tests(verbose=True):
    ok, fail = 0, []
    def T(name, cond):
        nonlocal ok
        if cond: ok += 1
        else: fail.append(name)
        if verbose: print(('  OK ' if cond else '  XX ') + name)

    print('== 晶格矢量 ==')
    for key in ('nacl', 'mg', 'rutile', 'graphite'):
        cr = DB[key]; M = lattice_matrix(cr)
        ang = lambda u, v: math.degrees(math.acos(
            np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
        T(f'{key}: |a|,|b|,|c|', _close(np.linalg.norm(M[0]), cr['a'], 1e-6) and
          _close(np.linalg.norm(M[1]), cr['b'], 1e-6) and
          _close(np.linalg.norm(M[2]), cr['c'], 1e-6))
        T(f'{key}: alpha,gamma', _close(ang(M[1], M[2]), cr['alpha'], 1e-3) and
          _close(ang(M[0], M[1]), cr['gamma'], 1e-3))

    print('== 均摊计数 ==')
    expect = {'po': {'Po': 1}, 'na': {'Na': 2}, 'k': {'K': 2}, 'fe': {'Fe': 2},
              'cu': {'Cu': 4}, 'mg': {'Mg': 2}, 'cscl': {'Cs': 1, 'Cl': 1},
              'nacl': {'Na': 4, 'Cl': 4}, 'caf2': {'Ca': 4, 'F': 8},
              'znsw': {'Zn': 4, 'S': 4}, 'znsz': {'Zn': 2, 'S': 2},
              'diamond': {'C': 8}, 'graphite': {'C': 4},
              'co2': {'C': 4, 'O': 8}, 'perov': {'Ca': 1, 'Ti': 1, 'O': 3},
              'na2o': {'Na': 8, 'O': 4}, 'rutile': {'Ti': 2, 'O': 4},
              'ar': {'Ar': 4}, 'mgo': {'Mg': 4, 'O': 4}}
    for k, exp in expect.items():
        T(f'Z({k})={exp}', apportioned_counts(DB[k]) == exp)

    print('== 教材式位置分解 ==')
    s = cell_atom_summary(DB['nacl'])
    T('NaCl: Cl 顶8×1/8+面6×1/2', s['Cl'].get('corner') == [8, .125] and
      s['Cl'].get('face') == [6, .5])
    T('NaCl: Na 体1+棱12×1/4', s['Na'].get('body') == [1, 1.0] and
      s['Na'].get('edge') == [12, .25])

    print('== 化学式 ==')
    for k, f0 in [('nacl', 'NaCl'), ('caf2', 'CaF₂'), ('znsw', 'ZnS'),
                  ('co2', 'CO₂'), ('perov', 'CaTiO₃'), ('na2o', 'Na₂O'),
                  ('rutile', 'TiO₂'), ('diamond', 'C'), ('cu', 'Cu')]:
        T(f'formula({k})={f0}', formula(DB[k]) == f0)

    print('== 密度 g/cm3 ==')
    for k, d0 in [('nacl', 2.16), ('cu', 8.91), ('fe', 7.87), ('mg', 1.74),
                  ('diamond', 3.51), ('cscl', 3.99), ('caf2', 3.18)]:
        T(f'rho({k})={d0}', _close(density(DB[k]), d0, 0.06))

    print('== 空间利用率 ==')
    for k, p0 in [('po', .5236), ('na', .6802), ('fe', .6802), ('cu', .7405),
                  ('mg', .7405), ('cscl', .68)]:
        T(f'eta({k})={p0}', _close(packing_fraction(DB[k]), p0, .012))
    T('eta(Mg六棱柱)', _close(packing_fraction(DB['mg'], prism=True), .7405, .012))

    print('== 配位数 ==')
    cn = coordination(DB['nacl'])
    T('NaCl 6:6', cn['Na'] == (6, {'Cl'}) and cn['Cl'] == (6, {'Na'}))
    cn = coordination(DB['cscl'])
    T('CsCl 8:8', cn['Cs'] == (8, {'Cl'}) and cn['Cl'] == (8, {'Cs'}))
    cn = coordination(DB['caf2'])
    T('CaF2 Ca=8 F=4', cn['Ca'] == (8, {'F'}) and cn['F'] == (4, {'Ca'}))
    cn = coordination(DB['znsw'])
    T('ZnS 4:4', cn['Zn'] == (4, {'S'}) and cn['S'] == (4, {'Zn'}))
    T('Cu 12', coordination(DB['cu'])['Cu'] == (12, {'Cu'}))
    T('Fe 8', coordination(DB['fe'])['Fe'] == (8, {'Fe'}))
    T('Po 6', coordination(DB['po'])['Po'] == (6, {'Po'}))
    T('C(金刚石) 4', coordination(DB['diamond'])['C'] == (4, {'C'}))
    T('CO2 C=2', coordination(DB['co2'])['C'] == (2, {'O'}))
    cn = coordination(DB['rutile'])
    T('金红石 Ti=6 O=3', cn['Ti'] == (6, {'O'}) and cn['O'] == (3, {'Ti'}))
    cn = coordination(DB['na2o'])
    T('Na2O Na=4 O=8', cn['Na'] == (4, {'O'}) and cn['O'] == (8, {'Na'}))
    T('钙钛矿 Ti=6(O)', coordination(DB['perov'])['Ti'][0] == 6)

    print('== 超晶胞显示原子数(教材图示数量) ==')
    T('NaCl 1^3=27', len(supercell_atoms(DB['nacl'], 1)) == 27)
    T('NaCl 2^3=125', len(supercell_atoms(DB['nacl'], 2)) == 125)
    T('Cu 1^3=14', len(supercell_atoms(DB['cu'], 1)) == 14)
    T('CsCl 1^3=9', len(supercell_atoms(DB['cscl'], 1)) == 9)
    T('Mg 1^3=9', len(supercell_atoms(DB['mg'], 1)) == 9)
    T('ZnS 1^3=18', len(supercell_atoms(DB['znsw'], 1)) == 18)
    T('金刚石 1^3=18', len(supercell_atoms(DB['diamond'], 1)) == 18)
    T('CaF2 1^3=22', len(supercell_atoms(DB['caf2'], 1)) == 22)
    pos = [tuple(np.round(x['cart'], 4)) for x in supercell_atoms(DB['nacl'], 2)]
    T('NaCl 2^3 无重复', len(set(pos)) == len(pos))

    print('== 化学键(显示原子之间) ==')
    T('金刚石 16键', len(bonds_for(DB['diamond'],
          supercell_atoms(DB['diamond'], 1))) == 16)
    T('NaCl 54键', len(bonds_for(DB['nacl'],
          supercell_atoms(DB['nacl'], 1))) == 54)
    T('CO2 8键', len(bonds_for(DB['co2'], supercell_atoms(DB['co2'], 1))) == 8)
    T('石墨 9键(底层3+中层3+顶面镜像层3)',
      len(bonds_for(DB['graphite'],
          supercell_atoms(DB['graphite'], 1))) == 9)
    T('Cu 0键', len(bonds_for(DB['cu'], supercell_atoms(DB['cu'], 1))) == 0)
    T('CaF2 32键', len(bonds_for(DB['caf2'],
          supercell_atoms(DB['caf2'], 1))) == 32)

    print('== 六棱柱 ==')
    pa = prism_atoms(DB['mg'], 1)
    nv = sum(1 for x in pa if x['ptype'] == 'prism_vertex')
    nf = sum(1 for x in pa if x['ptype'] == 'prism_face')
    nb = sum(1 for x in pa if x['ptype'] == 'prism_body')
    T('Mg六棱柱=12+2+3', (nv, nf, nb) == (12, 2, 3))
    T('Mg六棱柱均摊=6', prism_apportioned(DB['mg']) == {'Mg': 6.0})

    print('== 堆积演示 ==')
    st = build_stack('ABCA', r=1.3, extent=3)
    L0 = [x for x in st if x['layer'] == 0][0]
    L1 = [x for x in st if x['layer'] == 1]
    dmin = min(np.linalg.norm(L0['cart'] - y['cart']) for y in L1)
    T('层间最近距离=2r', _close(dmin, 2.6, .01))
    corners, okk = fcc_cube_in_stack('ABCABCA', 1.3, k0=0)
    T('FCC立方8角', okk and len(corners) == 8)
    if okk:
        st2 = build_stack('ABCABCA', 1.3, extent=4)
        pts2 = np.array([x['cart'] for x in st2])
        hit = sum(1 for cp in corners
                  if np.min(np.linalg.norm(pts2 - cp, axis=1)) < 1e-6)
        T('FCC立方8角都在原子上', hit == 8)
        e = np.linalg.norm(np.array(corners[1]) - np.array(corners[0]))
        T('棱长=2*sqrt(2)*r', _close(e, 2 * 1.3 * math.sqrt(2), .01))
    hp, okk = hcp_prism_in_stack('ABABA', 1.3, k0=0)
    T('六棱柱框架', okk)
    if okk:
        hexv, z0, z1 = hp
        T('棱柱高=2层', _close(z1 - z0, 2 * stack_layer_height(1.3), .01))
        T('棱柱边=2r', all(_close(math.hypot(hexv[i][0] - hexv[(i + 1) % 6][0],
                                            hexv[i][1] - hexv[(i + 1) % 6][1]),
                            2.6, .01) for i in range(6)))

    print('== 空隙 ==')
    o, t = fcc_holes(DB['nacl'])
    T('NaCl 八面体13个全Na(1体心+12棱镜像)',
      len(o) == 13 and all(x[1] == 'Na' for x in o))
    T('NaCl 四面体8全空', len(t) == 8 and all(x[1] is None for x in t))
    o, t = fcc_holes(DB['caf2'])
    T('CaF2 八面体13个全空', len(o) == 13 and all(x[1] is None for x in o))
    T('CaF2 四面体8全F', len(t) == 8 and all(x[1] == 'F' for x in t))
    o, t = fcc_holes(DB['znsw'])
    T('ZnS 四面体8个,4被Zn占', len(t) == 8 and
      sum(1 for x in t if x[1] == 'Zn') == 4)
    o, t = fcc_holes(DB['diamond'])
    T('金刚石 4被C占', len(t) == 8 and sum(1 for x in t if x[1] == 'C') == 4)

    print('== 凸包 ==')
    tet = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    T('四面体4面', len(convex_hull_faces(tet)) == 4)
    octa = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    T('八面体8面', len(convex_hull_faces(octa)) == 8)
    cube = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    T('立方24三角面(6正方面,暴力凸包每面4个共面三角)',
      len(convex_hull_faces(cube)) == 24)

    print('== 配位多面体 ==')
    T('ZnS 4个[ZnS4]', len(coordination_polyhedra(DB['znsw'],
          supercell_atoms(DB['znsw'], 1))) == 4)
    T('NaCl 1个完整[NaCl6]', len(coordination_polyhedra(DB['nacl'],
          supercell_atoms(DB['nacl'], 1))) == 1)

    print('== CIF ==')
    cif_text = """
data_NaCl_test
_cell_length_a 5.640
_cell_length_b 5.640
_cell_length_c 5.640
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_operation_xyz
'x, y, z'
'x, y+1/2, z+1/2'
'x+1/2, y, z+1/2'
'x+1/2, y+1/2, z'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cl 0 0 0
Na 0.5 0.5 0.5
"""
    cr = parse_cif(cif_text, 'NaCl-CIF')
    T('CIF 8原子', len(cr['basis']) == 8)
    T('CIF Na4Cl4', apportioned_counts(cr) == {'Na': 4, 'Cl': 4})
    T('CIF 密度一致', _close(density(cr), density(DB['nacl']), .02))
    cif2 = """
data_CsCl
_cell_length_a 4.123
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cl1 Cl 0 0 0
Cs1 Cs 0.5 0.5 0.5
"""
    cr2 = parse_cif(cif2)
    T('CIF2 CsCl', apportioned_counts(cr2) == {'Cs': 1, 'Cl': 1})
    T('CIF2 CN 8:8', coordination(cr2)['Cs'] == (8, {'Cl'}))

    print('== 最近邻距离 ==')
    md = min_distances(DB['cu'])
    T('Cu-Cu=a/sqrt2', _close(md[('Cu', 'Cu')], 3.615 / math.sqrt(2), .01))
    md = min_distances(DB['nacl'])
    T('Na-Cl=a/2', _close(md[('Cl', 'Na')], 5.640 / 2, .01))
    md = min_distances(DB['fe'])
    T('Fe-Fe=sqrt3*a/2', _close(md[('Fe', 'Fe')], 2.866 * math.sqrt(3) / 2, .01))

    print(f'\n通过 {ok} 项，失败 {len(fail)} 项')
    if fail:
        print('失败:', fail)
    return len(fail) == 0

if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        sys.exit(0 if run_tests() else 1)
    print(f'晶体库 {len(DB)} 种。运行 python crystal_core.py --test 自测。')
