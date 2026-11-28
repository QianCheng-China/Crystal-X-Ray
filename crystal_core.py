# -*- coding: utf-8 -*-
"""
Crystal X-Ray v0.2 —— 晶体学计算引擎（纯 Python+NumPy，无 GUI 依赖）
开发者: HighspeedG2304
晶体库: 93 种，按"晶系类别 > 结构原型"组织
自测: python crystal_core.py --test
"""
import math
import numpy as np

AVOGADRO = 6.02214076e23

# ============ 1. 元素表: (中文名, 摩尔质量, 颜色, 默认显示半径Å) ============
ELEMENTS = {
 'H': ('氢', 1.008, '#F9F9F9', 0.32), 'He': ('氦', 4.003, '#D9FFFF', 1.20),
 'Li': ('锂', 6.94, '#CC80FF', 1.52), 'Be': ('铍', 9.012, '#C2FF00', 1.12),
 'B': ('硼', 10.81, '#FFB5B5', 0.82), 'C': ('碳', 12.011, '#4A4A4A', 0.77),
 'N': ('氮', 14.007, '#3050F8', 0.75), 'O': ('氧', 15.999, '#FF0D0D', 0.73),
 'F': ('氟', 18.998, '#90E050', 1.31), 'Ne': ('氖', 20.18, '#B3E3F5', 1.60),
 'Na': ('钠', 22.990, '#AB5CF2', 1.02), 'Mg': ('镁', 24.305, '#8AFF00', 1.60),
 'Al': ('铝', 26.982, '#BFA6A6', 1.43), 'Si': ('硅', 28.086, '#F0C8A0', 1.11),
 'P': ('磷', 30.974, '#FF8000', 1.07), 'S': ('硫', 32.06, '#FFE22A', 1.84),
 'Cl': ('氯', 35.45, '#1FF01F', 1.81), 'Ar': ('氩', 39.948, '#80D1E3', 1.88),
 'K': ('钾', 39.098, '#8F40D4', 1.38), 'Ca': ('钙', 40.078, '#3DFF00', 1.00),
 'Ti': ('钛', 47.867, '#BFC2C7', 0.68), 'V': ('钒', 50.942, '#A2A2A2', 1.31),
 'Cr': ('铬', 51.996, '#8A99C7', 1.25), 'Mn': ('锰', 54.938, '#9C7AC7', 0.53),
 'Fe': ('铁', 55.845, '#E06633', 1.24), 'Co': ('钴', 58.933, '#F090A0', 1.25),
 'Ni': ('镍', 58.693, '#50D050', 0.69), 'Cu': ('铜', 63.546, '#C88033', 1.28),
 'Zn': ('锌', 65.38, '#7D80B0', 0.74), 'Ga': ('镓', 69.723, '#828282', 1.22),
 'Ge': ('锗', 72.63, '#668F8F', 1.22), 'As': ('砷', 74.922, '#BD80E3', 1.19),
 'Se': ('硒', 78.971, '#FFA100', 1.20), 'Br': ('溴', 79.904, '#A62929', 1.96),
 'Kr': ('氪', 83.798, '#5CB8D6', 2.00), 'Rb': ('铷', 85.468, '#702EB7', 2.44),
 'Sr': ('锶', 87.62, '#00FF00', 2.15), 'Zr': ('锆', 91.224, '#7DA29D', 1.61),
 'Nb': ('铌', 92.906, '#7DA2BC', 1.43), 'Mo': ('钼', 95.95, '#54B5B5', 1.36),
 'Ru': ('钌', 101.07, '#B5B5B5', 1.34), 'Rh': ('铑', 102.91, '#A0A0A0', 1.34),
 'Pd': ('钯', 106.42, '#B8B8B8', 1.37), 'Ag': ('银', 107.87, '#C0C0C0', 1.15),
 'Cd': ('镉', 112.41, '#FFD98F', 0.95), 'In': ('铟', 114.82, '#A67573', 1.44),
 'Sn': ('锡', 118.71, '#668080', 1.40), 'Sb': ('锑', 121.76, '#9E63B5', 1.36),
 'Te': ('碲', 127.60, '#D47A00', 1.38), 'I': ('碘', 126.90, '#940094', 2.20),
 'Xe': ('氙', 131.29, '#429EB0', 2.19), 'Cs': ('铯', 132.91, '#57178F', 1.76),
 'Ba': ('钡', 137.33, '#00C900', 1.35), 'Ce': ('铈', 140.12, '#B8A092', 1.01),
 'W': ('钨', 183.84, '#21994D', 1.37), 'Pt': ('铂', 195.08, '#D6D6D6', 1.39),
 'Au': ('金', 196.97, '#FFD123', 1.44), 'Hg': ('汞', 200.59, '#B8B8D0', 1.49),
 'Tl': ('铊', 204.38, '#A6544A', 1.50), 'Pb': ('铅', 207.2, '#575961', 1.19),
 'Bi': ('铋', 208.98, '#9E4FE0', 1.54), 'Po': ('钋', 209.0, '#AB5C00', 1.68),
 'U': ('铀', 238.03, '#4E7A4E', 0.89),
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

# ============ 2. 晶体库（93种，按结构原型） ============
def _cr(key, name, cat, sysm, abc, angles=(90, 90, 90), centering='P',
        basis=(), radii=None, bonds=None, poly=(), holes=None,
        prism=False, lattice='', notes='', prototype='',
        formula_override=None, mass_override=None):
    a, b, c = (list(abc) + [None, None, None])[:3]
    if b is None: b = a
    if c is None: c = a
    d = dict(key=key, name=name, category=cat, system=sysm, a=float(a),
             b=float(b), c=float(c), alpha=angles[0], beta=angles[1],
             gamma=angles[2], centering=centering,
             basis=[(e, (float(x), float(y), float(z))) for e, (x, y, z) in basis],
             radii=radii or {}, bonds=bonds, poly=list(poly), holes=holes,
             prism=prism, lattice=lattice, notes=notes, prototype=prototype,
             custom=False)
    if formula_override is not None:
        d['formula_override'] = formula_override
    if mass_override is not None:
        d['mass_override'] = mass_override
    return d


# ---- 结构原型生成器 ----
def _bcc(key, name, e, a, r, notes=''):
    return _cr(key, name, '金属晶体', '立方', (a,), centering='I',
               basis=[(e, (0, 0, 0))], radii={e: r}, bonds=[],
               prototype='体心立方(bcc)', lattice='体心立方(I)', notes=notes)

def _fcc_m(key, name, e, a, r, notes=''):
    return _cr(key, name, '金属晶体', '立方', (a,), centering='F',
               basis=[(e, (0, 0, 0))], radii={e: r}, bonds=[],
               prototype='面心立方(fcc)', lattice='面心立方(F)', notes=notes)

def _hcp_m(key, name, e, a, c, r, notes=''):
    return _cr(key, name, '金属晶体', '六方', (a, None, c), (90, 90, 120),
               centering='P', basis=[(e, (0, 0, 0)), (e, (2/3, 1/3, .5))],
               radii={e: r}, bonds=[], prism=True,
               prototype='六方最密(hcp)', lattice='简单六方格子', notes=notes)

def _rs(key, name, c1, c2, a, r1, r2, notes=''):
    return _cr(key, name, '离子晶体', '立方', (a,), centering='F',
               basis=[(c1, (0, 0, 0)), (c2, (.5, .5, .5))],
               radii={c1: r1, c2: r2}, bonds=[(c1, c2)], poly=[(c1, c2)],
               holes=c2, prototype='岩盐型', lattice='面心立方(F)', notes=notes)

def _cs(key, name, an, c, a, r1, r2, notes=''):
    return _cr(key, name, '离子晶体', '立方', (a,), centering='P',
               basis=[(an, (0, 0, 0)), (c, (.5, .5, .5))],
               radii={an: r1, c: r2}, bonds=[(c, an)], poly=[(c, an)],
               prototype='氯化铯型', lattice='简单立方(P)', notes=notes)

def _fl(key, name, c, an, a, r1, r2, notes=''):
    return _cr(key, name, '离子晶体', '立方', (a,), centering='F',
               basis=[(c, (0, 0, 0)), (an, (.25, .25, .25)), (an, (.75, .75, .75))],
               radii={c: r1, an: r2}, bonds=[(c, an)], poly=[(an, c)],
               holes=c, prototype='萤石型', lattice='面心立方(F)', notes=notes)

def _anti(key, name, c, an, a, r1, r2, notes=''):
    return _cr(key, name, '离子晶体', '立方', (a,), centering='F',
               basis=[(an, (0, 0, 0)), (c, (.25, .25, .25)), (c, (.75, .75, .75))],
               radii={c: r1, an: r2}, bonds=[(c, an)], poly=[(c, an)],
               holes=an, prototype='反萤石型', lattice='面心立方(F)', notes=notes)

def _zb(key, name, c, an, a, r1, r2, notes='', cat='离子晶体'):
    return _cr(key, name, cat, '立方', (a,), centering='F',
               basis=[(an, (0, 0, 0)), (c, (.25, .25, .25))],
               radii={c: r1, an: r2}, bonds=None, poly=[(c, an)],
               holes=an, prototype='闪锌矿型', lattice='面心立方(F)', notes=notes)

def _wz(key, name, c, an, a, c0, r1, r2, notes='', u=0.375):
    return _cr(key, name, '离子晶体', '六方', (a, None, c0), (90, 90, 120),
               centering='P',
               basis=[(c, (1/3, 2/3, 0)), (c, (2/3, 1/3, .5)),
                      (an, (1/3, 2/3, u)), (an, (2/3, 1/3, u + .5))],
               radii={c: r1, an: r2}, bonds=[(c, an)], poly=[(c, an)],
               prototype='纤锌矿型', lattice='简单六方格子', notes=notes)

def _rt(key, name, c, an, a, c0, r1, r2, notes=''):
    u = 0.305
    return _cr(key, name, '离子晶体', '四方', (a, None, c0), centering='P',
               basis=[(c, (0, 0, 0)), (c, (.5, .5, .5)),
                      (an, (u, u, 0)), (an, (1 - u, 1 - u, 0)),
                      (an, (.5 + u, .5 - u, .5)), (an, (.5 - u, .5 + u, .5))],
               radii={c: r1, an: r2}, bonds=[(c, an)], poly=[(c, an)],
               prototype='金红石型', lattice='简单四方(P)', notes=notes)

def _pv(key, name, A, B, X, a, rA, rB, rX, notes=''):
    return _cr(key, name, '离子晶体', '立方', (a,), centering='P',
               basis=[(A, (0, 0, 0)), (B, (.5, .5, .5)),
                      (X, (.5, .5, 0)), (X, (.5, 0, .5)), (X, (0, .5, .5))],
               radii={A: rA, B: rB, X: rX}, bonds=[(B, X)], poly=[(B, X)],
               prototype='钙钛矿型', lattice='简单立方(P)', notes=notes)

def _dia(key, name, e, a, r, notes='', cat='共价晶体'):
    return _cr(key, name, cat, '立方', (a,), centering='F',
               basis=[(e, (0, 0, 0)), (e, (.25, .25, .25))], radii={e: r},
               bonds=None, poly=[(e, e)], holes=e,
               prototype='金刚石型', lattice='面心立方(F)', notes=notes)

CRYSTAL_DB = [
 # ---- 金属晶体 (30) ----
 _cr('po', '钋 Po', '金属晶体', '立方', (3.35,), centering='P',
     basis=[('Po', (0, 0, 0))], radii={'Po': 1.68}, bonds=[],
     prototype='简单立方(sc)', lattice='简单立方(P)',
     notes='顶点8×1/8=1；CN=6；η=52.36%，唯一简单立方金属。'),
 _bcc('li', '锂 Li', 'Li', 3.51, 1.52),
 _bcc('na', '钠 Na', 'Na', 4.29, 1.86),
 _bcc('k', '钾 K', 'K', 5.33, 2.31),
 _bcc('rb', '铷 Rb', 'Rb', 5.59, 2.44),
 _bcc('cs', '铯 Cs', 'Cs', 6.05, 2.62),
 _bcc('v', '钒 V', 'V', 3.02, 1.31),
 _bcc('nb', '铌 Nb', 'Nb', 3.30, 1.43),
 _bcc('cr', '铬 Cr', 'Cr', 2.884, 1.25),
 _bcc('mo', '钼 Mo', 'Mo', 3.15, 1.36),
 _bcc('fe', '铁 Fe', 'Fe', 2.866, 1.24, 'α-铁：Z=2，CN=8，η=68.02%。'),
 _bcc('w', '钨 W', 'W', 3.165, 1.37, '熔点最高的金属。'),
 _fcc_m('ca', '钙 Ca', 'Ca', 5.58, 1.97),
 _fcc_m('sr', '锶 Sr', 'Sr', 6.08, 2.15),
 _fcc_m('al', '铝 Al', 'Al', 4.049, 1.43),
 _fcc_m('ni', '镍 Ni', 'Ni', 3.524, 1.24),
 _fcc_m('cu', '铜 Cu', 'Cu', 3.615, 1.28, '8×1/8+6×1/2=4；CN=12；η=74.05%（ABCABC）。'),
 _fcc_m('rh', '铑 Rh', 'Rh', 3.80, 1.34),
 _fcc_m('pd', '钯 Pd', 'Pd', 3.89, 1.37),
 _fcc_m('ag', '银 Ag', 'Ag', 4.086, 1.44),
 _fcc_m('pt', '铂 Pt', 'Pt', 3.924, 1.39),
 _fcc_m('au', '金 Au', 'Au', 4.078, 1.44),
 _fcc_m('pb', '铅 Pb', 'Pb', 4.95, 1.75),
 _hcp_m('be', '铍 Be', 'Be', 2.29, 3.58, 1.12),
 _hcp_m('mg', '镁 Mg', 'Mg', 3.21, 5.21, 1.60,
        '平行六面体Z=2；教材六棱柱=6；CN=12；η=74.05%（ABAB）。'),
 _hcp_m('ti', '钛 Ti', 'Ti', 2.95, 4.68, 1.47),
 _hcp_m('co', '钴 Co', 'Co', 2.51, 4.07, 1.25),
 _hcp_m('zn', '锌 Zn', 'Zn', 2.66, 4.95, 1.33,
        '理想c/a=1.633，锌1.86（畸变六方）。'),
 _hcp_m('zr', '锆 Zr', 'Zr', 3.23, 5.15, 1.61),
 _hcp_m('cd', '镉 Cd', 'Cd', 2.98, 5.62, 1.49),
 # ---- 离子晶体 (45) ----
 _rs('lif', '氟化锂 LiF', 'Li', 'F', 4.03, .76, 1.31),
 _rs('naf', '氟化钠 NaF', 'Na', 'F', 4.64, 1.02, 1.31),
 _cr('nacl', '氯化钠 NaCl', '离子晶体', '立方', (5.640,), centering='F',
     basis=[('Cl', (0, 0, 0)), ('Na', (.5, .5, .5))],
     radii={'Na': 1.02, 'Cl': 1.81}, bonds=[('Na', 'Cl')], poly=[('Na', 'Cl')],
     holes='Cl', prototype='岩盐型', lattice='面心立方(F)',
     notes='Cl⁻:8×1/8+6×1/2=4；Na⁺:12×1/4+1=4；6:6；Na⁺填满八面体空隙。'),
 _rs('nabr', '溴化钠 NaBr', 'Na', 'Br', 5.97, 1.02, 1.96),
 _rs('kf', '氟化钾 KF', 'K', 'F', 4.67, 1.05, 1.28),
 _rs('kcl', '氯化钾 KCl', 'K', 'Cl', 6.29, 1.38, 1.81),
 _rs('kbr', '溴化钾 KBr', 'K', 'Br', 6.60, 1.38, 1.96),
 _rs('licl', '氯化锂 LiCl', 'Li', 'Cl', 5.13, .76, 1.81),
 _rs('mgo', '氧化镁 MgO', 'Mg', 'O', 4.212, .72, 1.40, '岩盐型，6:6。'),
 _rs('cao', '氧化钙 CaO', 'Ca', 'O', 4.81, 1.00, 1.40),
 _rs('sro', '氧化锶 SrO', 'Sr', 'O', 5.16, 1.18, 1.40),
 _rs('bao', '氧化钡 BaO', 'Ba', 'O', 5.54, 1.35, 1.40),
 _rs('mgs', '硫化镁 MgS', 'Mg', 'S', 5.22, .72, 1.84),
 _rs('pbs', '硫化铅 PbS（方铅矿）', 'Pb', 'S', 5.94, 1.19, 1.84),
 _rs('agcl', '氯化银 AgCl', 'Ag', 'Cl', 5.55, 1.15, 1.81),
 _rs('nio', '氧化镍 NiO', 'Ni', 'O', 4.18, .69, 1.40),
 _cs('cscl', '氯化铯 CsCl', 'Cl', 'Cs', 4.123, 1.81, 1.76,
     'Cl⁻占顶点+Cs⁺体心；8:8；两套简单立方穿插，非体心立方格子。'),
 _cs('csbr', '溴化铯 CsBr', 'Br', 'Cs', 4.30, 1.96, 1.76),
 _cs('csi', '碘化铯 CsI', 'I', 'Cs', 4.57, 2.20, 1.76),
 _cs('tlcl', '氯化铊 TlCl', 'Cl', 'Tl', 3.84, 1.81, 1.50),
 _fl('caf2', '氟化钙 CaF₂（萤石）', 'Ca', 'F', 5.463, 1.09, 1.28,
     'Ca²⁺4；F⁻8（填满全部四面体空隙）；Ca:8, F:4。'),
 _fl('srf2', '氟化锶 SrF₂', 'Sr', 'F', 5.80, 1.18, 1.31),
 _fl('baf2', '氟化钡 BaF₂', 'Ba', 'F', 6.20, 1.35, 1.31),
 _fl('ceo2', '二氧化铈 CeO₂', 'Ce', 'O', 5.41, 1.01, 1.40),
 _fl('uo2', '二氧化铀 UO₂', 'U', 'O', 5.47, .89, 1.40),
 _anti('li2o', '氧化锂 Li₂O', 'Li', 'O', 4.61, .76, 1.40),
 _anti('na2o', '氧化钠 Na₂O', 'Na', 'O', 5.56, 1.02, 1.40,
     '反萤石：Na⁺填满四面体空隙；Na:4, O:8（与CaF₂互换）。'),
 _anti('k2o', '氧化钾 K₂O', 'K', 'O', 6.44, 1.38, 1.40),
 _zb('znsw', '硫化锌 ZnS（闪锌矿）', 'Zn', 'S', 5.41, .90, 1.45,
     'S²⁻面心立方，Zn²⁺占一半四面体空隙；4:4；与金刚石同型。'),
 _zb('cucl', '氯化亚铜 CuCl', 'Cu', 'Cl', 5.42, .96, 1.81),
 _zb('cubr', '溴化亚铜 CuBr', 'Cu', 'Br', 5.69, .96, 1.96),
 _wz('znsz', '硫化锌 ZnS（纤锌矿）', 'Zn', 'S', 3.82, 6.26, .90, 1.45,
     '六方ZnS，4:4，Z=2。'),
 _wz('zno', '氧化锌 ZnO（红锌矿）', 'Zn', 'O', 3.25, 5.21, .74, 1.40, u=.3825),
 _wz('cds', '硫化镉 CdS', 'Cd', 'S', 4.14, 6.72, .95, 1.84),
 _wz('aln', '氮化铝 AlN', 'Al', 'N', 3.11, 4.98, .53, 1.46, u=.382),
 _wz('beo', '氧化铍 BeO', 'Be', 'O', 2.70, 4.39, .45, 1.40, u=.378),
 _wz('gan', '氮化镓 GaN', 'Ga', 'N', 3.19, 5.19, .62, 1.46, u=.377),
 _rt('rutile', '二氧化钛 TiO₂（金红石）', 'Ti', 'O', 4.594, 2.958, .68, 1.32,
     'Ti:6（TiO₆八面体），O:3；白色颜料。'),
 _rt('sno2', '二氧化锡 SnO₂（锡石）', 'Sn', 'O', 4.74, 3.19, .69, 1.40),
 _rt('mno2', '二氧化锰 MnO₂（软锰矿）', 'Mn', 'O', 4.39, 2.87, .53, 1.40),
 _rt('mgf2', '氟化镁 MgF₂', 'Mg', 'F', 4.62, 3.05, .72, 1.31),
 _pv('perov', '钙钛矿 CaTiO₃', 'Ca', 'Ti', 'O', 3.84, 1.35, .68, 1.32,
     'Ca:1 Ti:1 O:6×1/2=3；TiO₆共顶点；BaTiO₃同型。'),
 _pv('batio3', '钛酸钡 BaTiO₃', 'Ba', 'Ti', 'O', 4.00, 1.35, .68, 1.32,
     'Ba:12配位，Ti:6，O:2；铁电材料。'),
 _pv('srtio3', '钛酸锶 SrTiO₃', 'Sr', 'Ti', 'O', 3.91, 1.18, .68, 1.32),
 _pv('kmgf3', '氟化镁钾 KMgF₃', 'K', 'Mg', 'F', 3.97, 1.38, .72, 1.31),
 # ---- 共价晶体 (13) ----
 _dia('diamond', '金刚石 C', 'C', 3.567, .77,
      '面心立方8+内部4=8；sp³；CN=4；键长1.55Å=a√3/4；η=34%。'),
 _dia('si', '硅 Si', 'Si', 5.431, 1.11, '半导体基石，键长2.35Å。'),
 _dia('ge', '锗 Ge', 'Ge', 5.658, 1.22),
 _dia('sn', '灰锡 α-Sn', 'Sn', 6.49, 1.40),
 _zb('sic', '碳化硅 SiC', 'Si', 'C', 4.36, 1.11, .77, cat='共价晶体'),
 _zb('bn', '氮化硼 BN（立方）', 'B', 'N', 3.615, .82, .75, cat='共价晶体',
     notes='俗称白石墨，硬度接近金刚石。'),
 _zb('gaas', '砷化镓 GaAs', 'Ga', 'As', 5.65, 1.22, 1.19, cat='共价晶体'),
 _zb('insb', '锑化铟 InSb', 'In', 'Sb', 6.48, 1.44, 1.36, cat='共价晶体'),
 _zb('cdte', '碲化镉 CdTe', 'Cd', 'Te', 6.48, .95, 2.20, cat='共价晶体'),
 _cr('graphite', '石墨 C', '共价晶体', '六方', (2.46, None, 6.71), (90, 90, 120),
     centering='P',
     basis=[('C', (0, 0, 0)), ('C', (1/3, 2/3, 0)),
            ('C', (0, 0, .5)), ('C', (2/3, 1/3, .5))],
     radii={'C': .77}, bonds=[('C', 'C')], prototype='层状',
     lattice='简单六方格子',
     notes='层内sp² 1.42Å；层间3.35Å范德华力→质软、导电。'),
 _cr('hbn', '氮化硼 h-BN（六方）', '共价晶体', '六方', (2.504, None, 6.66),
     (90, 90, 120), centering='P',
     basis=[('B', (0, 0, 0)), ('N', (1/3, 2/3, 0)),
            ('B', (0, 0, .5)), ('N', (2/3, 1/3, .5))],
     radii={'B': .82, 'N': .75}, bonds=[('B', 'N')], prototype='层状',
     lattice='简单六方格子', notes='类石墨层状，层间无共价。'),
 _cr('mos2', '二硫化钼 MoS₂', '共价晶体', '六方', (3.16, None, 12.30),
     (90, 90, 120), centering='P',
     basis=[('Mo', (0, 0, 0)), ('Mo', (0, 0, .5)),
            ('S', (1/3, 2/3, .129)), ('S', (2/3, 1/3, .371)),
            ('S', (1/3, 2/3, .629)), ('S', (2/3, 1/3, .871))],
     radii={'Mo': .70, 'S': 1.84}, bonds=[('Mo', 'S')], poly=[('Mo', 'S')],
     prototype='层状', lattice='简单六方格子',
     notes='S-Mo-S三明治层，Mo为三棱柱6配位；固体润滑剂。'),
 _cr('cdi2', '碘化镉 CdI₂', '共价晶体', '六方', (4.24, None, 6.84),
     (90, 90, 120), centering='P',
     basis=[('Cd', (0, 0, 0)), ('I', (1/3, 2/3, .25)), ('I', (2/3, 1/3, .75))],
     radii={'Cd': .95, 'I': 2.20}, bonds=[('Cd', 'I')], poly=[('Cd', 'I')],
     prototype='层状', lattice='简单六方格子',
     notes='Cd八面体6配位，I为3配位；层状CdI₂型。'),
 # ---- 分子晶体 (5) ----
 _cr('co2', '干冰 CO₂', '分子晶体', '立方', (5.62,), centering='P',
     basis=(('C', (0, 0, 0)), ('C', (0, .5, .5)), ('C', (.5, 0, .5)),
            ('C', (.5, .5, 0)),
            ('O', (.119, .119, .119)), ('O', (-.119, -.119, -.119)),
            ('O', (.119, .5 - .119, .5 - .119)),
            ('O', (-.119, .5 + .119, .5 + .119)),
            ('O', (.5 - .119, .119, .5 - .119)),
            ('O', (.5 + .119, -.119, .5 + .119)),
            ('O', (.5 - .119, .5 - .119, .119)),
            ('O', (.5 + .119, .5 + .119, -.119))),
     radii={'C': .77, 'O': .73}, bonds=[('C', 'O')], prototype='干冰型',
     lattice='面心立方排列的分子',
     notes='直线形CO₂分子占面心立方位置，轴沿体对角线；Z=4分子。'),
 _cr('ar', '固态氩 Ar', '分子晶体', '立方', (5.31,), centering='F',
     basis=[('Ar', (0, 0, 0))], radii={'Ar': 1.88}, bonds=[],
     prototype='面心密堆积', lattice='面心立方(F)', notes='单原子分子fcc密堆积。'),
 _cr('kr', '固态氪 Kr', '分子晶体', '立方', (5.65,), centering='F',
     basis=[('Kr', (0, 0, 0))], radii={'Kr': 2.00}, bonds=[],
     prototype='面心密堆积', lattice='面心立方(F)'),
 _cr('xe', '固态氙 Xe', '分子晶体', '立方', (6.20,), centering='F',
     basis=[('Xe', (0, 0, 0))], radii={'Xe': 2.19}, bonds=[],
     prototype='面心密堆积', lattice='面心立方(F)'),
 _cr('ice', '冰 Ih（H₂O）', '分子晶体', '六方', (4.52, None, 7.35), (90, 90, 120),
     centering='P',
     basis=[('O', (1/3, 2/3, 0)), ('O', (2/3, 1/3, .5)),
            ('O', (1/3, 2/3, .375)), ('O', (2/3, 1/3, .875))],
     radii={'O': .73}, bonds=[('O', 'O')], prototype='冰Ih',
     lattice='简单六方格子', formula_override='H₂O', mass_override=72.06,
     notes='O骨架类纤锌矿；O–O 2.76Å，四面体角≈109.5°；H位置无序未画出。'),
]


def make_custom(key, name, a, b, c, alpha, beta, gamma, centering, basis,
                bonds=None, notes=''):
    """兼容 (元素,(x,y,z)) 与 (元素,x,y,z) 两种格式"""
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
                basis=norm, radii={}, bonds=bonds, poly=[], holes=None,
                prism=False, lattice='自定义', prototype='', custom=True,
                notes=notes or '自定义晶体')

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

F_BASIS = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
I_BASIS = [(0, 0, 0), (.5, .5, .5)]
A_BASIS = [(0, 0, 0), (0, .5, .5)]
B_BASIS = [(0, 0, 0), (.5, 0, .5)]
C_BASIS = [(0, 0, 0), (.5, .5, 0)]
R_BASIS = [(0, 0, 0), (2/3, 1/3, 1/3), (1/3, 2/3, 2/3)]
CENTERING_BASIS = {'P': [(0, 0, 0)], 'I': I_BASIS, 'F': F_BASIS,
                   'A': A_BASIS, 'B': B_BASIS, 'C': C_BASIS, 'R': R_BASIS}

def unit_sites(cr):
    """晶胞内独立原子位置：基元×格子心化，折叠[0,1)，去重"""
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
    u, v, w = frac
    us = [u, 1.0] if abs(u) < 1e-9 else [u]
    vs = [v, 1.0] if abs(v) < 1e-9 else [v]
    ws = [w, 1.0] if abs(w) < 1e-9 else [w]
    return [(x, y, z) for x in us for y in vs for z in ws]

def supercell_atoms(cr, n=1, closed=True):
    """n×n×n 超晶胞原子。closed=True 时含闭胞边界镜像（教材图示约定）"""
    M = lattice_matrix(cr)
    atoms, seen = [], set()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for e, f in unit_sites(cr):
                    imgs = closed_cell_images(f) if closed else [f]
                    for img in imgs:
                        s = (i + img[0], j + img[1], k + img[2])
                        cart = M.T @ np.array(s)
                        key = tuple(np.round(cart, 4))
                        if key in seen:
                            continue
                        seen.add(key)
                        pt = pos_type(f)
                        atoms.append(dict(
                            elem=e, xyz=s, cart=cart, cell=(i, j, k), frac=f,
                            ptype=pt, share=POS_TYPE[pt][2],
                            nshare=POS_TYPE[pt][1]))
    return atoms

def apportioned_counts(cr):
    cnt = {}
    for e, f in unit_sites(cr):
        cnt[e] = cnt.get(e, 0) + 1
    return cnt

def cell_atom_summary(cr):
    out = {}
    for e, f in unit_sites(cr):
        pt = pos_type(f)
        out.setdefault(e, {}).setdefault(pt, [0, POS_TYPE[pt][2]])
        out[e][pt][0] += len(closed_cell_images(f))
    return out

# ---- 分摊切割球：每个原子的切割面（法向, 原点），保留块内部分 ----
def atom_cut_planes(atoms, n, cr):
    """n=1 闭胞显示时: 顶点3刀(1/8) 棱2刀(1/4) 面1刀(1/2) 体内0刀"""
    M = lattice_matrix(cr)
    crosses = [np.cross(M[1], M[2]), np.cross(M[2], M[0]), np.cross(M[0], M[1])]
    crosses = [c / np.linalg.norm(c) for c in crosses]
    out = []
    for a in atoms:
        pl = []
        s = a['xyz']
        for ax in range(3):
            v = s[ax]
            if abs(v) < 1e-6 or abs(v - n) < 1e-6:
                q = np.zeros(3)
                q[(ax + 1) % 3] = 0.5
                q[(ax + 2) % 3] = 0.5
                q[ax] = 0.0 if abs(v) < 1e-6 else float(n)
                sign = -1.0 if abs(v) < 1e-6 else 1.0   # 外法向
                pl.append((sign * crosses[ax], q @ M))
        out.append(pl)
    return out

# ============ 4. 六棱柱（教材式） ============
def _hex_normals():
    return [(math.cos(math.radians(30 + 60 * k)),
             math.sin(math.radians(30 + 60 * k))) for k in range(6)]

def _in_hex(q, center, a, eps=1e-6):
    lim = a * math.sqrt(3) / 2 + eps
    dx, dy = q[0] - center[0], q[1] - center[1]
    for nx, ny in _hex_normals():
        if dx * nx + dy * ny > lim:
            return False
    return True

def prism_cells(cr, nc=1):
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
                                  frac=None, ptype='prism_body', share=1.0,
                                  nshare=1))
    return atoms

PRISM_POS = {'prism_vertex': ('顶点', 6, 1/6), 'prism_face': ('面心', 2, 0.5),
             'prism_body': ('体内', 1, 1.0)}

def prism_apportioned(cr):
    return {cr['basis'][0][0]: 6.0}

def prism_summary(cr):
    return {cr['basis'][0][0]:
            {'prism_vertex': [12, 1/6], 'prism_face': [2, 0.5],
             'prism_body': [3, 1.0]}}

# ============ 5. 化学式 / 密度 / 利用率 / 配位 / 键 ============
_METALS = set('Li Be Na Mg Al K Ca Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Zr '
              'Nb Mo Ru Rh Pd Ag Cd In Sn Cs Ba Ce W Pt Au Tl Pb Bi U Hg Po'
              .split())

def formula(cr, counts=None):
    ov = cr.get('formula_override')
    if ov:
        return ov
    if counts is None:
        counts = apportioned_counts(cr)
    cnt = {e: v for e, v in counts.items() if v > 1e-6}
    if len(cnt) == 1:
        return next(iter(cnt))
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
    elif set(cnt) == {'C', 'Si'}:
        order = ['Si', 'C']
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
    if cr.get('mass_override') is not None:
        m = cr['mass_override']
    else:
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
    """{elem: (配位数, 邻居元素集合)}"""
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
    """显示原子之间的键（分块向量化）。bonds=[]无键; None自动; 列表按元素对"""
    spec = cr.get('bonds', None)
    md = min_distances(cr)
    if spec == []:
        return []
    if spec is None:
        allowed = {k for k, v in md.items()
                   if v <= 1.25 * (display_radius(cr, k[0]) +
                                   display_radius(cr, k[1]))}
    else:
        allowed = {tuple(sorted(p)) for p in spec}
    pos = np.asarray([a['cart'] for a in atoms], float)
    n = len(atoms)
    if n == 0:
        return []
    lims = {e: md.get(e, 1e9) * 1.15 + 0.02 for e in allowed}
    out = []
    CH = 400
    for i0 in range(0, n, CH):
        blk = pos[i0:i0 + CH]
        d = np.linalg.norm(blk[:, None, :] - pos[None, :, :], axis=2)
        for li in range(blk.shape[0]):
            i = i0 + li
            ei = atoms[i]['elem']
            row = d[li]
            for j in range(i + 1, n):
                e = (ei, atoms[j]['elem'])
                if e not in allowed:
                    e2 = (atoms[j]['elem'], ei)
                    e = e2 if e2 in allowed else None
                    if e is None:
                        continue
                if row[j] <= lims[e]:
                    out.append((i, j))
    return out

def fcc_holes(cr):
    if not cr.get('holes'):
        return [], []
    ua = unit_sites(cr)
    octa_sites = [(0.5, 0.5, 0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0),
                  (0.0, 0.0, 0.5)]
    tetra_sites = [(x, y, z) for x in (0.25, 0.75) for y in (0.25, 0.75)
                   for z in (0.25, 0.75)]
    def occ(f):
        q = np.array(f, float)
        for e, g in ua:
            d = q - np.array(g)
            d = d - np.round(d)
            if np.linalg.norm(d) < 0.05:
                return e
        return None
    oct_out = [(img, occ(f)) for f in octa_sites for img in closed_cell_images(f)]
    tet_out = [(img, occ(f)) for f in tetra_sites for img in closed_cell_images(f)]
    return oct_out, tet_out

def convex_hull_faces(points, eps=1e-7):
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
            others = [(j, np.linalg.norm(pos[j] - pos[i]))
                      for j in range(len(atoms))
                      if j != i and atoms[j]['elem'] == nb_e]
            if len(others) < 4:
                continue
            dmin = min(x[1] for x in others)
            verts = [j for j, dj in others if dj <= dmin * 1.15 + 0.05]
            if expect and len(verts) != expect:
                continue
            if 4 <= len(verts) <= 12 and \
               len(convex_hull_faces([pos[j] for j in verts])) >= 4:
                out.append(dict(center=i, verts=verts))
    return out

# ============ 6. 新增：键角 / d间距与晶面 / 投影 ============
def bond_angles(cr):
    """{elem: [(角度°, 次数)]}：以每个site为中心，最近邻两两夹角"""
    M = lattice_matrix(cr)
    ua = unit_sites(cr)
    ts = _trans_vectors(M)
    res = {}
    for e1, f1 in ua:
        d1 = np.array(f1) @ M
        nb = []
        for e2, f2 in ua:
            same = (e1 == e2 and f1 == f2)
            d2 = np.array(f2) @ M
            for t in ts:
                if same and np.allclose(t, 0):
                    continue
                v = d2 + t - d1
                if np.linalg.norm(v) > 1e-4:
                    nb.append(v)
        if len(nb) < 2:
            continue
        dmin = min(np.linalg.norm(v) for v in nb)
        verts = [v for v in nb
                 if np.linalg.norm(v) <= dmin * 1.03 + 1e-4]
        if len(verts) < 2:
            continue
        ang = {}
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                c = float(np.dot(verts[i], verts[j]) /
                          (np.linalg.norm(verts[i]) * np.linalg.norm(verts[j])))
                a = round(math.degrees(math.acos(max(-1.0, min(1.0, c)))), 1)
                ang[a] = ang.get(a, 0) + 1
        merged = {}
        for a in sorted(ang):
            for k in merged:
                if abs(k - a) < 0.5:
                    merged[k] += ang[a]
                    break
            else:
                merged[a] = ang[a]
        res[e1] = sorted(merged.items())
    return res

def d_spacing(cr, h, k, l):
    """晶面族(hkl)的d间距（任意晶系，度量张量法）"""
    M = lattice_matrix(cr)
    G = M @ M.T
    f = np.array([h, k, l], float)
    return 1.0 / math.sqrt(float(f @ np.linalg.inv(G) @ f))

def hkl_plane(cr, h, k, l):
    """(hkl)面：返回(单位法向, d间距)。法向= M⁻¹·(hkl)"""
    M = lattice_matrix(cr)
    N = np.linalg.inv(M) @ np.array([h, k, l], float)
    n = np.linalg.norm(N)
    return N / n, 1.0 / n

def view_direction(cr, uvw):
    """晶向[uvw]的笛卡尔单位向量"""
    v = np.array(uvw, float) @ lattice_matrix(cr)
    return v / np.linalg.norm(v)

def projection_2d(cr, atoms, uvw):
    """沿[uvw]压平投影：[(x, y, depth)]（depth为沿视向的有符号距离）"""
    n = view_direction(cr, uvw)
    tmp = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, tmp)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    out = []
    for a in atoms:
        p = np.asarray(a['cart'], float)
        out.append((float(p @ e1), float(p @ e2), float(p @ n)))
    return out

# ============ 7. 金属密堆积演示 ============
def build_stack(sequence, r=1.3, extent=4):
    v1 = np.array([2 * r, 0.0]); v2 = np.array([r, r * math.sqrt(3)])
    hole1 = (v1 + v2) / 3.0
    off = {'A': np.zeros(2), 'B': 2 * hole1, 'C': hole1}
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
    if k0 + 2 >= len(sequence) or sequence[k0:k0 + 3] != 'ABA':
        return None, False
    a = 2 * r
    hexv = [(a * math.cos(math.radians(t)), a * math.sin(math.radians(t)))
            for t in (0, 60, 120, 180, 240, 300)]
    return (hexv, k0 * stack_layer_height(r),
            (k0 + 2) * stack_layer_height(r)), True

# ============ 8. CIF 解析 ============
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
    al = num('_cell_angle_alpha', 90.0)
    be = num('_cell_angle_beta', 90.0)
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
                                np.allclose(op[0], o[0]) and
                                np.allclose(op[1], o[1]) for o in ops):
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

# ============ 9. 自测（~130项） ============
def _close(a, b, tol=0.02):
    return abs(a - b) <= tol

def run_tests(verbose=True):
    ok, fail = 0, []
    def T(name, cond):
        nonlocal ok
        if cond: ok += 1
        else: fail.append(name)
        if verbose: print(('  OK ' if cond else '  XX ') + name)

    print('== 库完整性 ==')
    T('晶体数>=90 (%d)' % len(DB), len(DB) >= 90)
    T('key唯一', len(DB) == len(CRYSTAL_DB))
    T('均有原型与类别', all(c.get('prototype') and
                           c['category'] in ('金属晶体', '离子晶体',
                                             '共价晶体', '分子晶体')
                           for c in CRYSTAL_DB))

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

    print('== 均摊计数（全部93种） ==')
    Z2 = {'li': ('Li', 2), 'na': ('Na', 2), 'k': ('K', 2), 'rb': ('Rb', 2),
          'cs': ('Cs', 2), 'v': ('V', 2), 'nb': ('Nb', 2), 'cr': ('Cr', 2),
          'mo': ('Mo', 2), 'fe': ('Fe', 2), 'w': ('W', 2),
          'ca': ('Ca', 4), 'sr': ('Sr', 4), 'al': ('Al', 4), 'ni': ('Ni', 4),
          'cu': ('Cu', 4), 'rh': ('Rh', 4), 'pd': ('Pd', 4), 'ag': ('Ag', 4),
          'pt': ('Pt', 4), 'au': ('Au', 4), 'pb': ('Pb', 4)}
    Z2h = {'be': ('Be', 2), 'mg': ('Mg', 2), 'ti': ('Ti', 2), 'co': ('Co', 2),
           'zn': ('Zn', 2), 'zr': ('Zr', 2), 'cd': ('Cd', 2)}
    zexp = {'po': {'Po': 1}}
    for k, (e, z) in Z2.items(): zexp[k] = {e: z}
    for k, (e, z) in Z2h.items(): zexp[k] = {e: z}
    for k, (c1, c2, z1, z2) in {
            'lif': ('Li', 'F', 4, 4), 'naf': ('Na', 'F', 4, 4),
            'nacl': ('Na', 'Cl', 4, 4), 'nabr': ('Na', 'Br', 4, 4),
            'kf': ('K', 'F', 4, 4), 'kcl': ('K', 'Cl', 4, 4),
            'kbr': ('K', 'Br', 4, 4), 'licl': ('Li', 'Cl', 4, 4),
            'mgo': ('Mg', 'O', 4, 4), 'cao': ('Ca', 'O', 4, 4),
            'sro': ('Sr', 'O', 4, 4), 'bao': ('Ba', 'O', 4, 4),
            'mgs': ('Mg', 'S', 4, 4), 'pbs': ('Pb', 'S', 4, 4),
            'agcl': ('Ag', 'Cl', 4, 4), 'nio': ('Ni', 'O', 4, 4)}.items():
        zexp[k] = {c1: z1, c2: z2}
    for k, (an, c) in {'cscl': ('Cl', 'Cs'), 'csbr': ('Br', 'Cs'),
                       'csi': ('I', 'Cs'), 'tlcl': ('Cl', 'Tl')}.items():
        zexp[k] = {an: 1, c: 1}
    for k, (c, an) in {'caf2': ('Ca', 'F'), 'srf2': ('Sr', 'F'),
                       'baf2': ('Ba', 'F'), 'ceo2': ('Ce', 'O'),
                       'uo2': ('U', 'O')}.items():
        zexp[k] = {c: 4, an: 8}
    for k, (c, an) in {'li2o': ('Li', 'O'), 'na2o': ('Na', 'O'),
                       'k2o': ('K', 'O')}.items():
        zexp[k] = {c: 8, an: 4}
    for k, (c, an) in {'znsw': ('Zn', 'S'), 'cucl': ('Cu', 'Cl'),
                       'cubr': ('Cu', 'Br'), 'bn': ('B', 'N'),
                       'sic': ('Si', 'C'), 'gaas': ('Ga', 'As'),
                       'insb': ('In', 'Sb'), 'cdte': ('Cd', 'Te')}.items():
        zexp[k] = {c: 4, an: 4}
    for k, (c, an) in {'znsz': ('Zn', 'S'), 'zno': ('Zn', 'O'),
                       'cds': ('Cd', 'S'), 'aln': ('Al', 'N'),
                       'beo': ('Be', 'O'), 'gan': ('Ga', 'N')}.items():
        zexp[k] = {c: 2, an: 2}
    for k, (c, an) in {'rutile': ('Ti', 'O'), 'sno2': ('Sn', 'O'),
                       'mno2': ('Mn', 'O')}.items():
        zexp[k] = {c: 2, an: 4}
    zexp['mgf2'] = {'Mg': 2, 'F': 4}
    for k, (A, B) in {'perov': ('Ca', 'Ti'), 'batio3': ('Ba', 'Ti'),
                      'srtio3': ('Sr', 'Ti'), 'kmgf3': ('K', 'Mg')}.items():
        zexp[k] = {A: 1, B: 1, 'O' if k != 'kmgf3' else 'F': 3}
    for k, (e, r) in {'diamond': ('C', 8), 'si': ('Si', 8), 'ge': ('Ge', 8),
                      'sn': ('Sn', 8)}.items():
        zexp[k] = {e: r}
    zexp['graphite'] = {'C': 4}
    zexp['hbn'] = {'B': 2, 'N': 2}
    zexp['mos2'] = {'Mo': 2, 'S': 4}
    zexp['cdi2'] = {'Cd': 1, 'I': 2}
    zexp['co2'] = {'C': 4, 'O': 8}
    zexp['ar'] = {'Ar': 4}; zexp['kr'] = {'Kr': 4}; zexp['xe'] = {'Xe': 4}
    zexp['ice'] = {'O': 4}
    bad = [k for k, e in zexp.items()
           if k in DB and apportioned_counts(DB[k]) != e]
    T('Z全部正确(%d种)' % len(zexp), not bad and len(zexp) == len(DB))
    if bad: print('   错误:', bad)

    print('== 教材式位置分解 ==')
    s = cell_atom_summary(DB['nacl'])
    T('NaCl: Cl 顶8×1/8+面6×1/2', s['Cl'].get('corner') == [8, .125] and
      s['Cl'].get('face') == [6, .5])
    T('NaCl: Na 体1+棱12×1/4', s['Na'].get('body') == [1, 1.0] and
      s['Na'].get('edge') == [12, .25])

    print('== 化学式 ==')
    fexp = {'nacl': 'NaCl', 'kcl': 'KCl', 'caf2': 'CaF₂', 'baf2': 'BaF₂',
            'znsw': 'ZnS', 'co2': 'CO₂', 'perov': 'CaTiO₃',
            'batio3': 'BaTiO₃', 'srtio3': 'SrTiO₃', 'kmgf3': 'KMgF₃',
            'na2o': 'Na₂O', 'li2o': 'Li₂O', 'k2o': 'K₂O',
            'rutile': 'TiO₂', 'sno2': 'SnO₂', 'mno2': 'MnO₂',
            'diamond': 'C', 'cu': 'Cu', 'mos2': 'MoS₂', 'cdi2': 'CdI₂',
            'gaas': 'GaAs', 'sic': 'SiC', 'hbn': 'BN', 'cscl': 'CsCl',
            'pbs': 'PbS', 'cdte': 'CdTe', 'insb': 'InSb', 'ice': 'H₂O',
            'tlcl': 'TlCl', 'csi': 'CsI', 'csbr': 'CsBr', 'agcl': 'AgCl',
            'nio': 'NiO', 'cao': 'CaO', 'sro': 'SrO', 'bao': 'BaO',
            'mgs': 'MgS', 'lif': 'LiF', 'naf': 'NaF', 'nabr': 'NaBr',
            'kf': 'KF', 'kbr': 'KBr', 'licl': 'LiCl', 'ceo2': 'CeO₂',
            'uo2': 'UO₂', 'gan': 'GaN', 'aln': 'AlN', 'beo': 'BeO',
            'cds': 'CdS', 'zno': 'ZnO', 'sn': 'Sn', 'graphite': 'C',
            'kr': 'Kr', 'xe': 'Xe', 'ar': 'Ar', 'po': 'Po'}
    bad = [k for k, f0 in fexp.items() if formula(DB[k]) != f0]
    T('化学式(%d种)' % len(fexp), not bad)
    if bad: print('   错误:', [(k, formula(DB[k]), f0)
                              for k, f0 in fexp.items() if k in bad])

    print('== 密度 g/cm3 ==')
    dexp = {'nacl': 2.16, 'cu': 8.91, 'fe': 7.87, 'mg': 1.74,
            'diamond': 3.51, 'cscl': 3.99, 'caf2': 3.18, 'kcl': 1.99,
            'pbs': 7.59, 'baf2': 4.89, 'gaas': 5.33, 'zno': 5.67,
            'batio3': 6.05, 'mos2': 5.00, 'cdi2': 5.73, 'ice': 0.91,
            'hbn': 2.28, 'sr': 2.59}
    bad = [k for k, d0 in dexp.items() if not _close(density(DB[k]), d0, 0.08)]
    T('密度(%d种)' % len(dexp), not bad)
    if bad: print('   错误:', [(k, round(density(DB[k]), 3), d0)
                              for k, d0 in dexp.items() if k in bad])

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
    cn = coordination(DB['ceo2'])
    T('CeO2 Ce=8 O=4', cn['Ce'] == (8, {'O'}) and cn['O'] == (4, {'Ce'}))
    cn = coordination(DB['znsw'])
    T('ZnS 4:4', cn['Zn'] == (4, {'S'}) and cn['S'] == (4, {'Zn'}))
    T('Cu 12', coordination(DB['cu'])['Cu'] == (12, {'Cu'}))
    T('Fe 8', coordination(DB['fe'])['Fe'] == (8, {'Fe'}))
    T('Po 6', coordination(DB['po'])['Po'] == (6, {'Po'}))
    T('C(金刚石) 4', coordination(DB['diamond'])['C'] == (4, {'C'}))
    T('CO2 C=2', coordination(DB['co2'])['C'] == (2, {'O'}))
    cn = coordination(DB['rutile'])
    T('金红石 Ti=6 O=3', cn['Ti'] == (6, {'O'}) and cn['O'] == (3, {'Ti'}))
    cn = coordination(DB['sno2'])
    T('SnO2 Sn=6 O=3', cn['Sn'] == (6, {'O'}) and cn['O'] == (3, {'Sn'}))
    cn = coordination(DB['na2o'])
    T('Na2O Na=4 O=8', cn['Na'] == (4, {'O'}) and cn['O'] == (8, {'Na'}))
    cn = coordination(DB['batio3'])
    T('BaTiO3 Ti=6 Ba=12 O=2', cn['Ti'] == (6, {'O'}) and
      cn['Ba'] == (12, {'O'}) and cn['O'] == (2, {'Ti'}))
    cn = coordination(DB['kmgf3'])
    T('KMgF3 Mg=6 K=12', cn['Mg'] == (6, {'F'}) and cn['K'] == (12, {'F'}))
    cn = coordination(DB['zno'])
    T('ZnO 4:4', cn['Zn'] == (4, {'O'}) and cn['O'] == (4, {'Zn'}))
    cn = coordination(DB['cdi2'])
    T('CdI2 Cd=6 I=3', cn['Cd'] == (6, {'I'}) and cn['I'] == (3, {'Cd'}))
    cn = coordination(DB['mos2'])
    T('MoS2 Mo=6 S=3', cn['Mo'] == (6, {'S'}) and cn['S'] == (3, {'Mo'}))
    cn = coordination(DB['hbn'])
    T('hBN 3:3', cn['B'] == (3, {'N'}) and cn['N'] == (3, {'B'}))
    cn = coordination(DB['ice'])
    T('冰 O=4(O)', cn['O'] == (4, {'O'}))
    T('灰锡 4', coordination(DB['sn'])['Sn'] == (4, {'Sn'}))
    cn = coordination(DB['gaas'])
    T('GaAs 4:4', cn['Ga'] == (4, {'As'}) and cn['As'] == (4, {'Ga'}))

    print('== 超晶胞显示原子数 ==')
    T('NaCl 1^3=27', len(supercell_atoms(DB['nacl'], 1)) == 27)
    T('NaCl 2^3=125', len(supercell_atoms(DB['nacl'], 2)) == 125)
    T('Cu 1^3=14', len(supercell_atoms(DB['cu'], 1)) == 14)
    T('CsCl 1^3=9', len(supercell_atoms(DB['cscl'], 1)) == 9)
    T('Mg 1^3=9', len(supercell_atoms(DB['mg'], 1)) == 9)
    T('ZnS 1^3=18', len(supercell_atoms(DB['znsw'], 1)) == 18)
    T('金刚石 1^3=18', len(supercell_atoms(DB['diamond'], 1)) == 18)
    T('CaF2 1^3=22', len(supercell_atoms(DB['caf2'], 1)) == 22)
    T('BaTiO3 1^3=35', len(supercell_atoms(DB['batio3'], 1)) == 15)
    T('开胞 NaCl=8', len(supercell_atoms(DB['nacl'], 1, closed=False)) == 8)
    T('开胞 Cu=4', len(supercell_atoms(DB['cu'], 1, closed=False)) == 4)
    T('开胞 Mg=2', len(supercell_atoms(DB['mg'], 1, closed=False)) == 2)
    pos = [tuple(np.round(x['cart'], 4)) for x in supercell_atoms(DB['nacl'], 2)]
    T('NaCl 2^3 无重复', len(set(pos)) == len(pos))

    print('== 分摊切割球体积 ==')
    for k, z0 in [('nacl', 8.0), ('cu', 4.0), ('mg', 2.0), ('cscl', 2.0),
                  ('diamond', 8.0), ('caf2', 12.0)]:
        at = supercell_atoms(DB[k], 1)
        pls = atom_cut_planes(at, 1, DB[k])
        vol = sum(0.5 ** len(p) for p in pls)
        T(f'{k} 切割体积={z0}', _close(vol, z0, 0.05))

    print('== 化学键 ==')
    T('金刚石 16键', len(bonds_for(DB['diamond'],
          supercell_atoms(DB['diamond'], 1))) == 16)
    T('NaCl 54键', len(bonds_for(DB['nacl'],
          supercell_atoms(DB['nacl'], 1))) == 54)
    T('CO2 8键', len(bonds_for(DB['co2'], supercell_atoms(DB['co2'], 1))) == 8)
    T('石墨 9键', len(bonds_for(DB['graphite'],
          supercell_atoms(DB['graphite'], 1))) == 9)
    T('Cu 0键', len(bonds_for(DB['cu'], supercell_atoms(DB['cu'], 1))) == 0)
    T('CaF2 32键', len(bonds_for(DB['caf2'],
          supercell_atoms(DB['caf2'], 1))) == 32)

    print('== 键角 ==')
    ba = bond_angles(DB['diamond'])
    T('金刚石 109.5×6', ba['C'] == [(109.5, 6)])
    ba = bond_angles(DB['graphite'])
    T('石墨 120×3', ba['C'] == [(120.0, 3)])
    ba = bond_angles(DB['co2'])
    T('CO2 180×1', ba['C'] == [(180.0, 1)])
    ba = bond_angles(DB['nacl'])
    T('NaCl Na: 90×12,180×3', ba['Na'] == [(90.0, 12), (180.0, 3)])
    ba = bond_angles(DB['cscl'])
    T('CsCl 70.5/109.5/180', ba['Cs'] == [(70.5, 12), (109.5, 12), (180.0, 4)])
    ba = bond_angles(DB['caf2'])
    T('CaF2 F: 109.5×6', ba['F'] == [(109.5, 6)])
    ba = bond_angles(DB['zno'])
    T('ZnO 109~110.5两组', len(ba['Zn']) == 2 and
      all(107.5 <= a <= 111.5 for a, c in ba['Zn']))

    print('== d间距 / 晶面 ==')
    T('d(100)=a', _close(d_spacing(DB['nacl'], 1, 0, 0), 5.640, 1e-6))
    T('d(110)=a/√2', _close(d_spacing(DB['nacl'], 1, 1, 0),
                            5.640 / math.sqrt(2), 1e-6))
    T('d(111)=a/√3', _close(d_spacing(DB['nacl'], 1, 1, 1),
                            5.640 / math.sqrt(3), 1e-6))
    T('六方d(100)=a√3/2', _close(d_spacing(DB['mg'], 1, 0, 0),
                                3.21 * math.sqrt(3) / 2, 1e-3))
    T('六方d(001)=c', _close(d_spacing(DB['mg'], 0, 0, 1), 5.21, 1e-6))
    n, d = hkl_plane(DB['nacl'], 1, 1, 0)
    T('hkl_plane 法向', _close(abs(n[0]), math.sqrt(.5), 1e-6) and
      _close(d, 5.640 / math.sqrt(2), 1e-6))

    print('== 投影 ==')
    v = view_direction(DB['nacl'], (1, 1, 0))
    T('NaCl[110]方向', _close(v[0], math.sqrt(.5), 1e-6) and
      _close(v[2], 0, 1e-9))
    v = view_direction(DB['mg'], (0, 0, 1))
    T('Mg[0001]→+z', _close(v[2], 1.0, 1e-9))
    at = supercell_atoms(DB['nacl'], 1)
    pr = projection_2d(DB['nacl'], at, (0, 0, 1))
    T('NaCl[001]投影≥3层', len(set(round(p[2], 3) for p in pr)) >= 3)
    pr = projection_2d(DB['nacl'], at, (1, 1, 1))
    T('NaCl[111]投影≥3层', len(set(round(p[2], 3) for p in pr)) >= 3)

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
        T('棱长=2√2r', _close(e, 2 * 1.3 * math.sqrt(2), .01))
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
    T('NaCl 八面体13全Na', len(o) == 13 and all(x[1] == 'Na' for x in o))
    T('NaCl 四面体8全空', len(t) == 8 and all(x[1] is None for x in t))
    o, t = fcc_holes(DB['caf2'])
    T('CaF2 八面体13空', len(o) == 13 and all(x[1] is None for x in o))
    T('CaF2 四面体8全F', len(t) == 8 and all(x[1] == 'F' for x in t))
    o, t = fcc_holes(DB['ceo2'])
    T('CeO2 四面体8全O', len(t) == 8 and all(x[1] == 'O' for x in t))
    o, t = fcc_holes(DB['znsw'])
    T('ZnS 四面体8个4被Zn占', len(t) == 8 and
      sum(1 for x in t if x[1] == 'Zn') == 4)
    o, t = fcc_holes(DB['diamond'])
    T('金刚石 4被C占', len(t) == 8 and sum(1 for x in t if x[1] == 'C') == 4)

    print('== 凸包 / 多面体 ==')
    tet = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    T('四面体4面', len(convex_hull_faces(tet)) == 4)
    octa = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    T('八面体8面', len(convex_hull_faces(octa)) == 8)
    cube = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    T('立方24三角面', len(convex_hull_faces(cube)) == 24)
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
    T('Cu-Cu=a/√2', _close(md[('Cu', 'Cu')], 3.615 / math.sqrt(2), .01))
    md = min_distances(DB['nacl'])
    T('Na-Cl=a/2', _close(md[('Cl', 'Na')], 5.640 / 2, .01))
    md = min_distances(DB['fe'])
    T('Fe-Fe=√3a/2', _close(md[('Fe', 'Fe')], 2.866 * math.sqrt(3) / 2, .01))
    md = min_distances(DB['mos2'])
    T('Mo-S=2.41', _close(md[('Mo', 'S')], 2.41, .03))
    md = min_distances(DB['ice'])
    T('冰 O-O≈2.76', _close(md[('O', 'O')], 2.76, .03))

    print(f'\n通过 {ok} 项，失败 {len(fail)} 项')
    if fail:
        print('失败:', fail)
    return len(fail) == 0

if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        sys.exit(0 if run_tests() else 1)
    print('Crystal X-Ray v0.2 引擎 | 晶体库 %d 种 | python crystal_core.py --test 自测'
          % len(DB))
