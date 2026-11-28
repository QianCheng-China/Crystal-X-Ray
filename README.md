# Crystal X-Ray

晶胞演示软件。面向化学结构教学，可视化 93 种典型晶体，覆盖均摊法、密堆积、空隙、投影与晶面计算等核心考点。

开发者：HighspeedG2304

## 功能

- **晶体库**：93 种晶体，按晶系类别与结构原型组织，含金属、离子、共价、分子晶体
- **结构显示**：球棍 / 空间填充 / 骨棒 / 多面体四种风格；n×n×n 超晶胞；平行六面体与六棱柱两种形式
- **均摊法**：切割分摊显示（顶点 1/8、棱 1/4、面 1/2、体内 1），点击原子查看位置分解与分摊过程
- **密堆积**：ABAB / ABCABC 及自定义序列密置层堆积，自动标出 FCC 立方与 HCP 六棱柱晶胞框，点击原子显示配位数
- **空隙**：八面体与四面体空隙着色显示（NaCl、CaF2、ZnS、金刚石等）
- **投影**：沿任意晶向 [uvw] 正交观察或压平投影（按深度着色并标注 z 坐标），内置 NaCl [111]、金刚石 [110] 等预设
- **工具**：键长键角表、晶面 (hkl) 绘制、d 间距计算（任意晶系）、立方晶系密度与晶胞参数互算
- **测量**：点击原子测键长，三点测键角
- **自定义**：分数坐标建胞，导入 CIF（解析空间群对称操作），可保存至个人晶体库

## 运行环境

- Windows x64（其他平台理论可用，未验证）
- Python 3.10 及以上
- 依赖：numpy、PySide6、pyvista、pyvistaqt

## 从源码运行

```bat
git clone https://github.com/HighspeedG2304/Crystal-X-Ray.git
cd Crystal-X-Ray
pip install numpy PySide6 pyvista pyvistaqt
python xibao_app.py
```

虚拟机或无 GPU 环境启用软件渲染：

```bat
python xibao_app.py --sw
```

## 引擎自测

```bat
python crystal_core.py --test
```

约 130 项断言，覆盖均摊计数、化学式、密度、空间利用率、配位数、键角、d 间距、CIF 解析等。

## 打包 exe

```bat
pip install pyinstaller
python -m PyInstaller --clean --noconfirm crystal-xray.spec
```

产物位于 `dist\Crystal X-Ray\`，整个目录可直接分发，目标机器无需安装 Python 与依赖。

## 项目结构

```
crystal_core.py     计算引擎：晶体库、晶格、均摊、配位、键角、d 间距、密堆积、CIF 解析（无 GUI 依赖）
xibao_app.py        PySide6 + PyVista 图形界面
crystal-xray.spec   PyInstaller 打包配置
```

## 许可

MIT License
