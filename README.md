# MPS 加速的康威生命游戏 (Game of Life)

利用 **PyTorch + Apple Metal (MPS)** GPU 加速的康威生命游戏模拟器。

核心思路：每一步更新就是"统计 8 邻居存活数 + 规则判断"。统计邻居数等价于用一个
3×3 全 1（中心为 0）的卷积核对网格做卷积，因此直接交给 `torch.conv2d` 在 GPU(MPS)
上整张网格并行计算。整个盒子采用**周期边界 (toroidal)**：上下、左右相连，图案越过
边缘会从对侧绕回；图案放置也按周期边界取模。

## 安装

需要 Python 3.9+，以及：

```bash
pip install torch matplotlib
```

在 Apple Silicon (M 系列) 上 PyTorch 会自动启用 MPS 后端；无 GPU 时自动回退到 CPU。

## 快速开始

```bash
# 实时可视化（默认 256×256，随机初始化）
python life_mps.py

# 指定网格大小与代数
python life_mps.py --size 512 --gens 1000

# 无限运行（关闭窗口或 Ctrl-C 停止）
python life_mps.py --gens -1
```

## 命令行参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--size N` | 网格边长 | `256` |
| `--gens N` | 运行代数；`-1` 表示无限运行 | `500` |
| `--pattern NAME` | 单个初始图案（居中放置） | `random` |
| `--place NAME[@ROW,COL]` | 放置一个图案，可重复以放置多个；省略坐标则随机放置 | — |
| `--list-patterns` | 列出所有可用图案并退出 | — |
| `--seed N` | 随机种子 | `0` |
| `--interval MS` | 可视化帧间隔（毫秒） | `30` |
| `--no-wrap` | 使用固定边界而非周期边界 | 周期边界 |
| `--benchmark` | 跑性能基准（MPS vs CPU），不显示界面 | — |
| `--headless` | 只计算不绘图（用于无显示环境） | — |
| `--save_t X` | 把第 X 代的构象保存为 `.rle`（保存后从该代继续运行） | — |
| `--save_file PATH` | `--save_t` 的输出文件名 | `snapshot_t{X}.rle` |

## 交互

可视化窗口中按 **空格键** 暂停/继续动画（暂停时计时器真正停止，不消耗代数）。

## 可用图案

按类型分组（`python life_mps.py --list-patterns` 可随时查看）：

- **静物 (still lifes)**：`block` `beehive` `loaf` `boat` `tub`
- **振荡器 (oscillators)**：`blinker` `toad` `beacon` `pulsar`、`pentadecathlon`（周期 15 的十五周期发射台）、`queenbee`（Queen Bee Shuttle，周期 30）
- **反射器 (reflectors)**：`snark`（Mike Playle 2013，最小的稳定 90° 滑翔机反射器；内置图案自带一个示范滑翔机演示反射）
- **飞船 (spaceships)**：`glider` `lwss` `mwss` `hwss`
- **枪 (guns)**：`glider-gun`（Gosper glider gun）
- **玛士撒拉 (methuselahs)**：`r-pentomino` `acorn` `diehard`
- **其他**：`random`（按密度随机填充）

## 多图案放置

`--place` 可重复使用，把多个图案放到指定坐标（`@行,列`，左上角为原点）。
坐标按整盒周期边界取模，越过边缘会绕回对侧。

```bash
# 在不同位置放置三个图案（acorn 随机位置）
python life_mps.py --place glider@5,5 --place pulsar@40,40 --place acorn

# 无限播放滑翔机枪
python life_mps.py --place glider-gun@5,5 --gens -1

# 观察 Snark 反射器把滑翔机转 90°
python life_mps.py --place snark@30,30 --gens -1

# 周期 30 的 Queen Bee Shuttle
python life_mps.py --pattern queenbee
```

## 朝向（四种方向）

飞船、滑翔机、反射器等带方向的图案可以旋转到四个朝向（0°/90°/180°/270°），
再配合水平镜像可得全部 8 种朝向。

单图案用 `--rotate` / `--flip`：

```bash
# 滑翔机朝四个对角方向
python life_mps.py --pattern glider --rotate 0
python life_mps.py --pattern glider --rotate 90
python life_mps.py --pattern glider --rotate 180 --flip
```

多图案放置时，在规格末尾加 `:朝向`（`0/90/180/270` 或 `n/e/s/w`，末尾加 `m` 镜像）：

```bash
# 四艘轻量级飞船朝不同方向出发
python life_mps.py --size 120 \
  --place lwss@10,10:e --place lwss@10,100:w \
  --place lwss@100,10:s --place lwss@100,100:n --gens -1
```

## 从文件加载图案

除了内置库，`--pattern` 和 `--place` 都能直接接受 **`.rle`** 或 **`.cells`** 文件路径
（conwaylife.com 通用格式），无需改代码即可加载任意图案：

```bash
# 下载任意图案后直接加载
python life_mps.py --pattern ./patterns/spacefiller.rle

# 与库内图案混合放置
python life_mps.py --place snark@20,20 --place ./my_gun.cells@60,60 --gens -1
```

## 性能基准

对比同一初始网格在 CPU 与 MPS 上的运行速度：

```bash
python life_mps.py --benchmark --size 1024 --gens 200
```

输出示例：

```
基准测试: 1024x1024 网格, 200 代
----------------------------------------
CPU :    X.XXX s  (   XX.X 代/秒)
MPS :    X.XXX s  (  XXX.X 代/秒)

加速比: N.NNx
```

## 边框

可视化窗口会在网格四周绘制一个边框，标示周期边界盒子的边缘。

## 规则

标准生命游戏规则 **B3/S23**：

- 死细胞恰好有 3 个活邻居 → 复活（Birth）
- 活细胞有 2 或 3 个活邻居 → 存活（Survive），否则死亡
