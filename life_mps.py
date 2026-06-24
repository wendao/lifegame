#!/usr/bin/env python3
"""
利用 MPS (Apple Metal) 加速的康威生命游戏模拟器
=================================================

核心思路：生命游戏的每一步更新就是一次"统计 8 邻居存活数 + 规则判断"。
统计邻居数等价于用一个 3x3 的全 1 卷积核（中心为 0）对网格做卷积，
因此可以直接交给 PyTorch 的 conv2d 在 GPU(MPS) 上批量并行计算。

用法示例：
    # 实时可视化（默认 256x256，随机初始化）
    python life_mps.py

    # 指定网格大小与代数
    python life_mps.py --size 512 --gens 1000

    # 无界面跑性能基准（MPS vs CPU）
    python life_mps.py --benchmark --size 1024 --gens 200

    # 用经典图案初始化
    python life_mps.py --pattern glider-gun
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 设备选择
# --------------------------------------------------------------------------- #
def pick_device(prefer: str = "mps") -> torch.device:
    """优先 MPS，其次 CUDA，最后 CPU。"""
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# 统计邻居数的 3x3 卷积核（中心为 0，自己不算邻居）
_NEIGHBOR_KERNEL = torch.tensor(
    [[1.0, 1.0, 1.0],
     [1.0, 0.0, 1.0],
     [1.0, 1.0, 1.0]]
).view(1, 1, 3, 3)


class LifeMPS:
    """在指定 device 上运行的生命游戏。

    网格 shape = (1, 1, H, W)，dtype = float32（MPS 对 float32 支持最佳）。
    边界采用环形（toroidal）拓扑：上下、左右相连。
    """

    def __init__(self, grid: torch.Tensor, device: torch.device, wrap: bool = True):
        self.device = device
        self.wrap = wrap
        self.grid = grid.to(device=device, dtype=torch.float32).view(
            1, 1, *grid.shape[-2:]
        )
        self.kernel = _NEIGHBOR_KERNEL.to(device=device, dtype=torch.float32)

    @property
    def board(self) -> torch.Tensor:
        """返回 (H, W) 的 0/1 网格（仍在 device 上）。"""
        return self.grid.view(*self.grid.shape[-2:])

    def step(self) -> None:
        """推进一代。全部计算发生在 self.device 上。"""
        if self.wrap:
            # 环形边界：先做循环 padding，再用 valid 卷积
            padded = F.pad(self.grid, (1, 1, 1, 1), mode="circular")
            neighbors = F.conv2d(padded, self.kernel)
        else:
            # 固定边界：零 padding
            neighbors = F.conv2d(self.grid, self.kernel, padding=1)

        alive = self.grid == 1
        # B3/S23 规则：
        #   死细胞恰好 3 个邻居 -> 复活
        #   活细胞 2 或 3 个邻居 -> 存活，否则死亡
        born = (~alive) & (neighbors == 3)
        survive = alive & ((neighbors == 2) | (neighbors == 3))
        self.grid = (born | survive).to(torch.float32)

    def run(self, gens: int) -> None:
        """运行 gens 代；gens < 0 表示无限运行（Ctrl-C 停止）。"""
        if gens < 0:
            while True:
                self.step()
        else:
            for _ in range(gens):
                self.step()

    def to_numpy(self):
        return self.board.detach().to("cpu").numpy()


# --------------------------------------------------------------------------- #
# 图案库
# --------------------------------------------------------------------------- #
def _ascii(block: str):
    """把 ASCII 图案（'O'/'.'）解析成相对坐标列表 [(r, c), ...]。"""
    rows = block.strip("\n").split("\n")
    return [
        (i, j)
        for i, row in enumerate(rows)
        for j, ch in enumerate(row)
        if ch in "O*#"
    ]


def _rle(text: str):
    """解析 RLE 格式（conwaylife.com 通用格式）为坐标列表 [(r, c), ...]。"""
    lines = [
        ln for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    body = []
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("x") and "=" in s:  # header: x = .., y = ..
            continue
        body.append(s)
    data = "".join(body)
    cells = []
    r = c = 0
    num = ""
    for ch in data:
        if ch.isdigit():
            num += ch
        elif ch in "bo.O":
            n = int(num) if num else 1
            if ch in "oO":
                cells.extend((r, c + k) for k in range(n))
            c += n
            num = ""
        elif ch == "$":  # 换行（可带计数表示多行）
            n = int(num) if num else 1
            r += n
            c = 0
            num = ""
        elif ch == "!":
            break
    return cells


def load_pattern_file(path: str):
    """从 .rle 或 .cells 文件加载图案, 自动识别格式。"""
    with open(path, "r") as f:
        text = f.read()
    if path.lower().endswith(".rle") or ("$" in text and "=" in text):
        return _rle(text)
    # .cells / plaintext: 以 '!' 开头为注释
    block = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("!")
    )
    return _ascii(block)


# 每个图案是一组相对坐标（以左上角为原点）。
PATTERNS = {
    # ---- 静物 (still lifes) ----
    "block":        _ascii("OO\nOO"),
    "beehive":      _ascii(".OO.\nO..O\n.OO."),
    "loaf":         _ascii(".OO.\nO..O\n.O.O\n..O."),
    "boat":         _ascii("OO.\nO.O\n.O."),
    "tub":          _ascii(".O.\nO.O\n.O."),

    # ---- 振荡器 (oscillators) ----
    "blinker":      _ascii("OOO"),
    "toad":         _ascii(".OOO\nOOO."),
    "beacon":       _ascii("OO..\nO...\n...O\n..OO"),
    "pulsar": _ascii(
        "..OOO...OOO..\n"
        ".............\n"
        "O....O.O....O\n"
        "O....O.O....O\n"
        "O....O.O....O\n"
        "..OOO...OOO..\n"
        ".............\n"
        "..OOO...OOO..\n"
        "O....O.O....O\n"
        "O....O.O....O\n"
        "O....O.O....O\n"
        ".............\n"
        "..OOO...OOO.."
    ),
    "pentadecathlon": _ascii(
        "..O....O..\n"
        "OO.OOOO.OO\n"
        "..O....O.."
    ),
    # Queen bee shuttle: 周期 30 (两端的 block 作为缓冲)
    "queenbee": _ascii(
        ".........O............\n"
        ".......O.O............\n"
        "......O.O.............\n"
        "OO...O..O...........OO\n"
        "OO....O.O...........OO\n"
        ".......O.O............\n"
        ".........O............"
    ),

    # ---- 反射器 (reflectors) ----
    # Snark: 最小的稳定 90° 滑翔机反射器 (Mike Playle, 2013)
    "snark": _ascii(
        "......OO...OO....\n"
        "......OO..O.OOO..\n"
        "..........O....O.\n"
        "......OOOO.OO..O.\n"
        "......O..O.O.O.OO\n"
        ".........O.O.O.O.\n"
        "..........OO.O.O.\n"
        "..............O..\n"
        ".................\n"
        "OO...............\n"
        ".O.......OO......\n"
        ".O.O.....OO......\n"
        "..OO.............\n"
        ".................\n"
        ".................\n"
        ".................\n"
        ".................\n"
        ".................\n"
        ".................\n"
        "............OO...\n"
        "...OO.......O....\n"
        "..O.O........OOO.\n"
        "....O..........O."
    ),

    # ---- 飞船 (spaceships) ----
    "glider":       _ascii(".O.\n..O\nOOO"),
    "lwss":         _ascii(".O..O\nO....\nO...O\nOOOO."),
    "mwss": _ascii(
        "...O..\n"
        ".O...O\n"
        "O.....\n"
        "O....O\n"
        "OOOOO."
    ),
    "hwss":         _ascii(
        "...OO..\n"
        ".O....O\n"
        "O......\n"
        "O.....O\n"
        "OOOOOO."
    ),

    # ---- 枪 (guns) ----
    "glider-gun": _ascii(
        "........................O...........\n"
        "......................O.O...........\n"
        "............OO......OO............OO\n"
        "...........O...O....OO............OO\n"
        "OO........O.....O...OO..............\n"
        "OO........O...O.OO....O.O...........\n"
        "..........O.....O.......O...........\n"
        "...........O...O....................\n"
        "............OO......................"
    ),

    # ---- 玛士撒拉 (methuselahs，长寿小图案) ----
    "r-pentomino":  _ascii(".OO\nOO.\n.O."),
    "acorn":        _ascii(".O.....\n...O...\nOO..OOO"),
    "diehard":      _ascii("......O.\nOO......\n.O...OOO"),
}

PATTERN_NAMES = sorted(PATTERNS) + ["random"]


def transform_cells(cells, rot: int = 0, flip: bool = False):
    """旋转/镜像图案坐标。

    rot: 顺时针旋转的 90° 次数 (0/1/2/3 -> 0°/90°/180°/270°)。
    flip: 是否先做水平镜像 (得到另外 4 个朝向, 共 8 种)。
    旋转后归一化到非负坐标 (左上角为原点)。
    """
    pts = [(r, -c) for r, c in cells] if flip else list(cells)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]  # 顺时针 90°
    minr = min(r for r, _ in pts)
    minc = min(c for _, c in pts)
    return [(r - minr, c - minc) for r, c in pts]


# 朝向记号 -> (旋转次数, 是否镜像)
_ORIENT = {
    "": (0, False),
    "0": (0, False), "90": (1, False), "180": (2, False), "270": (3, False),
    "r0": (0, False), "r90": (1, False), "r180": (2, False), "r270": (3, False),
    "n": (0, False), "e": (1, False), "s": (2, False), "w": (3, False),
    "north": (0, False), "east": (1, False),
    "south": (2, False), "west": (3, False),
}


def parse_orient(tok: str):
    """解析朝向记号: 0/90/180/270 或 n/e/s/w; 末尾加 m/f 表示镜像。"""
    t = tok.strip().lower()
    flip = False
    if t and t[-1] in "mf":
        flip = True
        t = t[:-1]
    if t not in _ORIENT:
        raise ValueError(f"未知朝向: {tok!r} (可用 0/90/180/270 或 n/e/s/w[+m])")
    rot, base_flip = _ORIENT[t]
    return rot, (flip or base_flip)


def board_to_rle(board, rule: str = "B3/S23") -> str:
    """把 (H,W) 的 0/1 网格编码为 RLE 字符串 (conwaylife.com 通用格式)。"""
    import numpy as np

    a = np.asarray(board).astype(int)
    ys, xs = np.where(a == 1)
    if len(ys) == 0:
        return f"x = 0, y = 0, rule = {rule}\n!\n"
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    H, W = a.shape
    tokens = []
    for r in range(H):
        row, line, c = a[r], "", 0
        while c < W:
            v = row[c]
            k = 1
            while c + k < W and row[c + k] == v:
                k += 1
            line += (str(k) if k > 1 else "") + ("o" if v else "b")
            c += k
        tokens.append(line)
    body = "$".join(tokens) + "!"
    wrapped = "\n".join(body[i:i + 70] for i in range(0, len(body), 70))
    return f"x = {W}, y = {H}, rule = {rule}\n{wrapped}\n"


def resolve_cells(name: str):
    """把图案名或文件路径解析为坐标列表。

    - 内置库名 (如 'glider', 'snark') -> 取自 PATTERNS
    - .rle / .cells 文件路径 -> 即时从文件加载
    """
    if name in PATTERNS:
        return PATTERNS[name]
    if name.lower().endswith((".rle", ".cells")) or os.path.exists(name):
        return load_pattern_file(name)
    raise ValueError(
        f"未知图案: {name}。可用: {', '.join(sorted(PATTERNS))} "
        f"(或给出 .rle/.cells 文件路径)"
    )


def stamp(
    grid: torch.Tensor, name: str, top: int, left: int,
    rot: int = 0, flip: bool = False,
) -> None:
    """在 grid 的 (top, left) 处盖上一个图案（就地修改）。

    name 可为内置库名或 .rle/.cells 文件路径；rot/flip 控制朝向。坐标按整盒
    周期边界 (toroidal) 取模, 越过边缘的部分会从对侧绕回, 与模拟步进所用的环形
    边界保持一致。
    """
    H, W = grid.shape
    for r, c in transform_cells(resolve_cells(name), rot, flip):
        grid[(top + r) % H, (left + c) % W] = 1.0


def make_grid(
    size: int, pattern: str = "random", seed: int = 0,
    rot: int = 0, flip: bool = False,
) -> torch.Tensor:
    """单图案初始化（向后兼容）。random 时按密度随机填充。"""
    g = torch.zeros(size, size, dtype=torch.float32)
    if pattern == "random":
        gen = torch.Generator().manual_seed(seed)
        return (torch.rand(size, size, generator=gen) < 0.25).to(torch.float32)
    # 居中放置单个图案（按朝向变换后）
    cells = transform_cells(resolve_cells(pattern), rot, flip)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    g_top, g_left = (size - h) // 2, (size - w) // 2
    for r, c in cells:
        g[(g_top + r) % size, (g_left + c) % size] = 1.0
    return g


def parse_placements(specs, size: int, seed: int = 0) -> torch.Tensor:
    """根据 --place 规格列表构建网格。

    每条规格形如  name[@row,col][:orient]
      name   : 库名或 .rle/.cells 路径
      @row,col: 放置位置 (省略则随机)
      :orient: 朝向 0/90/180/270 或 n/e/s/w, 末尾加 m 表示镜像
    """
    g = torch.zeros(size, size, dtype=torch.float32)
    rng = torch.Generator().manual_seed(seed)
    for spec in specs:
        spec = spec.strip()
        rot, flip = 0, False
        # 末尾可选 :orient (谨慎处理, 避免吃掉文件路径里的冒号)
        if ":" in spec:
            base, otok = spec.rsplit(":", 1)
            try:
                rot, flip = parse_orient(otok)
                spec = base
            except ValueError:
                pass
        if "@" in spec:
            name, pos = spec.split("@", 1)
            name = name.strip()
            try:
                row, col = (int(x) for x in pos.split(","))
            except ValueError:
                raise ValueError(f"坐标格式应为 name@row,col, 收到: {spec!r}")
        else:
            name = spec
            cells = transform_cells(resolve_cells(name), rot, flip)
            h = max(r for r, _ in cells) + 1
            w = max(c for _, c in cells) + 1
            row = int(torch.randint(0, max(1, size - h), (1,), generator=rng))
            col = int(torch.randint(0, max(1, size - w), (1,), generator=rng))
        stamp(g, name, row, col, rot, flip)
    return g


# --------------------------------------------------------------------------- #
# 性能基准
# --------------------------------------------------------------------------- #
def benchmark(size: int, gens: int, seed: int) -> None:
    grid = make_grid(size, "random", seed)

    def _time(device: torch.device) -> float:
        sim = LifeMPS(grid.clone(), device)
        # 预热（首次 kernel 编译/分配开销不计入）
        sim.step()
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sim.run(gens)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    print(f"\n基准测试: {size}x{size} 网格, {gens} 代\n" + "-" * 40)
    cpu_t = _time(torch.device("cpu"))
    print(f"CPU : {cpu_t:8.3f} s  ({gens / cpu_t:8.1f} 代/秒)")

    gpu = pick_device("mps")
    if gpu.type != "cpu":
        gpu_t = _time(gpu)
        print(f"{gpu.type.upper():3} : {gpu_t:8.3f} s  ({gens / gpu_t:8.1f} 代/秒)")
        print(f"\n加速比: {cpu_t / gpu_t:.2f}x")
    else:
        print("未检测到 MPS/CUDA，仅运行了 CPU。")


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def visualize(sim: LifeMPS, gens: int, interval: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(f"Game of Life - device: {sim.device.type.upper()}")
    ax.set_xticks([])
    ax.set_yticks([])
    img = ax.imshow(sim.to_numpy(), cmap="binary", interpolation="nearest")

    # 给盒子加一个边框 (环形边界的可视化边缘)
    H, W = sim.board.shape
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5), W, H,
            fill=False, edgecolor="#1f77b4", linewidth=2.5,
        )
    )
    ax.set_xlim(-1.5, W + 0.5)
    ax.set_ylim(H + 0.5, -1.5)

    state = {"gen": 0, "paused": False}
    dev = sim.device.type.upper()

    def title():
        tag = "  [PAUSED - press SPACE]" if state["paused"] else ""
        ax.set_title(f"Game of Life - {dev} - gen {state['gen']}{tag}")

    def update(_frame):
        if state["paused"]:          # 暂停时不推进, 只保持画面
            return (img,)
        sim.step()
        state["gen"] += 1
        img.set_data(sim.to_numpy())
        title()
        return (img,)

    # gens < 0 -> frames=None: 无限循环更新, 关闭窗口即停止
    frames = None if gens < 0 else gens
    # 保存引用，避免动画被 GC
    anim = FuncAnimation(
        fig, update, frames=frames, interval=interval, blit=False,
        repeat=False, cache_frame_data=False,
    )

    def on_key(event):
        if event.key == " ":         # 空格键切换暂停/继续
            state["paused"] = not state["paused"]
            try:                     # 真正停掉计时器, 暂停时不消耗帧
                anim.pause() if state["paused"] else anim.resume()
            except AttributeError:
                pass                 # 旧版 matplotlib 回退到 paused 标志
            title()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("提示: 在动画窗口中按 空格 暂停/继续")
    plt.show()


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="MPS 加速的康威生命游戏")
    p.add_argument("--size", type=int, default=256, help="网格边长 (默认 256)")
    p.add_argument(
        "--gens", type=int, default=500,
        help="代数 (默认 500)；设为 -1 表示无限运行 (Ctrl-C 或关窗停止)",
    )
    p.add_argument(
        "--pattern",
        default="random",
        help="单个初始图案 (居中放置)。库名、random, 或 .rle/.cells 路径。"
        "可用图案见 --list-patterns",
        metavar="NAME",
    )
    p.add_argument(
        "--rotate",
        choices=["0", "90", "180", "270"],
        default="0",
        help="--pattern 的朝向 (顺时针旋转角度), 共四种方向",
    )
    p.add_argument(
        "--flip", action="store_true", help="--pattern 水平镜像 (配合 --rotate 可得 8 向)"
    )
    p.add_argument(
        "--place",
        action="append",
        default=None,
        metavar="NAME[@ROW,COL][:DIR]",
        help="放置一个图案, 可重复使用以放置多个。DIR 为朝向 "
        "0/90/180/270 或 n/e/s/w(末尾加 m 镜像)。"
        "例: --place glider@10,10:e --place lwss@40,40:s "
        "(省略坐标则随机放置)",
    )
    p.add_argument(
        "--list-patterns", action="store_true", help="列出所有可用图案并退出"
    )
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument(
        "--interval", type=int, default=30, help="可视化帧间隔(ms)"
    )
    p.add_argument(
        "--no-wrap", action="store_true", help="使用固定边界而非环形边界"
    )
    p.add_argument(
        "--benchmark", action="store_true", help="跑性能基准, 不显示界面"
    )
    p.add_argument(
        "--headless", action="store_true", help="只计算不绘图(用于无显示环境)"
    )
    p.add_argument(
        "--save_t", type=int, default=None, metavar="X",
        help="把第 X 代的构象保存为 .rle 文件 (保存后从该代继续运行)",
    )
    p.add_argument(
        "--save_file", default=None, metavar="PATH",
        help="--save_t 的输出文件名 (默认 snapshot_t{X}.rle)",
    )
    args = p.parse_args()

    if args.list_patterns:
        print("可用图案:")
        groups = {
            "静物": ["block", "beehive", "loaf", "boat", "tub"],
            "振荡器": ["blinker", "toad", "beacon", "pulsar",
                       "pentadecathlon (周期15)", "queenbee (周期30)"],
            "反射器": ["snark (稳定90°滑翔机反射器)"],
            "飞船": ["glider", "lwss", "mwss", "hwss"],
            "枪": ["glider-gun"],
            "玛士撒拉": ["r-pentomino", "acorn", "diehard"],
        }
        for g, names in groups.items():
            print(f"  {g:6}: {', '.join(names)}")
        print("  其他   : random, 或任意 .rle/.cells 文件路径")
        return

    if args.benchmark:
        benchmark(args.size, args.gens, args.seed)
        return

    device = pick_device("mps")
    print(f"使用设备: {device.type.upper()}")

    if args.place:
        grid = parse_placements(args.place, args.size, args.seed)
    else:
        rot = {"0": 0, "90": 1, "180": 2, "270": 3}[args.rotate]
        grid = make_grid(args.size, args.pattern, args.seed, rot, args.flip)
    sim = LifeMPS(grid, device, wrap=not args.no_wrap)

    if args.save_t is not None:
        if args.save_t > 0:
            sim.run(args.save_t)
            if device.type == "mps":
                torch.mps.synchronize()
        out = args.save_file or f"snapshot_t{args.save_t}.rle"
        with open(out, "w") as f:
            f.write(board_to_rle(sim.to_numpy()))
        print(f"已保存第 {args.save_t} 代构象到 {out}")

    if args.headless:
        live0 = int(sim.board.sum().item())
        t0 = time.perf_counter()
        if args.gens < 0:
            # 无限运行: 每 1000 代打印一次速率, Ctrl-C 停止
            print("无限运行中 (Ctrl-C 停止)...")
            n = 0
            try:
                while True:
                    sim.run(1000)
                    if device.type == "mps":
                        torch.mps.synchronize()
                    n += 1000
                    dt = time.perf_counter() - t0
                    print(
                        f"  第 {n} 代  ({n / dt:.1f} 代/秒)  "
                        f"存活 {int(sim.board.sum().item())}"
                    )
            except KeyboardInterrupt:
                print(f"\n已停止于第 {n} 代。")
            return
        sim.run(args.gens)
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        live1 = int(sim.board.sum().item())
        print(
            f"运行 {args.gens} 代完成, 用时 {dt:.3f}s "
            f"({args.gens / dt:.1f} 代/秒)。 存活细胞 {live0} -> {live1}"
        )
        return

    visualize(sim, args.gens, args.interval)


if __name__ == "__main__":
    main()
