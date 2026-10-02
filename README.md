# PopStar 求解器

本地的同色块消除求解器：可以自己玩，也可以让求解器给出走法。规则只在 Python 里实现，网页只负责显示和把点击发给服务端。

玩法属于 SameGame 这一支，名字沿用了后来在手机上流行的 PopStar（消灭星星）。本仓库是独立实现，不含那款商业游戏的画面、关卡或源码。默认计分沿用公开介绍里的通行规则。出处写在 [SOURCES.md](SOURCES.md)。

基准规模是 **10×10、4 色**。更小的棋盘只用来确认程序能跑通。

## 规则

- 点一个同色四连通块（至少 2 格），整块消失。点块里任意一格都可以。
- 先竖直下落，再把空列向左压实。
- 没有可消块时结束。总分是各步得分之和，再加上清盘奖励：

```text
F = Σ g(k) + B(R)
```

消去 `n` 颗得 `5n²`。剩余 `R < 10` 时再加 `2000 − 5R`（清盘 2000），否则这项奖励为 0。这是《消灭星星》公开介绍里的通行计分。

## 启动

需要 Python 3.9 及以上，标准库即可，不必再装包。

Windows 双击 `启动.bat`，或在项目目录运行：

```text
python start.py
```

浏览器会打开游玩页（默认 `http://127.0.0.1:8765`）。服务只监听本机。端口被占用时会自动往后试。

```text
python start.py --page analyze    # 同一页面，直接打开分析标签
python start.py --page scan       # 同一页面，并弹出图片识别
python start.py --no-browser      # 只起服务
python start.py --port 9000
```

分析和图片识别都在这个页面里，没有单独入口。

## 页面上能做什么

- **游玩**：生成一局、点击消除、撤销、重开。点连通块里任意一格都会消掉整块。「提示」用 beam 搜索给出一步建议。
- **分析**：对当前初始局面求解，并在棋盘上标出下一步要消除的整块。用棋盘下方的「下一步」或「自动播放」逐步回放。也可以算终局 `(G, R)` 的 Pareto 前沿，再用 bonus 系数 λ 看该取哪一个终局。
- **从图片读取**：把截图里的棋盘还原成可玩局面。自动框选不准时，可以在图上拖动框选，再改行列数和误判的格子。

## 改计分

默认已经是通行规则。要改数字，打开 `popstar/scoring.py` 里「服务端计分」那一段：`GROUP_COEFFICIENT` 目前是 5，清盘奖励是 `2000 − 5R`（剩余少于 10 颗）。设了 `GROUP_GAIN_FN` 或 `CLEAR_REWARD_FN` 就整段换掉。改完重新运行 `启动.bat`，再刷新页面。

## 命令行

不打开网页时，可以用交互模式自己走棋：

```text
python main.py
python main.py --height 6 --width 6 --colors 3 --seed 7
python main.py --file board.txt
python main.py --auto
```

交互时输入连通块序号，或 `行 列`（从 0 开始），`q` 退出。

求解器入口是 `popstar.solver.solve`，`mode="fast"` 走 beam 搜索，`mode="exact"` 走带记忆化的精确搜索。10×10 上精确搜索通常证明不了最优，页面上的 fast 结果标的是 best found。

## 测试

```text
python -m unittest discover -s tests
```

图片识别回归测试需要本机有 Node。用 `识别测试/` 里的实机截图对拍时，还需要 Pillow 和 numpy；没有这些依赖时，对应测试会跳过，其余测试照常跑。

`python -m bench.bench10` 是 10×10 基准，全量大约几分钟，结果写到 `runs/`（该目录不纳入版本库）。其它测量脚本也在 `bench/` 里，同样用 `python -m bench.脚本名` 运行。

## 来源

规则来自 1985 年 Kuniaki Moribe 发表在《月刊 ASCII》上的 Chain Shot!，1992 年后以 SameGame 这个名字流传。手机上的 PopStar!《消灭星星》是这一玩法的商业版本（开发者 Brian Baek，约 2009 年；中文版由掌游天下引进）。

本仓库的程序、页面和图片识别都是在这里写的，没有搬入第三方求解器或游戏源码。求解用的是记忆化搜索、分支定界和 beam search。详细引用见 [SOURCES.md](SOURCES.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `popstar/board.py` | 唯一的规则实现：连通块、消除、下落、左压 |
| `popstar/scoring.py` | 计分，和棋盘逻辑分开 |
| `popstar/game.py` | 一局的走子、撤销和终局结算 |
| `popstar/solver.py` | beam 搜索、精确搜索、Pareto 前沿 |
| `popstar/generators.py` | 均匀、不均、聚集等开局生成 |
| `popstar/server.py` | 本机 HTTP 服务 |
| `ui/` | 页面、样式、图片识别 |
| `start.py`、`启动.bat` | 一键启动，打开游玩页 |
| `main.py` | 命令行走棋 |
| `bench/` | 基准和对照实验，不参与对局 |
| `tests/` | 单元测试 |
