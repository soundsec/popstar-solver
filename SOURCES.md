# 来源与代码审查

发布到公开仓库之前核对过一遍：本仓库的程序是在这里写的，玩法规则有公开的历史来源，默认计分不是任何一款商业游戏的原公式。

## 代码

`popstar/`（含本机服务 `popstar/server.py`）、`ui/`、`start.py`、`tests/` 和 `bench/` 都是本仓库自己的实现。没有拷入第三方游戏客户端、求解器仓库或杂志上的 Chain Shot! 源码。

用到的算法是搜索里的常规做法，在本仓库里写成可对拍的几层：

- 同色四连通用泛洪填充
- 精确解用记忆化深度优先，可选分支定界、走法排序、颜色重编号后的置换表
- 近似解用 beam search，结果标明 best found
- 终局前沿是把「group 分 + 系数 × 清盘形状」当成对系数的直线，再找切换点

注释里的「第 N 节」是开发过程中的提纲编号（哪一层搜索、输出哪些字段、怎样和穷举对拍），不是某篇论文的章节号。

下面两篇是 SameGame 求解的公开研究，本仓库没有移植它们的算法，列出来是为了划清界限：

- Frank W. Takes, Walter A. Kosters. *Solving SameGame and its Chessboard Variant*. Leiden University. <https://liacs.leidenuniv.nl/~takesfw/pdf/samegame.pdf>
- Maarten P. D. Schadd, Mark H. M. Winands, Mandy J. W. Tak, Jos W. H. M. Uiterwijk. Single-player Monte-Carlo tree search for SameGame. *Knowledge-Based Systems* 34 (2012): 3–11. 介绍页：<https://project.dke.maastrichtuniversity.nl/games/games_samegame.htm>

图片识别（`ui/vision.js`）也是为本仓库的截图写的：按方块大小拟合格子，在避开中心图案的环带上取色，再按色相聚类。亮度权重 `0.299, 0.587, 0.114` 来自 ITU-R BT.601：<https://www.itu.int/rec/R-REC-BT.601>

仓库里没有商业游戏的图片、音效、关卡或名称标识。`识别测试/` 若放入实机截图，那些截图属于原游戏画面，公开仓库前请自行决定要不要放上去。

## 玩法从哪来

消除同色相邻块、掉落、空列横移，这一支规则始于：

- **Chain Shot!**（チェーンショット），Kuniaki Moribe（森部邦昭，亦署 Morisuke），1985 年 11 月发表在《月刊 ASCII》（Gekkan ASCII），原为富士通 FM-8 / FM-7 等机型，棋盘常见为 20×10、四色。
  - <https://en.wikipedia.org/wiki/SameGame>
  - <https://www.mobygames.com/game/206839/chain-shot/>
  - <http://wecmuseum.org/index.php/Chain_Shot>

1992 年 Eiji Fukumoto 在 Unix 上做的移植使用了 **SameGame** 这个名字；同年 Wataru Yoshioka 做到 PC-9801，之后出现 Clickomania、Jawbreaker、Bubble Breaker 等名称。一位后来的移植作者把这条脉络写在：

- <https://hp.vector.co.jp/authors/VA001976/softwares/macigame/mghistory_e.html>

各移植的计分并不统一。维基百科和上面的 Leiden 论文记录了 `(n−2)²`、`(n−1)²`、`n(n−1)`、`n²−3n+4` 等几种，很多版本另有清盘奖励或按剩余块扣分。本仓库没有采用其中某一种作为「原版公式」。

## PopStar / 消灭星星

本仓库标题里的 PopStar，指的是后来在手机上流行的那一款，不是 1985 年的原程序。

公开报道里的说法是：独立开发者 **Brian Baek** 约在 2009 年做出 **PopStar!**；中文版《消灭星星》约在 2014 年由掌游天下（北京）引进。这些是采访和百科的转述，不是本仓库能核对的原始合同：

- GameLook，2017-03-22：<http://www.gamelook.com.cn/2017/03/286790/>
- 壹读对开发者与掌游天下的采访：<https://read01.com/NaJ3dG.html>
- 百度百科「消灭星星」：<https://baike.baidu.com/item/%E6%B6%88%E7%81%AD%E6%98%9F%E6%98%9F/9963893>

玩法仍是 SameGame 那一套：至少两颗相邻同色星星一次消除，然后下落、空列左移。百科和攻略里常见的**商业版计分**是：

- 消去 `n` 颗得 `5 n²`
- 结束时剩余 `R < 10` 则奖励 `2000 − 5 R`，否则奖励 0

本仓库的默认计分与上面这条通行规则相同，写在 `popstar/scoring.py`：消去 `n` 颗得 `5n²`，剩余 `R < 10` 时奖励 `2000 − 5R`，否则为 0。搜索只调用这两个函数，没有为这套数字另写求解器。文件里留下的 `bonus_coefficient × (A−R)²` 是实验公式，默认对局不会用到。

PopStar、消灭星星等名称属于各自权利人。本仓库只用它们说明规则从哪一支游戏来，不是官方版本，也不提供那款游戏的客户端。
