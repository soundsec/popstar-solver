"""一键启动：起本地服务 + 自动打开浏览器。

用法::

    python start.py                  # 起服务并打开游玩页
    python start.py --page analyze   # 打开同一页面的分析标签
    python start.py --page scan      # 打开同一页面并弹出图片识别
    python start.py --no-browser     # 只起服务，不打开浏览器
    python start.py --port 9000

Windows 下也可以直接双击 ``启动.bat``。
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from typing import Optional

from popstar.server import UI_DIR, build_server

DEFAULT_PORT = 8765
MAX_PORT_TRIES = 20

# 所有入口都是同一个页面，只是初始标签不同（游玩 / 求解 / 图片识别已合并）
PAGES = {
    "solver": "/",
    "play": "/?tab=play",
    "analyze": "/?tab=analyze",
    "scan": "/?tab=scan",
}


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.35)
        return sock.connect_ex((host, port)) == 0


def pick_port(host: str, preferred: int, allow_scan: bool) -> int:
    """若端口被占用，向后找第一个空闲的（避免「一键启动」卡在端口冲突上）。"""
    if not allow_scan:
        return preferred
    for offset in range(MAX_PORT_TRIES):
        candidate = preferred + offset
        if not port_in_use(host, candidate):
            return candidate
    return preferred


def wait_until_up(host: str, port: int, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_in_use(host, port):
            return True
        time.sleep(0.1)
    return False


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="popstar 一键启动")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--page", choices=sorted(PAGES), default="play",
                        help="打开游玩页、分析标签，或直接弹出图片识别（默认 play）")
    parser.add_argument("--no-browser", action="store_true",
                        help="只起服务，不自动打开浏览器")
    parser.add_argument("--no-port-scan", action="store_true",
                        help="端口被占用时直接报错，不自动换端口")
    args = parser.parse_args(argv)

    if not os.path.isdir(UI_DIR):
        print(f"缺少 ui 目录：{UI_DIR}", file=sys.stderr)
        return 1

    port = pick_port(args.host, args.port, not args.no_port_scan)
    try:
        server = build_server(args.host, port)
    except OSError as exc:
        print(f"无法监听 {args.host}:{port} —— {exc}", file=sys.stderr)
        return 1

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base = f"http://{args.host}:{port}"
    url = base + PAGES[args.page]
    print("=" * 62)
    print("  PopStar 求解器 —— 本地已启动")
    print("=" * 62)
    print(f"  游玩页：{base}/?tab=play")
    print("  分析和图片识别都在这个页面里，不再分开入口。")
    print(f"\n  本次打开：{url}")
    print("  Ctrl+C 退出")
    print("=" * 62)

    if not wait_until_up(args.host, port):
        print("服务未能就绪，请检查端口。", file=sys.stderr)
        return 1

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:  # 浏览器打不开不应影响服务
            print(f"（未能自动打开浏览器：{exc}；请手动访问 {url}）")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n正在关闭…")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
