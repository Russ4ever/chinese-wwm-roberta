# -*- coding: utf-8 -*-
"""无第三方依赖的进度条,输出到 stderr,供取数脚本复用。

total 已知时显示百分比 + 剩余时间;未知时显示计数 + 速率。
stderr 是终端时同一行刷新,被重定向/管道时周期性打一行,避免看不到进度。
"""
import sys
import time


class ProgressBar:
    """进度条(不依赖 tqdm)。

    - total 为 int 时:显示 |████░░| 百分比、N/total、速率、用时、剩余。
    - total 为 None 时:显示旋转符、N、用时、速率(流式,总数未知)。
    """

    _SPIN = "|/-\\"

    def __init__(self, total=None, desc="", unit="行", min_interval=0.5, width=30):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.min_interval = min_interval
        self.width = width
        self.n = 0
        self._t0 = time.time()
        self._last = 0.0
        self._spin_i = 0
        self._tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def update(self, n=1):
        self.n += n
        now = time.time()
        if now - self._last < self.min_interval:
            return
        self._last = now
        self._render(now)

    def _render(self, now):
        elapsed = now - self._t0
        rate = self.n / elapsed if elapsed > 0 else 0.0
        if self.total:
            pct = min(100.0, 100.0 * self.n / max(1, self.total))
            filled = int(self.width * pct / 100.0)
            bar = "█" * filled + "░" * (self.width - filled)
            eta = (self.total - self.n) / rate if rate > 0 else 0.0
            msg = (f"{self.desc} |{bar}| {pct:5.1f}%  "
                   f"{self.n:,}/{self.total:,}  {rate:,.0f}{self.unit}/s  "
                   f"用时{elapsed:.0f}s 剩{eta:.0f}s")
        else:
            sp = self._SPIN[self._spin_i % len(self._SPIN)]
            self._spin_i += 1
            msg = (f"{self.desc} {sp} {self.n:,} {self.unit}  "
                   f"用时{elapsed:.0f}s  {rate:,.0f}{self.unit}/s")
        if self._tty:
            sys.stderr.write("\r" + msg + "\033[K")
        else:
            sys.stderr.write(msg + "\n")
        sys.stderr.flush()

    def close(self):
        if self._tty:
            self._render(time.time())
            sys.stderr.write("\n")
        sys.stderr.flush()
