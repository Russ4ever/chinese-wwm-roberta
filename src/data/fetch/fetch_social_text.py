# -*- coding: utf-8 -*-
"""抓取社交文本(datayes FTP: ftp.datayes.com)。

⚠ 当前网络环境(2026-08 实测)无法访问该 FTP:
   - 域名 DNS 解析失败;直接 FTP 被网关拦截;ftp 代理 squid 403;http 代理要求 Basic 认证。
   因此本脚本默认只做"连接 + 目录自检",能通后才能按日期窗口下载。

目录结构未知,先用 --list-only 看根目录/已配置的 FTP_REMOTE_ROOT 结构,
确认后把远端路径填到 config.FTP_REMOTE_ROOT,再正式下载。

用法:
    python data_fetch/fetch_social_text.py --list-only        # 列目录
    python data_fetch/fetch_social_text.py                     # 下载窗口内文件
输出:
    <OUTPUT_ROOT>/social_text/<远端相对路径>
"""
import os
import sys
import time
import argparse
import ftplib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from common import window_dates, ProgressBar


def connect():
    ftp = ftplib.FTP()
    ftp.connect(cfg.FTP_HOST, cfg.FTP_PORT, timeout=25)
    ftp.login(cfg.FTP_USER, cfg.FTP_PWD)
    ftp.set_pasv(True)
    return ftp


def list_dir(ftp, path):
    """返回 (entries, is_dir_map)。优先 MLSD,失败回退 LIST。"""
    lines = []
    try:
        ftp.retrlines(f"MLSD {path}", lines.append)
        items = []
        for ln in lines:
            facts, _, name = ln.rpartition(" ")
            kind = "dir" if 'type=dir' in facts else "file"
            items.append((name, kind))
        return items
    except Exception:
        pass
    lines = []
    ftp.retrlines(f"LIST {path}", lines.append)
    items = []
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        kind = "dir" if parts[0].startswith("d") else "file"
        name = " ".join(parts[8:])
        items.append((name, kind))
    return items


def date_in_window(name):
    """从文件名/目录名里猜日期(YYYYMMDD / YYYY-MM-DD / YYYYMM),判断是否落在窗口。"""
    import re
    s, e, _ = window_dates()
    m = re.search(r"(20\d{2})[-]?(\d{2})[-]?(\d{2})", name)
    if m:
        try:
            from datetime import datetime
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
            return s <= d <= e
        except ValueError:
            return False
    return False


def download_tree(ftp, remote_root, local_root, dry=True):
    """递归下载。dry=True 时只打印将下载的文件。"""
    got = 0
    for name, kind in list_dir(ftp, remote_root):
        rpath = f"{remote_root}/{name}" if remote_root else name
        lpath = os.path.join(local_root, name)
        if kind == "dir":
            if date_in_window(name):
                os.makedirs(lpath, exist_ok=True)
                got += download_tree(ftp, rpath, lpath, dry=dry)
        else:
            if date_in_window(name):
                if dry:
                    print(f"  [计划下载] {rpath}")
                    got += 1
                else:
                    os.makedirs(local_root, exist_ok=True)
                    size = None
                    try:
                        size = ftp.size(rpath)
                    except Exception:
                        size = None
                    pb = ProgressBar(total=size, desc=os.path.basename(rpath), unit="B")

                    def _cb(block):
                        pb.update(len(block))

                    with open(lpath, "wb") as f:
                        ftp.retrbinary(f"RETR {rpath}", lambda b: (f.write(b), _cb(b)))
                    pb.close()
                    got += 1
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true", help="只列出远端目录结构")
    ap.add_argument("--download", action="store_true", help="实际下载(需先确认目录结构)")
    args = ap.parse_args()

    t0 = time.time()
    try:
        ftp = connect()
    except Exception as e:
        print(f"[FAIL] FTP 连接失败: {e}")
        print("提示: 当前机器无法访问 ftp.datayes.com(DNS/网关/代理均不通),")
        print("      需先确认公司内对该 FTP 的正常访问方式(直连/VPN/代理账号)。")
        sys.exit(2)

    print(f"[OK] 已连接 {cfg.FTP_HOST},欢迎消息: {ftp.getwelcome()}")
    root = cfg.FTP_REMOTE_ROOT
    print(f"=== 根目录: '{root or '/'}' ===")
    for name, kind in list_dir(ftp, root):
        print(f"  [{kind}] {name}")

    if args.download:
        local_root = os.path.join(cfg.OUTPUT_ROOT, "social_text")
        n = download_tree(ftp, root, local_root, dry=False)
        print(f"[OK] 下载 {n} 个文件 -> {local_root} ({time.time() - t0:.1f}s)")
    else:
        print(f"== (dry-run) 当前窗口 {cfg.START_DATE}~{cfg.END_DATE} 匹配文件数: "
              f"{download_tree(ftp, root, os.path.join(cfg.OUTPUT_ROOT, 'social_text'), dry=True)}")
        print("确认目录结构后,把远端路径填入 config.FTP_REMOTE_ROOT,再带 --download 执行。")
    ftp.quit()


if __name__ == "__main__":
    main()
