# -*- coding: utf-8 -*-
"""
查看 RKNN 模型输入/输出参数（含输入 W×H）。

原因说明：rknn-toolkit-lite2 是精简部署版，`NATIVE_INPUT_ATTR` 等张量属性查询
在不同 lite 版本里 flag 名可能不同、甚至不提供。本脚本先打印 `dir(RKNNLite)`，
再逐个尝试所有「长得像 query flag」的整数类属性，尽量适配任意版本。

在 RK3588 板上运行：
    python3 test/debug/check_rknn_model.py [模型路径]

不传路径默认查 weights/yolov8n-pose.rknn；可一次传多个路径。
"""

import os
import sys

# 疑似 query flag 的属性名关键字（用于过滤，避免拿无关整数去 query 崩掉运行时）
_FLAG_KEYWORDS = (
    "VERSION", "NUM", "SIZE", "ATTR", "INPUT", "OUTPUT", "MEM",
    "SDK", "NPU", "NATIVE", "CURRENT", "QUERY", "FREQ",
)


def _fmt_name(fmt):
    if isinstance(fmt, str):
        return fmt
    names = {0: "NCHW", 1: "NHWC"}
    return names.get(int(fmt), f"{fmt}")


def _attr_dict(attr):
    info = {}
    dims = getattr(attr, "dims", None)
    if dims is None:
        try:
            dims = list(attr)
        except Exception:
            dims = None
    if dims is not None and hasattr(dims, "__len__"):
        try:
            info["dims"] = [int(d) for d in dims]
        except Exception:
            info["dims"] = dims
    for name in ("fmt", "type", "size", "n_dims", "index", "qnt_type",
                 "scale", "zp", "name"):
        if hasattr(attr, name):
            info[name] = getattr(attr, name)
    return info


def _parse_wh(dims, fmt):
    if not dims or len(dims) < 2:
        return None, None
    d = [x for x in dims if x > 0]
    if d and d[0] == 1:
        d = d[1:]
    if len(d) < 3:
        return (d[-1], d[-2]) if len(d) >= 2 else (None, None)
    if "NHWC" in str(fmt) or str(fmt) == "1":
        return d[-2], d[-3]          # (H, W, C) -> W, H
    return d[-1], d[-2]              # NCHW (C, H, W) -> W, H


def _print_tensor_list(title, attrs):
    attrs = attrs if isinstance(attrs, (list, tuple)) else [attrs]
    print(title)
    for i, a in enumerate(attrs):
        info = _attr_dict(a)
        w, h = _parse_wh(info.get("dims"), info.get("fmt"))
        print(f"  [{i}]: dims={info.get('dims')} fmt={_fmt_name(info.get('fmt'))} "
              f"type={info.get('type')} size={info.get('size')}")
        if w and h:
            print(f"       => 宽 W={w}  高 H={h}")


def check_model(rknn_path):
    if not os.path.exists(rknn_path):
        print(f"[跳过] 模型文件不存在: {rknn_path}")
        return

    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("[错误] 未安装 rknn-toolkit-lite2，请在 RK3588 板上运行本脚本")
        return

    print("=" * 70)
    print(f"模型文件: {rknn_path}")

    rknn = RKNNLite()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        print(f"load_rknn 失败，ret={ret}")
        return

    # 1) 列出 RKNNLite 的整数类属性（疑似 query flag）
    print("\nRKNNLite 的整数类属性（query flag 候选）:")
    flag_candidates = {}
    for name in dir(RKNNLite):
        if name.startswith("_"):
            continue
        val = getattr(RKNNLite, name)
        if isinstance(val, int):
            flag_candidates[name] = val
    for name, val in flag_candidates.items():
        print(f"  {name} = {val}")

    # 2) 逐个尝试 query（只试名字含 flag 关键字的，避免无关整数崩运行时）
    print("\n逐个尝试 query:")
    found_input = found_output = False
    for name, val in flag_candidates.items():
        if not any(k in name.upper() for k in _FLAG_KEYWORDS):
            continue
        try:
            r = rknn.query(val)
        except Exception as e:
            print(f"  query({name}) 异常: {type(e).__name__}: {e}")
            continue
        print(f"  query({name}) = {r}")
        # 识别输入/输出属性
        if "INPUT" in name.upper() and "ATTR" in name.upper() and not found_input:
            _print_tensor_list(f"\n  输入属性（来自 {name}）:", r)
            found_input = True
        if "OUTPUT" in name.upper() and "ATTR" in name.upper() and not found_output:
            _print_tensor_list(f"\n  输出属性（来自 {name}）:", r)
            found_output = True

    if not (found_input or found_output):
        print("\n[提示] 当前 lite2 版本未提供张量属性查询 flag。")
        print("  替代方案：直接看运行时报错反推——跑一次推理，报错里会有")
        print("  'param input size(X) < model input size(Y)'，对 uint8-RGB 输入，Y/3 就是 W×H 的乘积，")
        print("  正方形模型 W=H=sqrt(Y/3)，例如 1228800/3=409600 -> 640×640。")
        print("  也可以把模型拷到装了「完整 rknn-toolkit2」的 PC 上查（PC 端工具链含张量属性查询）。")

    rknn.release()
    print()


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        default = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "weights", "yolov8n-pose.rknn")
        paths = [default]
    for p in paths:
        check_model(p)


if __name__ == "__main__":
    main()
