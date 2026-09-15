# -*- coding: utf-8 -*-
"""
2.2 模型信息诊断：直接输出 RKNN 模型的输入/输出维度、张量、量化类型、runtime 版本。

对齐《测试报告》2.2 表列：
    模型文件 | 输入维度 | 输出维度 | 量化类型 | runtime 版本 | 是否匹配 | 备注

在 RK3588 板上运行（项目根目录）：
    python3 test/debug/check_rknn_model.py \
        weights/best_int8.rknn \
        weights/yolov8n_pose_int8_320.rknn

不传参数时，默认诊断项目内两个模型：
    weights/best_int8.rknn              检测，2 类（0=player, 1=basketball）
    weights/yolov8n_pose_int8_320.rknn  姿态，17 点

可选参数：
    --json <目录>   把结果落盘为 JSON（默认 data/output），便于回填报告。

设计说明：
  - rknn-toolkit-lite2 是精简部署版，`NATIVE_INPUT_ATTR` 等张量属性查询 flag
    在不同 lite 版本里名称可能不同、甚至不提供。本脚本先列出 `dir(RKNNLite)`
    里所有整数类属性，再按关键字匹配输入/输出属性 flag，尽量适配任意版本。
  - 若 lite2 完全不提供张量查询，脚本会打印替代方案（见结尾 fallback）。
  - 「是否匹配」为轻量启发式：检测模型输入应为 640x640、姿态模型应为 320x320；
    类别/关键点通道数从输出张量反推并作为备注输出，最终以人工核对为准。
"""

import argparse
import json
import os
import sys
import time

# 疑似 query flag 的属性名关键字（用于过滤，避免拿无关整数去 query 崩掉运行时）
_FLAG_KEYWORDS = (
    "VERSION", "NUM", "SIZE", "ATTR", "INPUT", "OUTPUT", "MEM",
    "SDK", "NPU", "NATIVE", "CURRENT", "QUERY", "FREQ", "CORE",
    "PRIORITY", "WEIGHT", "NORM",
)

# ── RKNN 枚举 -> 可读名 ──────────────────────────────────────────
# 量化类型（rknn_tensor_qnt_type）
_QNT_NAMES = {
    0: "NONE(浮点/未量化)",
    1: "DFP(动态定点)",
    2: "AFFINE_ASYMMETRIC(非对称 int8)",
    3: "AFFINE_SYMMETRIC(对称 int8)",
    4: "BFP16",
}
# 张量 dtype（rknn_tensor_type）
_TYPE_NAMES = {
    0: "FLOAT32", 1: "FLOAT16", 2: "INT8", 3: "UINT8", 4: "INT16",
    5: "UINT16", 6: "INT32", 7: "UINT32", 8: "INT64", 9: "UINT64",
    10: "BOOL", 11: "INT4", 12: "BFLOAT16",
}
# 布局（rknn_tensor_format）
_FMT_NAMES = {
    0: "NCHW", 1: "NHWC", 2: "NC1HWC2", 3: "UNDEFINED",
}


def _fmt_qnt(v):
    if v is None:
        return "?"
    return _QNT_NAMES.get(int(v), f"QNT_{v}")


def _fmt_type(v):
    if v is None:
        return "?"
    return _TYPE_NAMES.get(int(v), f"TYPE_{v}")


def _fmt_fmt(v):
    if isinstance(v, str):
        return v
    if v is None:
        return "?"
    return _FMT_NAMES.get(int(v), f"FMT_{v}")


def _attr_dict(attr):
    """把一个张量属性对象规整为 dict，兼容不同 lite 版本字段差异。"""
    info = {}
    dims = getattr(attr, "dims", None)
    if dims is None:
        try:
            dims = list(attr)
        except Exception:
            dims = None
    if dims is not None and hasattr(dims, "__len__"):
        try:
            info["dims"] = [int(d) for d in dims if d is not None]
        except Exception:
            info["dims"] = list(dims)
    for name in ("fmt", "type", "size", "n_dims", "index", "qnt_type",
                 "scale", "zp", "name", "pass_through"):
        if hasattr(attr, name):
            info[name] = getattr(attr, name)
    return info


def _parse_wh(dims, fmt):
    """从 dims 解析输入/输出的宽 W 与高 H（静态模型，dims 无 0）。"""
    if not dims or len(dims) < 2:
        return None, None
    d = [x for x in dims if x and x > 0]
    if d and d[0] == 1:          # 去掉 batch=1
        d = d[1:]
    if len(d) < 3:
        return (d[-1], d[-2]) if len(d) >= 2 else (None, None)
    if "NHWC" in str(fmt) or str(fmt) == "1":
        return d[-2], d[-3]      # (H, W, C) -> W, H
    return d[-1], d[-2]          # NCHW (C, H, W) -> W, H


def _discover_flags(rknn_cls):
    """列出 RKNNLite 类上的所有整数类属性，作为 query flag 候选。"""
    flags = {}
    for name in dir(rknn_cls):
        if name.startswith("_"):
            continue
        val = getattr(rknn_cls, name)
        if isinstance(val, int):
            flags[name] = val
    return flags


def _query_sdk_version(rknn, flags):
    """尽量取 SDK/runtime 版本字符串；取不到返回 None。"""
    for key in ("SDK_VERSION", "DRV_VERSION", "API_VERSION"):
        if key in flags:
            try:
                r = rknn.query(flags[key])
                if isinstance(r, (str, bytes)):
                    return r.decode() if isinstance(r, bytes) else r
                return r
            except Exception:
                pass
    # 兜底：直接读已导入模块的 __version__
    for mod_name in ("rknnlite", "rknnlite.api.rknn_lite"):
        try:
            mod = __import__(mod_name, fromlist=["__version__"])
            v = getattr(mod, "__version__", None)
            if v:
                return str(v)
        except Exception:
            continue
    return None


def _pick_attr_flag(flags, is_input):
    """按关键字优先级挑一个输入/输出属性 flag。"""
    want = "INPUT" if is_input else "OUTPUT"
    opposite = "OUTPUT" if is_input else "INPUT"
    # 优先级：NATIVE_INPUT_ATTR > INPUT_ATTR > IN_ATTR > 仅含 INPUT 的 attr
    ordered = []
    if is_input:
        ordered = ["NATIVE_INPUT_ATTR", "INPUT_ATTR", "IN_ATTR", "INPUT_NUM",
                   "GET_INPUT"]
    else:
        ordered = ["NATIVE_OUTPUT_ATTR", "OUTPUT_ATTR", "OUT_ATTR", "OUTPUT_NUM",
                   "GET_OUTPUT"]
    for key in ordered:
        if key in flags:
            return key, flags[key]
    # 兜底：名字里含 want 且含 ATTR 的
    for key, val in flags.items():
        u = key.upper()
        if "ATTR" in u and want in u and opposite not in u:
            return key, val
    return None, None


def _query_tensors(rknn, flags, is_input):
    """查询并返回 (张量列表, 来源 flag 名)。失败返回 (None, None)。"""
    key, val = _pick_attr_flag(flags, is_input)
    if key is None:
        return None, None
    try:
        r = rknn.query(val)
    except Exception as e:
        print(f"    query({key}) 异常: {type(e).__name__}: {e}")
        return None, None
    if r is None:
        return None, key
    if isinstance(r, (list, tuple)):
        return list(r), key
    return [r], key


def _print_tensors(title, attrs, src_flag):
    """打印张量属性清单，返回规整后的 dict 列表。"""
    attrs = attrs if isinstance(attrs, (list, tuple)) else [attrs]
    print(title + f"   (来源 flag: {src_flag})")
    result = []
    for i, a in enumerate(attrs):
        info = _attr_dict(a)
        w, h = _parse_wh(info.get("dims"), info.get("fmt"))
        qnt = _fmt_qnt(info.get("qnt_type"))
        dtype = _fmt_type(info.get("type"))
        fmt = _fmt_fmt(info.get("fmt"))
        info["W"], info["H"] = w, h
        info["qnt_name"] = qnt
        info["type_name"] = dtype
        info["fmt_name"] = fmt
        result.append(info)
        line = (f"  [{i}] dims={info.get('dims')}  fmt={fmt}  "
                f"type={dtype}  量化={qnt}")
        if info.get("size") is not None:
            line += f"  size={info['size']}B"
        print(line)
        if w and h:
            print(f"       => 输入宽 W={w}  高 H={h}")
        # 量化参数（scale/zp），量化模型必带
        scale = info.get("scale")
        zp = info.get("zp")
        if scale is not None or zp is not None:
            print(f"       => scale={scale}  zero_point={zp}")
    return result


def _classify_model(path):
    """按文件名粗分模型类型，返回 (kind, 预期输入边长, 备注)。"""
    base = os.path.basename(path).lower()
    if "pose" in base:
        return "pose", 320, "姿态，17 点（yolov8-pose）"
    return "det", 640, "检测，2 类（0=player, 1=basketball）"


def _infer_remark(outputs_info):
    """从输出张量反推类别/关键点通道，作为备注参考。"""
    hints = []
    for o in outputs_info:
        dims = o.get("dims") or []
        for d in dims:
            if d == 66:
                hints.append("检出 2 类（66=64 box + 2 cls 合并式）")
            elif d == 65:
                hints.append("检出 1 类（65=64 box + 1 cls 合并式）")
            elif d == 51:
                hints.append("检出关键点分支 51 通道（17 点 × 3）")
            elif d == 116:
                hints.append("检出姿态合并输出（64 box + 1 cls + 51 kps）")
            elif d == 17 and 3 in dims:
                hints.append("检出已解码关键点 (17,3) 张量")
    return "；".join(dict.fromkeys(hints))


def _annotate_output_roles(path, outputs_info):
    """给输出张量标注角色（pose 4 输出：前 3 = box，第 4 = flatten 关键点）。

    返回 list of str（与 outputs_info 等长）。仅对 4 输出且文件名含 pose 的模型生效，
    其余情况按通道数做通用标注。
    """
    base = os.path.basename(path).lower()
    n = len(outputs_info)
    roles = []
    if n == 4 and "pose" in base:
        roles = ["box 输出（尺度1）", "box 输出（尺度2）", "box 输出（尺度3）",
                 "flatten 关键点输出（outputs[3]）"]
    else:
        for o in outputs_info:
            dims = o.get("dims") or []
            if any(d == 51 for d in dims):
                roles.append("关键点分支（51=17×3）")
            elif any(d in (64, 65, 66) for d in dims):
                roles.append("box(+cls) 分支")
            else:
                roles.append("-")
    return roles


def diagnose_model(rknn_path):
    """诊断单个模型，返回用于 JSON 的结果 dict（诊断失败返回 None）。"""
    if not os.path.exists(rknn_path):
        print(f"[跳过] 模型文件不存在: {rknn_path}")
        return None

    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("[错误] 未安装 rknn-toolkit-lite2，请在 RK3588 板上运行本脚本")
        return None

    print("=" * 78)
    print(f"模型文件: {rknn_path}")
    try:
        print(f"文件大小: {os.path.getsize(rknn_path) / 1e6:.2f} MB")
    except OSError:
        pass

    rknn = RKNNLite()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        print(f"load_rknn 失败，ret={ret}")
        return None

    flags = _discover_flags(RKNNLite)
    sdk = _query_sdk_version(rknn, flags)
    print(f"runtime 版本: {sdk if sdk else '(无法获取，见下方 flag 清单)'}")

    # 打印 flag 清单（便于核对哪个 lite 版本提供哪些 query）
    print("\nRKNNLite 可用的 query flag（整数类属性）:")
    for name, val in sorted(flags.items(), key=lambda kv: kv[0]):
        print(f"  {name} = {val}")

    # 输入/输出张量
    print()
    in_info, in_flag = _query_tensors(rknn, flags, is_input=True)
    out_info, out_flag = _query_tensors(rknn, flags, is_input=False)

    if in_info is None and out_info is None:
        print("\n[提示] 当前 lite2 版本未提供张量属性查询 flag。")
        print("  替代方案：")
        print("  1) 跑一次推理看报错反推 W×H——报错 'param input size(X) < model input size(Y)'，")
        print("     对 uint8-RGB 输入，Y/3 即 W×H 乘积；正方形模型 W=H=sqrt(Y/3)。")
        print("     例：1228800/3=409600 -> 640×640。")
        print("  2) 把模型拷到装了「完整 rknn-toolkit2」的 PC 上，用 RKNN.query(NATIVE_INPUT_ATTR)")
        print("     与 NATIVE_OUTPUT_ATTR 查张量属性。")
        rknn.release()
        print()
        return {"file": rknn_path, "runtime_version": sdk,
                "input": None, "outputs": None, "queryable": False,
                "match": None, "remark": "lite2 不提供张量查询，需按替代方案核对"}

    ins = _print_tensors("输入张量:", in_info, in_flag) if in_info is not None else []
    outs = _print_tensors("输出张量:", out_info, out_flag) if out_info is not None else []

    # 输出角色识别（区分 box 输出 vs flatten 关键点输出）
    if outs:
        roles = _annotate_output_roles(rknn_path, outs)
        print("\n输出张量角色识别:")
        for i, (o, role) in enumerate(zip(outs, roles)):
            print(f"  outputs[{i}] <- {role}  dims={o.get('dims')}")

    # 量化类型：以输入张量为准（输入 dtype + qnt 是模型量化方式的直观体现）
    quant = "?"
    if ins:
        quant = f"{ins[0]['type_name']} / {ins[0]['qnt_name']}"
    print(f"\n量化类型（输入张量）: {quant}")

    # 是否匹配：输入 W×H 是否符合该模型预期（检测 640 / 姿态 320）
    kind, expect_wh, remark = _classify_model(rknn_path)
    w = ins[0]["W"] if ins else None
    h = ins[0]["H"] if ins else None
    match = None
    if w is not None and h is not None:
        match = (w == expect_wh and h == expect_wh)
    extra_remark = _infer_remark(outs) if outs else ""
    if extra_remark:
        remark = f"{remark}；{extra_remark}"

    print("\n" + "-" * 78)
    print("2.2 表回填速览:")
    print(f"  模型文件   : {os.path.basename(rknn_path)}")
    print(f"  输入维度   : {ins[0]['dims'] if ins else '?'}  (W×H = {w}x{h})")
    print(f"  输出维度   : {[o.get('dims') for o in outs] if outs else '?'}")
    print(f"  量化类型   : {quant}")
    print(f"  runtime版本: {sdk if sdk else '?'}")
    print(f"  是否匹配   : {'是' if match else ('否' if match is False else '待定')}"
          + (f"（输入 {w}x{h}，预期 {expect_wh}x{expect_wh}）"
             if w and h else ""))
    print(f"  备注       : {remark}")
    print("-" * 78)

    rknn.release()
    print()
    return {
        "file": rknn_path,
        "size_bytes": os.path.getsize(rknn_path),
        "runtime_version": sdk,
        "input": ins,
        "outputs": outs,
        "quantization": quant,
        "queryable": True,
        "match": match,
        "expected_wh": [expect_wh, expect_wh],
        "remark": remark,
    }


def main():
    ap = argparse.ArgumentParser(description="RKNN 模型信息诊断（报告 2.2）")
    ap.add_argument("models", nargs="*", help="RKNN 模型路径（可多个；默认项目内两个模型）")
    ap.add_argument("--json", default=None,
                    help="JSON 落盘目录（默认 data/output；传空串 '' 关闭落盘）")
    args = ap.parse_args()

    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.models:
        paths = args.models
    else:
        # 默认从 Config 读取检测/姿态模型路径（跟随项目配置），读不到则回退 weights 目录
        paths = []
        try:
            sys.path.insert(0, proj_root)
            from config import Config
            paths = [Config.DET_RKNN_PATH, Config.POSE_RKNN_PATH]
        except Exception:
            weights_dir = os.path.join(proj_root, "weights")
            paths = [
                os.path.join(weights_dir, "best_int8.rknn"),
                os.path.join(weights_dir, "yolov8n_pose_int8_320.rknn"),
            ]

    results = []
    for p in paths:
        r = diagnose_model(p)
        if r is not None:
            results.append(r)

    # JSON 落盘（默认 data/output；--json '' 显式关闭）
    if args.json != "":
        out_dir = args.json or os.path.join(proj_root, "data", "output")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"model_info_report_{ts}.json")
        payload = {"generated_at": ts, "models": results}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"报告已落盘: {out_path}")


if __name__ == "__main__":
    main()
