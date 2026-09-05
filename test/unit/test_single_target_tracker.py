# -*- coding: utf-8 -*-
"""
SingleTargetTracker 单元测试（Windows 本地可跑，零依赖 RKNN）
=============================================================
覆盖场景：
  1. 身份跳变（多人同屏面积互换）→ 锁定 IoU 最大的同一人，不跟面积跳
  2. 单帧误检 → tentative 续不上，不 confirmed，直接放弃
  3. 单帧漏检 → confirmed 后 hangover 兜底，框不丢
  4. 连续漏检超限 → hangover 耗尽后重置回 NO_TARGET
  5. EMA 平滑 → 输出落在新旧框之间
  6. 一阶外推 → 漏检帧沿速度方向平移
  7. 便捷入口 update_from_dets / extract_player_candidates
"""

import os
import sys

# 把项目根加入 sys.path（test/ 的上一级）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision_algorithm.detection.single_target_tracker import SingleTargetTracker as T


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _near(a, b, tol=1e-6):
    return abs(a - b) <= tol


def main():
    passed, failed = 0, 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name}  {detail}")

    # ==================== 场景 1：身份跳变 ====================
    print("\n① 身份跳变（多人同屏面积互换，应锁定同一人）")
    tr = T(iou_thresh=0.3, hangover=3, confirm=1, ema_alpha=1.0)
    A0 = (100, 100, 200, 300)   # 面积 100*200=20000，首帧更大 → 采信
    B0 = (400, 100, 480, 300)   # 面积 80*200=16000
    A1 = (105, 105, 205, 305)   # A 轻微移动（IoU 高）
    B1 = (400, 100, 500, 300)   # B 突然变大（面积 20000 > A，静态会选 B）

    box, st = tr.update([A0, B0])
    check("首帧采信面积最大的 A", tr.locked_box == A0 and st == T.TENTATIVE)
    box, st = tr.update([A1, B1])
    # tracker 应续 A1（IoU 最大），而非面积更大的 B1
    iou_A = T._iou(box, A1)
    iou_B = T._iou(box, B1)
    check("锁定 A 而非面积更大的 B", iou_A > 0.8 and iou_B < 0.1,
          f"iou_A={iou_A:.3f} iou_B={iou_B:.3f}")
    check("状态已 CONFIRMED", st == T.CONFIRMED)

    # ==================== 场景 2：单帧误检 ====================
    print("\n② 单帧误检（tentative 续不上，应放弃）")
    tr = T(iou_thresh=0.3, hangover=3, confirm=1, ema_alpha=1.0)
    box, st = tr.update([])
    check("空场 → NO_TARGET", st == T.NO_TARGET and box is None)
    box, st = tr.update([(50, 50, 120, 200)])   # 单帧闪现误检
    check("误检帧 → TENTATIVE（暂定未确认）", st == T.TENTATIVE and box is not None)
    box, st = tr.update([])                       # 下一帧消失
    check("下一帧消失 → 放弃回 NO_TARGET", st == T.NO_TARGET and box is None)

    # ==================== 场景 3：单帧漏检（兜底） ====================
    print("\n③ 单帧漏检（confirmed 后 hangover 兜底）")
    tr = T(iou_thresh=0.3, hangover=3, confirm=1, ema_alpha=1.0)
    tr.update([(100, 100, 200, 300)])             # tentative
    box, st = tr.update([(100, 100, 200, 300)])   # confirmed
    check("已 CONFIRMED", st == T.CONFIRMED)
    box, st = tr.update([])                       # 漏检 1 帧
    check("漏检帧 → HOLDING 且框非空", st == T.HOLDING and box is not None)
    box, st = tr.update([(102, 100, 202, 300)])   # 恢复命中
    check("恢复 → CONFIRMED", st == T.CONFIRMED and box is not None)

    # ==================== 场景 4：连续漏检超限 ====================
    print("\n④ 连续漏检超限（hangover=3 耗尽后重置）")
    tr = T(iou_thresh=0.3, hangover=3, confirm=1, ema_alpha=1.0)
    tr.update([(100, 100, 200, 300)])
    tr.update([(100, 100, 200, 300)])   # confirmed
    states = []
    for _ in range(5):
        _, st = tr.update([])
        states.append(st)
    check("前 3 帧兜底 HOLDING", states[:3] == [T.HOLDING] * 3, f"got {states[:3]}")
    check("第 4 帧起重置 NO_TARGET", states[3] == T.NO_TARGET and states[4] == T.NO_TARGET,
          f"got {states[3:]}")

    # ==================== 场景 5：EMA 平滑 ====================
    print("\n⑤ EMA 平滑（alpha=0.5，输出落在新旧框中间）")
    tr = T(iou_thresh=0.3, hangover=3, confirm=1, ema_alpha=0.5)
    tr.update([(100, 100, 200, 200)])             # 中心 (150,150)
    box, st = tr.update([(120, 100, 220, 200)])   # 命中(IoU=0.667)，新中心 (170,150)
    cx, cy = _center(box)
    # EMA: cx = 0.5*170 + 0.5*150 = 160
    check("中心 X 落在 150 与 170 中间（EMA 平滑）", _near(cx, 160.0), f"cx={cx}")
    check("中心 Y 不变", _near(cy, 150.0), f"cy={cy}")

    # ==================== 场景 6：一阶外推 ====================
    print("\n⑥ 一阶外推（漏检帧沿速度方向平移）")
    tr = T(iou_thresh=0.3, hangover=3, confirm=1, ema_alpha=1.0)
    tr.update([(100, 100, 200, 200)])             # 中心 (150,150)
    tr.update([(110, 100, 210, 200)])             # 中心 (160,150) → 速度 (+10, 0)
    box, st = tr.update([])                       # 漏检 → 外推
    cx, cy = _center(box)
    check("外推中心 X=170（沿速度 +10）", _near(cx, 170.0), f"cx={cx}")
    check("外推状态 HOLDING", st == T.HOLDING)

    # ==================== 场景 7：便捷入口 ====================
    print("\n⑦ 便捷入口 update_from_dets / extract_player_candidates")
    dets = [
        {'box': (100, 100, 200, 300), 'cls': 0, 'conf': 0.9},   # player
        {'box': (10, 10, 30, 30), 'cls': 1, 'conf': 0.8},       # ball
        {'box': (400, 100, 500, 300), 'cls': 0, 'conf': 0.7},   # player
    ]
    cands = T.extract_player_candidates(dets, player_cls_id=0)
    check("只提取 cls==0 的 player", cands == [(100, 100, 200, 300), (400, 100, 500, 300)],
          f"got {cands}")
    tr = T(iou_thresh=0.3, hangover=3, confirm=1, ema_alpha=1.0)
    box, st = tr.update_from_dets(dets)
    check("update_from_dets 采信面积最大 player",
          tr.locked_box == (100, 100, 200, 300) and st == T.TENTATIVE)

    # ==================== 汇总 ====================
    print("\n" + "=" * 56)
    print(f"结果：{passed} 通过 / {failed} 失败")
    if failed == 0:
        print("=> 全部通过")
    else:
        print("=> 存在失败，请检查")
    print("=" * 56)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
