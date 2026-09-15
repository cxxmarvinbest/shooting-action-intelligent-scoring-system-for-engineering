#!/bin/bash
# =============================================================================
# disk_clean.sh —— save_data 磁盘自动维护脚本（N9）
#
# 用途：当根分区使用率 >= 80% 时，从「最旧日期目录」开始，逐个删除整个
#       YYYY-MM-DD 目录，直到使用率 < 80% 为止。
#
# 部署（RK3588）：
#   1. 上传本脚本到服务器，如 /home/linaro/scripts/disk_clean.sh
#      chmod +x /home/linaro/scripts/disk_clean.sh
#   2. 宝塔面板 -> 计划任务 -> Shell 脚本，周期「每 2 小时」，命令：
#      /bin/bash /home/linaro/scripts/disk_clean.sh
#   等价 crontab：
#      0 */2 * * * /bin/bash /home/linaro/scripts/disk_clean.sh >> /dev/null 2>&1
#
# 安全护栏：
#   1. 严格正则 ^\d{4}-\d{2}-\d{2}$ 只匹配日期目录，其他目录一律不碰
#   2. 跳过当天目录（date +%F 白名单）
#   3. rm 前二次校验目标路径以 ROOT 开头且目录名是合法日期
#   4. flock 文件锁防并发
#   5. 删除前后记录 df / du，日志可追溯
#   6. 支持 DRY_RUN=1 演练模式（只打印不删）：DRY_RUN=1 bash disk_clean.sh
#
# 说明：本脚本是纯运维独立脚本，不依赖 Python 项目运行时，不占用 NPU。
# =============================================================================

set -u   # 未定义变量报错（不用 set -e，删除流程自行控制）

# ── 配置区（部署时按实际环境修改）────────────────────────────────────────
ROOT="/home/linaro/code/intelligent_scoring_system/save_data"  # save_data 绝对路径
THRESHOLD=80                                    # 触发阈值：分区使用率 %
MOUNT="/"                                       # 监视的挂载点（根分区）
LOG_DIR="/home/linaro/logs"                     # 日志目录
LOG_FILE="$LOG_DIR/disk_clean.log"              # 日志文件
LOCK_FILE="/tmp/disk_clean.lock"                # flock 锁文件
DATE_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}$'          # 严格日期正则
# ────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR" 2>/dev/null

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

# 获取分区使用率（整数，不带 %）。失败返回空。
get_usage() {
    df -P "$MOUNT" 2>/dev/null | awk 'NR==2 {gsub("%", "", $5); print $5}'
}

# 是否为合法整数（用于校验 usage）
is_int() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

# 安全删除：rm 前二次校验目标路径
#   - 目录名必须严格匹配 YYYY-MM-DD
#   - 目标路径必须以 "$ROOT/" 开头（防路径拼错/注入）
safe_remove() {
    local target="$1"
    local base
    base=$(basename "$target")
    case "$base" in
        [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
        *) log "跳过：目录名非法（$base），不删除"; return 1 ;;
    esac
    case "$target" in
        "$ROOT"/*) ;;
        *) log "跳过：路径越界（$target），不删除"; return 1 ;;
    esac
    [ -d "$target" ] || { log "跳过：不存在或非目录（$target）"; return 1; }

    if [ "${DRY_RUN:-0}" = "1" ]; then
        log "[DRY-RUN] 将删除: $target"
        return 0
    fi

    # ionice -c3 降低 IO 优先级（系统支持则用，否则直接 rm）
    if command -v ionice >/dev/null 2>&1; then
        ionice -c3 rm -rf "$target"
    else
        rm -rf "$target"
    fi
    log "已删除: $target"
    return 0
}

# ── 主流程 ────────────────────────────────────────────────────────────────

# flock 防并发：已有清理实例在跑则本轮直接退出
# 若系统无 flock（极少见），降级为无锁直行——每 2h 一次，并发风险可忽略，
# 不能因缺 flock 而永远跳过清理。
if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "检测到上一轮清理仍在执行，本轮跳过"
        exit 0
    fi
else
    log "警告：flock 不可用，跳过并发锁（每 2h 一次，并发风险可忽略）"
fi

usage=$(get_usage)
if ! is_int "$usage"; then
    log "获取分区使用率失败（df 输出异常: '$usage'），退出"
    exit 1
fi

log "──── 巡检开始 ────"
log "分区 $MOUNT 使用率 = ${usage}%"

if [ "$usage" -lt "$THRESHOLD" ]; then
    log "使用率 ${usage}% < ${THRESHOLD}%，无需清理，本次结束"
    exit 0
fi

# 触发清理：记录删除前状态
log "使用率 ${usage}% >= ${THRESHOLD}%，触发清理"
log "删除前 df: $(df -h "$MOUNT" 2>/dev/null | awk 'NR==2{print $2, $3, $5}')"
log "删除前 du: $(du -sh "$ROOT" 2>/dev/null | cut -f1) ($ROOT)"

TODAY=$(date '+%F')          # 当天日期白名单
deleted=0

# 列出所有合法日期目录，升序（最旧在前）；仅一级、仅目录
while IFS= read -r d; do
    [ -z "$d" ] && continue
    [ "$d" = "$TODAY" ] && { log "跳过当天目录（白名单）: $d"; continue; }
    # 跳过未来日期目录（时钟异常/误建的异常数据，绝不删）
    [[ "$d" > "$TODAY" ]] && { log "跳过未来日期目录（异常）: $d"; continue; }

    safe_remove "$ROOT/$d" || continue
    deleted=$((deleted + 1))

    usage=$(get_usage)
    if ! is_int "$usage"; then
        log "复查分区使用率失败，停止清理（已删 $deleted 个）"
        break
    fi
    log "删除后复查：分区 $MOUNT 使用率 = ${usage}%"

    if [ "$usage" -lt "$THRESHOLD" ]; then
        log "已降至 ${usage}% < ${THRESHOLD}%，清理达标，停止"
        break
    fi
done < <(find "$ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E "$DATE_RE" | sort)

# 记录删除后状态
log "删除后 df: $(df -h "$MOUNT" 2>/dev/null | awk 'NR==2{print $2, $3, $5}')"
log "删除后 du: $(du -sh "$ROOT" 2>/dev/null | cut -f1) ($ROOT)"

final_usage=$(get_usage)
log "本次清理结束：共删除 $deleted 个日期目录，最终使用率 = ${final_usage}%"

# 删光后仍 >= 阈值：磁盘压力可能不在 save_data，提示人工排查
if is_int "$final_usage" && [ "$final_usage" -ge "$THRESHOLD" ]; then
    log "警告：清理后使用率仍 >= ${THRESHOLD}%（当前 ${final_usage}%），save_data 已无可删日期目录，磁盘占用可能来自其他位置，请人工排查"
fi

exit 0
