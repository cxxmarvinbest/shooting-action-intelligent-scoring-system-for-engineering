#!/bin/bash
# =============================================================================
# keepalive.sh —— pipeline.py 存活守护脚本（单脚本版）
#
# 职责（只做一件事）：检查 pipeline.py 是否存活，未存活则启动。
#   1. 进程不在 → setsid 脱离进程组启动 pipeline.py；
#   2. 进程在但 /health 卡死 → SIGTERM 优雅重启；
#   3. flock 文件锁防多实例并发；全程写日志。
#
# 关键修复（相比旧版）：
#   - 用 `setsid nohup ... &` 启动，让 pipeline.py 进入独立会话、脱离宝塔
#     cron 的进程组。否则宝塔 cron 任务结束时会对整个进程组清理，把
#     nohup 起的 pipeline.py 连坐杀掉（日志表现为「收到退出信号 2」SIGINT），
#     导致进程反复「启动→被杀→再启动」的死循环。
#
# 宝塔配置（只保留这一条计划任务）：
#   任务类型：Shell 脚本
#   执行周期：每 2 分钟（N 分钟）
#   脚本内容：/bin/bash /home/linaro/code/intelligent_scoring_system/deploy/keepalive.sh
#
# 首次手动验证（板子终端）：
#   sudo /bin/bash /home/linaro/code/intelligent_scoring_system/deploy/keepalive.sh
#   tail -f /home/linaro/code/intelligent_scoring_system/logs/keepalive.log
# =============================================================================

set -u   # 未定义变量报错（不用 set -e，流程自行控制）

# ── 配置区（部署时按实际环境修改；均可用环境变量覆盖）──────────────────
APP_DIR="${KEEPALIVE_APP_DIR:-/home/linaro/code/intelligent_scoring_system}"  # 项目绝对路径
PYTHON="${KEEPALIVE_PYTHON:-python3}"                                        # Python 解释器
APP_ENTRY="pipeline.py"                                                      # 业务入口

PID_FILE="${KEEPALIVE_PID_FILE:-$APP_DIR/logs/basketball_scoring.pid}"                # 业务进程 pidfile
LOCK_FILE="${KEEPALIVE_LOCK_FILE:-$APP_DIR/logs/basketball_scoring_keepalive.lock}"   # flock 锁文件
LOG_FILE="$APP_DIR/logs/keepalive.log"                                       # 守护脚本日志
STDOUT_LOG="$APP_DIR/logs/keepalive_stdout.log"                              # 业务进程 stdout/stderr

HEALTH_RETRY="${KEEPALIVE_HEALTH_RETRY:-2}"         # 健康检查连续失败判定次数
HEALTH_INTERVAL="${KEEPALIVE_HEALTH_INTERVAL:-2}"   # 健康检查重试间隔（秒）
HEALTH_TIMEOUT="${KEEPALIVE_HEALTH_TIMEOUT:-5}"     # 单次 HTTP 请求超时（秒）
POST_START_WAIT="${KEEPALIVE_POST_START_WAIT:-15}"  # 启动后等待模型/HTTP 就绪（秒）
KILL_GRACE="${KEEPALIVE_KILL_GRACE:-10}"            # SIGTERM 后优雅退出宽限（秒）
# ────────────────────────────────────────────────────────────────────────────

mkdir -p "$APP_DIR/logs" 2>/dev/null

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

# flock 防并发：已有守护实例在跑则本轮直接退出（避免多实例重复拉起）
if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "检测到另一守护实例正在执行，本轮跳过"
        exit 0
    fi
else
    log "警告：flock 不可用，跳过并发锁（建议安装 util-linux）"
fi

# 从 config/http.yaml 动态解析探活地址，与业务配置保持一致
HTTP_HOST=$(grep -E '^HTTP_HOST:' "$APP_DIR/config/http.yaml" 2>/dev/null | awk '{print $2}' | tr -d '"' | tr -d "'")
HTTP_PORT=$(grep -E '^HTTP_PORT:' "$APP_DIR/config/http.yaml" 2>/dev/null | awk '{print $2}' | tr -d '"' | tr -d "'")
HTTP_HOST="${HTTP_HOST:-127.0.0.1}"
HTTP_PORT="${HTTP_PORT:-8899}"
[ "$HTTP_HOST" = "0.0.0.0" ] && HTTP_HOST="127.0.0.1"
HEALTH_URL="http://${HTTP_HOST}:${HTTP_PORT}/health"

# HTTP GET 封装（curl 优先，wget 兜底）
http_get() {
    if command -v curl >/dev/null 2>&1; then
        curl -s -m "$HEALTH_TIMEOUT" "$1" 2>/dev/null
    elif command -v wget >/dev/null 2>&1; then
        wget -q -T "$HEALTH_TIMEOUT" -O - "$1" 2>/dev/null
    else
        return 1
    fi
}

# 进程是否存活（pidfile 存在且 kill -0 通过）
is_alive() {
    local pid
    [ -f "$PID_FILE" ] || return 1
    pid=$(cat "$PID_FILE" 2>/dev/null)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# 查找实际运行的 pipeline.py 进程 PID（无 pidfile / pidfile 丢失时兜底）
find_running_pid() {
    pgrep -f "pipeline\.py" 2>/dev/null | head -1
}

# 业务探活：连续 HEALTH_RETRY 次 GET /health，返回含 "code" 字段即健康
health_ok() {
    local i resp
    for i in $(seq 1 "$HEALTH_RETRY"); do
        resp=$(http_get "$HEALTH_URL")
        if echo "$resp" | grep -q '"code"'; then
            return 0
        fi
        [ "$i" -lt "$HEALTH_RETRY" ] && sleep "$HEALTH_INTERVAL"
    done
    return 1
}

# 启动业务进程：setsid 脱离进程组 + 写 pidfile + 等待就绪
start_app() {
    cd "$APP_DIR" || { log "错误：无法进入 $APP_DIR，启动中止"; return 1; }
    # 关键：setsid 创建新会话、脱离宝塔 cron 进程组，避免任务结束被连坐 SIGINT
    if command -v setsid >/dev/null 2>&1; then
        setsid nohup "$PYTHON" "$APP_ENTRY" >> "$STDOUT_LOG" 2>&1 < /dev/null &
    else
        nohup "$PYTHON" "$APP_ENTRY" >> "$STDOUT_LOG" 2>&1 < /dev/null &
    fi
    local newpid=$!
    echo "$newpid" > "$PID_FILE"
    log "已启动 pipeline.py（PID=$newpid），stdout/stderr -> $STDOUT_LOG"
    sleep "$POST_START_WAIT"
    if health_ok; then
        log "启动后探活通过（$HEALTH_URL）"
    else
        log "启动后探活暂未通过，等待下次巡检确认（模型可能仍在加载）"
    fi
    return 0
}

# 停止业务进程：SIGTERM 优雅优先，超时 SIGKILL 兜底
stop_app() {
    local pid="$1"
    log "发送 SIGTERM 优雅停止 PID=$pid ..."
    kill "$pid" 2>/dev/null
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$KILL_GRACE" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        log "SIGTERM ${KILL_GRACE}s 内未退出，SIGKILL 兜底 PID=$pid"
        kill -9 "$pid" 2>/dev/null
        sleep 1
    else
        log "进程 PID=$pid 已优雅退出"
    fi
    rm -f "$PID_FILE"
}

# ── 主流程 ────────────────────────────────────────────────────────────────

if is_alive; then
    pid=$(cat "$PID_FILE")
    if health_ok; then
        # 进程存活且业务健康：无需动作（健康时不刷日志）
        exit 0
    fi
    # 进程在但 /health 连续失败 → 判定业务卡死，重启
    log "进程 PID=$pid 存活但探活连续 ${HEALTH_RETRY} 次失败（$HEALTH_URL），触发重启"
    stop_app "$pid"
    start_app
else
    # pidfile 缺失或进程已死
    running=$(find_running_pid)
    if [ -n "$running" ]; then
        # 进程在跑但 pidfile 丢失（被误删/异常退出未清理）→ 仅重建 pidfile
        log "检测到 pipeline.py 已在运行（PID=$running）但无 pidfile，重建 pidfile"
        echo "$running" > "$PID_FILE"
        exit 0
    fi
    # 无进程 → 启动
    log "pipeline.py 未运行，启动 ..."
    start_app
fi

exit 0
