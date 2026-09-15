# pipeline.py 存活守护 —— 宝塔计划任务配置说明（单脚本版）

> 目标：只保留一条宝塔计划任务，每 2 分钟检查一次 `pipeline.py` 是否存活，未存活则自动拉起；异常自动重启，全程有日志、防多实例并发。

## 一、部署脚本（板子终端执行）

把 `deploy/keepalive.sh` 上传到板子项目目录后，在板子终端依次执行：

```bash
# 1. 确认脚本就位
ls -l /home/linaro/code/intelligent_scoring_system/deploy/keepalive.sh

# 2. 赋可执行权限
chmod +x /home/linaro/code/intelligent_scoring_system/deploy/keepalive.sh

# 3. 【重要】若脚本在 Windows 编辑后上传，先转 LF 换行，否则报 bad interpreter / \r 语法错
sed -i 's/\r$//' /home/linaro/code/intelligent_scoring_system/deploy/keepalive.sh

# 4. 确认脚本顶部 APP_DIR 与板端实际路径一致（默认 /home/linaro/code/intelligent_scoring_system），不一致就改这一行
```

## 二、手动验证一次（板子终端执行）

```bash
# 先确认当前没有残留进程
pkill -f pipeline.py 2>/dev/null; sleep 1

# 手动跑一次守护脚本（无副作用：进程没起会自动拉起）
sudo /bin/bash /home/linaro/code/intelligent_scoring_system/deploy/keepalive.sh

# 等 15~20 秒，看日志确认拉起 + 探活
tail -20 /home/linaro/code/intelligent_scoring_system/logs/keepalive.log

# 再确认端口在听
ss -tlnp 2>/dev/null | grep 8899
```

## 三、宝塔面板配置（只加这一条）

打开宝塔面板 → 左侧「计划任务」→ 点「添加任务」，任务类型选 **「Shell脚本」**：

| 项 | 值 |
|---|---|
| 任务名称 | 篮球评分-守护巡检 |
| 执行周期 | **每 2 分钟**（N 分钟，填 2） |
| 脚本内容 | `/bin/bash /home/linaro/code/intelligent_scoring_system/deploy/keepalive.sh` |

> ⚠️ **执行周期务必选「每 2 分钟」，不要选成秒级/每分钟**。上一版踩的坑就是周期配成了秒级，脚本被每秒触发、海量刷日志，且 pipeline.py 反复被拉起又被杀。

> 不再需要「开机启动」任务：开机后第一个巡检周期（最多 2 分钟）会自动把 `pipeline.py` 拉起。代价是开机后服务最长延迟约 2 分钟才就绪，可接受。

---

## 四、等效 crontab（不用宝塔时的替代方案）

板端直接 `crontab -e`，加一行：

```cron
*/2 * * * * /bin/bash /home/linaro/code/intelligent_scoring_system/deploy/keepalive.sh >> /dev/null 2>&1
```

---

## 五、守护逻辑（脚本做了什么）

```
宝塔每 2 分钟触发
        │
        ▼
  flock -n 加锁 ──失败──► 已有实例在跑，本轮跳过（防并发）
        │成功
        ▼
  pidfile 存活?
   ├─ 是 ──► GET /health 探活（连续2次、间隔2s、超时5s）
   │           ├─ 通过 ──► 静默退出
   │           └─ 失败 ──► SIGTERM 优雅停止 → 超时 SIGKILL → 重启
   └─ 否 ──► pgrep 找实际进程
               ├─ 找到 ──► 重建 pidfile（进程没丢，只是 pidfile 丢了）
               └─ 没找到 ──► setsid 启动 pipeline.py
```

**关键点：**

1. **`setsid` 脱离进程组（本版核心修复）**：用 `setsid nohup ... &` 启动，让 `pipeline.py` 进入独立会话、脱离宝塔 cron 进程组。否则宝塔 cron 任务结束时会清理整个进程组，把 nohup 起的 pipeline.py 连坐杀掉（业务日志表现为「收到退出信号 2」SIGINT），导致「启动→被杀→再启动」死循环。
2. **探活地址自动同步**：脚本从 `config/http.yaml` 读 `HTTP_HOST`/`HTTP_PORT` 拼 `/health`；`HTTP_HOST=0.0.0.0` 时回退 `127.0.0.1`。
3. **优雅重启**：先 `SIGTERM`（`pipeline.py` 有 handler，会保存录制、释放 NPU），宽限 `KILL_GRACE`(10s) 后才 `SIGKILL`。
4. **防并发双保险**：`flock` 防多实例撞车；`pidfile` 防重复拉起。
5. **健康时不刷日志**：健康时静默退出，只有「启动/重启/告警」才落日志。

---

## 六、验收验证

```bash
# 1. 看守护日志
tail -20 /home/linaro/code/intelligent_scoring_system/logs/keepalive.log

# 2. 看业务 stdout/stderr（pipeline.py 打印与崩溃堆栈）
tail -20 /home/linaro/code/intelligent_scoring_system/logs/keepalive_stdout.log

# 3. 手动探活（应返回 {"code":0,"msg":"ok"}）
curl -s -m 5 http://192.168.8.249:8899/health

# 4. 模拟崩溃 → 手动跑一次守护脚本，应立即拉起
kill $(cat /tmp/basketball_scoring.pid)
sleep 1
sudo /bin/bash /home/linaro/code/intelligent_scoring_system/deploy/keepalive.sh
```

---

## 七、可配置参数（脚本顶部，均支持环境变量覆盖）

| 参数 | 默认 | 说明 |
|---|---|---|
| `APP_DIR` | `/home/linaro/code/intelligent_scoring_system` | 项目绝对路径 |
| `PYTHON` | `python3` | Python 解释器 |
| `HEALTH_RETRY` | `2` | 探活连续失败判定次数 |
| `HEALTH_INTERVAL` | `2` | 探活重试间隔（秒） |
| `HEALTH_TIMEOUT` | `5` | 单次 HTTP 超时（秒） |
| `POST_START_WAIT` | `15` | 启动后等待就绪（秒） |
| `KILL_GRACE` | `10` | SIGTERM 优雅退出宽限（秒） |

---

## 八、常见问题

**Q1：开机后要等多久服务才起来？**
→ 最长约 2 分钟（等第一个巡检周期）。可接受就无需开机任务；若必须开机秒起，改用 systemd service 托管 pipeline.py（`Restart=always`），宝塔只负责巡检。

**Q2：`bad interpreter: /bin/bash^M`？**
→ 脚本是 CRLF 换行，执行 `sed -i 's/\r$//' .../deploy/keepalive.sh` 转 LF。

**Q3：探活一直失败但进程明明在？**
→ 检查 `config/http.yaml` 的 `HTTP_HOST` 是不是被改成了不可达地址；脚本照这个 IP 探活，`0.0.0.0` 回退 `127.0.0.1`，具体内网 IP 必须本机可达。

**Q4：日志又出现每秒一条「启动/跳过」？**
→ 说明宝塔任务周期又被配成了秒级，回宝塔把「执行周期」改成「每 2 分钟」。

**Q5：进程还是反复被 SIGINT 杀掉？**
→ 确认脚本里 `start_app` 用的是 `setsid nohup ... &`（本版已加）；若板端 `setsid` 不存在，脚本会自动回退普通 `nohup`（此时建议 `apt install util-linux` 补上 setsid）。
