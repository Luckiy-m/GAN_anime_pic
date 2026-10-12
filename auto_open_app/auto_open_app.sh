#!/data/data/com.termux/files/usr/bin/bash
# ============================================================
# auto_open_app.sh - 自动打开指定 Android 应用
# 依赖: Android 系统 + Termux
# 用法: bash auto_open_app.sh [配置路径]
# ============================================================

set -euo pipefail

# ---------- 加载配置 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-${SCRIPT_DIR}/config.conf}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[ERROR] 配置文件不存在: ${CONFIG_FILE}"
    echo "请先复制并编辑 config.conf，然后重新运行。"
    exit 1
fi
# shellcheck source=config.conf
source "${CONFIG_FILE}"

# ---------- 默认值 ----------
PACKAGE_NAME="${PACKAGE_NAME:-com.client.xrxs.com.xrxsapp}"
LAUNCHER_ACTIVITY="${LAUNCHER_ACTIVITY:-}"
LOG_FILE="${LOG_FILE:-${HOME}/.auto_open_app.log}"

# ---------- 日志函数 ----------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "${msg}"
    echo "${msg}" >> "${LOG_FILE}"
}

# ---------- 日志轮转 ----------
rotate_log() {
    local max_lines="${LOG_MAX_LINES:-500}"
    if [[ -f "${LOG_FILE}" ]]; then
        local line_count
        line_count=$(wc -l < "${LOG_FILE}" 2>/dev/null || echo 0)
        if (( line_count > max_lines )); then
            tail -n "${max_lines}" "${LOG_FILE}" > "${LOG_FILE}.tmp" && \
                mv "${LOG_FILE}.tmp" "${LOG_FILE}"
        fi
    fi
}

# ---------- 检查应用是否已安装 ----------
check_app_installed() {
    if pm list packages 2>/dev/null | grep -q "^package:${PACKAGE_NAME}$"; then
        return 0
    fi
    # 也尝试通过 dumpsys 检查
    if dumpsys package "${PACKAGE_NAME}" 2>/dev/null | grep -q "${PACKAGE_NAME}"; then
        return 0
    fi
    return 1
}

# ---------- 获取启动 Activity ----------
get_launcher_activity() {
    # 从包信息中自动提取主 Activity
    local activity
    activity=$(dumpsys package "${PACKAGE_NAME}" 2>/dev/null | \
        grep -A5 "android.intent.action.MAIN" | \
        grep -oP "${PACKAGE_NAME}/\K[^ ]+" | head -1)
    echo "${activity}"
}

# ---------- 启动应用 ----------
launch_app() {
    log "正在启动应用: ${PACKAGE_NAME}"

    # 尝试方法1: 查询并启动主 Activity
    local activity="${LAUNCHER_ACTIVITY}"
    if [[ -z "${activity}" ]]; then
        activity=$(get_launcher_activity)
        log "自动检测到启动 Activity: ${activity}"
    fi

    if [[ -n "${activity}" ]]; then
        am start -n "${PACKAGE_NAME}/${activity}" >/dev/null 2>&1 && return 0
    fi

    # 尝试方法2: 使用 monkey 命令启动
    log "方法1失败，尝试使用 monkey 命令..."
    monkey -p "${PACKAGE_NAME}" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 && return 0

    # 尝试方法3: 使用 am start 简化方式
    log "方法2失败，尝试使用 am start 简化方式..."
    am start -a android.intent.action.MAIN \
        -c android.intent.category.LAUNCHER \
        "${PACKAGE_NAME}" >/dev/null 2>&1 && return 0

    return 1
}

# ---------- 主流程 ----------
main() {
    rotate_log
    log "========== 开始执行 =========="

    if ! check_app_installed; then
        log "[ERROR] 应用 ${PACKAGE_NAME} 未安装！"
        log "请确认已安装薪人薪事 APP，或修改 config.conf 中的 PACKAGE_NAME。"
        exit 1
    fi

    if launch_app; then
        log "[SUCCESS] 应用 ${PACKAGE_NAME} 启动成功"
    else
        log "[ERROR] 应用启动失败，请检查应用权限或手动启动一次。"
        exit 1
    fi

    log "========== 执行完毕 =========="
}

main