#!/data/data/com.termux/files/usr/bin/bash
# ============================================================
# setup_schedule.sh - 一键配置定时任务
# 支持两种调度方式:
#   1. termux-job-scheduler (推荐，Android 原生定时任务)
#   2. cron (termux-cron，需要安装 cronie 包)
# 用法: bash setup_schedule.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.conf"
SCRIPT_FILE="${SCRIPT_DIR}/auto_open_app.sh"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ---------- 检查 Termux 环境 ----------
check_termux() {
    if [[ ! -d /data/data/com.termux/files/usr ]]; then
        echo -e "${RED}[ERROR] 请在 Termux 环境中运行此脚本！${NC}"
        echo ""
        echo "安装 Termux 步骤："
        echo "  1. 从 F-Droid 下载 Termux: https://f-droid.org/packages/com.termux/"
        echo "  2. 安装后打开 Termux，执行: pkg update && pkg upgrade"
        echo "  3. 将此脚本目录复制到手机存储中"
        echo "  4. 在 Termux 中 cd 到脚本目录，执行: bash setup_schedule.sh"
        exit 1
    fi
}

# ---------- 加载配置 ----------
load_config() {
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo -e "${RED}[ERROR] 配置文件不存在: ${CONFIG_FILE}${NC}"
        exit 1
    fi
    source "${CONFIG_FILE}"
    SCHEDULE_TIME="${SCHEDULE_TIME:-09:00}"
}

# ---------- 安装依赖 ----------
install_deps() {
    echo -e "${YELLOW}[1/4] 安装必要依赖...${NC}"
    pkg update -y -q 2>/dev/null || true
    # 安装 termux-api（提供 termux-job-scheduler 等命令）
    if ! command -v termux-job-scheduler &>/dev/null; then
        echo "  安装 termux-api..."
        pkg install termux-api -y -q
    fi
    # 安装 jq（用于 JSON 处理）
    if ! command -v jq &>/dev/null; then
        echo "  安装 jq..."
        pkg install jq -y -q
    fi
    echo -e "${GREEN}  依赖安装完成${NC}"
}

# ---------- 赋予脚本可执行权限 ----------
set_permissions() {
    echo -e "${YELLOW}[2/4] 设置脚本权限...${NC}"
    chmod +x "${SCRIPT_FILE}"
    echo -e "${GREEN}  权限设置完成${NC}"
}

# ---------- 配置 termux-job-scheduler ----------
setup_job_scheduler() {
    echo -e "${YELLOW}[3/4] 配置 termux-job-scheduler...${NC}"

    # 解析时间
    local hour="${SCHEDULE_TIME%%:*}"
    local minute="${SCHEDULE_TIME##*:}"
    # 去除前导零
    hour=$((10#${hour}))
    minute=$((10#${minute}))

    # 取消已有同名任务
    termux-job-scheduler --cancel "auto-xinrenxinshi" 2>/dev/null || true

    # 创建定时任务
    # termux-job-scheduler 使用 --period-ms 参数（毫秒）
    # 设为 86400000ms = 24小时，首次触发时间通过 --trigger-at 指定
    local job_script="${SCRIPT_DIR}/auto_open_app.sh"
    local job_log="${SCRIPT_DIR}/job_output.log"

    cat > "${SCRIPT_DIR}/job_runner.sh" << 'JOB_EOF'
#!/data/data/com.termux/files/usr/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/auto_open_app.sh" >> "${SCRIPT_DIR}/job_output.log" 2>&1
JOB_EOF
    chmod +x "${SCRIPT_DIR}/job_runner.sh"

    termux-job-scheduler \
        --script "${SCRIPT_DIR}/job_runner.sh" \
        --job-id 1 \
        --period-ms 86400000 \
        --network none \
        --persisted true 2>/dev/null

    echo -e "${GREEN}  termux-job-scheduler 配置完成${NC}"
    echo -e "${YELLOW}  注意: termux-job-scheduler 从当前时间开始每24小时执行一次，"
    echo "  首次执行时间无法精确设定到指定时刻。"
    echo "  如需精确时间控制，请使用 cron 方案。${NC}"
}

# ---------- 配置 cron ----------
setup_cron() {
    echo -e "${YELLOW}[3/4] 配置 cron 定时任务...${NC}"

    # 安装 cronie
    if ! command -v crond &>/dev/null; then
        echo "  安装 cronie..."
        pkg install cronie -y -q
    fi

    # 解析时间
    local hour="${SCHEDULE_TIME%%:*}"
    local minute="${SCHEDULE_TIME##*:}"
    # 去除前导零
    hour=$((10#${hour}))
    minute=$((10#${minute}))

    # 创建 cron 条目: 每天 HH:MM 执行
    local cron_entry="${minute} ${hour} * * * bash ${SCRIPT_FILE} >> ${HOME}/.auto_open_app.log 2>&1"

    # 写入 crontab（保留已有条目，仅添加新条目）
    local tmp_cron="/tmp/crontab_$$"
    crontab -l 2>/dev/null | grep -v "auto_open_app.sh" > "${tmp_cron}" || true
    echo "${cron_entry}" >> "${tmp_cron}"
    crontab "${tmp_cron}"
    rm -f "${tmp_cron}"

    # 启动 crond 服务
    if ! pgrep -x crond >/dev/null 2>&1; then
        crond
        echo "  crond 服务已启动"
    else
        echo "  crond 服务已在运行中"
    fi

    echo -e "${GREEN}  cron 定时任务配置完成，每天 ${SCHEDULE_TIME} 执行${NC}"
}

# ---------- 配置 Termux:Boot 开机自启 ----------
setup_boot() {
    echo -e "${YELLOW}[4/4] 配置开机自启动...${NC}"

    local boot_dir="${HOME}/.termux/boot"
    mkdir -p "${boot_dir}"

    cat > "${boot_dir}/auto_open_app_boot" << BOOT_EOF
#!/data/data/com.termux/files/usr/bin/bash
# Auto Open App - 开机自启动
# 确保 crond 在开机后运行
if command -v crond &>/dev/null; then
    pgrep -x crond >/dev/null 2>&1 || crond
fi
BOOT_EOF
    chmod +x "${boot_dir}/auto_open_app_boot"

    echo -e "${GREEN}  开机自启动配置完成${NC}"
    echo -e "${YELLOW}  注意: 需要安装 Termux:Boot 应用才能生效${NC}"
    echo "  F-Droid: https://f-droid.org/packages/com.termux.boot/"
}

# ---------- 选择调度方式 ----------
choose_scheduler() {
    echo ""
    echo "=============================================="
    echo "  请选择定时调度方式:"
    echo "=============================================="
    echo "  1) termux-job-scheduler (推荐)"
    echo "     - Android 原生定时任务，更省电"
    echo "     - 首次执行时间不可精确控制"
    echo "     - 执行间隔固定为 24 小时"
    echo ""
    echo "  2) cron (精确时间控制)"
    echo "     - 精确到分钟，每天 ${SCHEDULE_TIME} 准时执行"
    echo "     - 需要后台运行 crond 进程"
    echo "     - 需要安装 Termux:Boot 保持开机自启"
    echo ""
    echo "  3) 两种都配置"
    echo "=============================================="

    read -r -p "请输入选择 [1-3] (默认 2): " choice
    choice="${choice:-2}"

    case "${choice}" in
        1)
            setup_job_scheduler
            ;;
        2)
            setup_cron
            setup_boot
            ;;
        3)
            setup_job_scheduler
            setup_cron
            setup_boot
            ;;
        *)
            echo -e "${RED}无效选择，使用默认 cron 方案${NC}"
            setup_cron
            setup_boot
            ;;
    esac
}

# ---------- 测试运行 ----------
test_run() {
    echo ""
    echo -e "${YELLOW}=============================================="
    echo "  测试运行..."
    echo "==============================================${NC}"
    bash "${SCRIPT_FILE}"
    echo ""
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}测试通过！${NC}"
    else
        echo -e "${RED}测试失败，请检查日志: ${HOME}/.auto_open_app.log${NC}"
    fi
}

# ---------- 显示状态 ----------
show_status() {
    echo ""
    echo "=============================================="
    echo "  配置完成！当前状态:"
    echo "=============================================="
    echo "  目标应用: ${PACKAGE_NAME:-com.client.xrxs.com.xrxsapp} (薪人薪事)"
    echo "  定时时间: 每天 ${SCHEDULE_TIME:-09:00}"
    echo "  脚本路径: ${SCRIPT_FILE}"
    echo "  日志文件: ${HOME}/.auto_open_app.log"
    echo ""
    echo "  常用命令:"
    echo "    查看日志:  cat ${HOME}/.auto_open_app.log"
    echo "    手动执行:  bash ${SCRIPT_FILE}"
    echo "    查看定时任务: crontab -l"
    echo "    卸载定时任务: bash ${SCRIPT_DIR}/uninstall.sh"
    echo "=============================================="
}

# ---------- 主流程 ----------
main() {
    echo "=============================================="
    echo "  自动打开应用 - 定时任务配置向导"
    echo "=============================================="

    check_termux
    load_config
    install_deps
    set_permissions
    choose_scheduler
    test_run
    show_status
}

main