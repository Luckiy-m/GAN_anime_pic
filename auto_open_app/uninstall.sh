#!/data/data/com.termux/files/usr/bin/bash
# ============================================================
# uninstall.sh - 卸载定时任务，清理所有配置
# 用法: bash uninstall.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo "  卸载自动打开应用定时任务"
echo "=============================================="

# 1. 移除 termux-job-scheduler 任务
if command -v termux-job-scheduler &>/dev/null; then
    echo "取消 termux-job-scheduler 任务..."
    termux-job-scheduler --cancel --job-id 1 2>/dev/null || true
    echo -e "${GREEN}  已取消${NC}"
fi

# 2. 移除 cron 任务
if command -v crontab &>/dev/null; then
    echo "移除 cron 定时任务..."
    local tmp_cron="/tmp/crontab_uninstall_$$"
    crontab -l 2>/dev/null | grep -v "auto_open_app.sh" > "${tmp_cron}" 2>/dev/null || true
    if [[ -s "${tmp_cron}" ]]; then
        crontab "${tmp_cron}"
    else
        crontab -r 2>/dev/null || true
    fi
    rm -f "${tmp_cron}"
    echo -e "${GREEN}  已移除${NC}"
fi

# 3. 停止 crond 服务
if pgrep -x crond >/dev/null 2>&1; then
    echo "停止 crond 服务..."
    pkill crond 2>/dev/null || true
    echo -e "${GREEN}  已停止${NC}"
fi

# 4. 移除开机自启动脚本
local boot_script="${HOME}/.termux/boot/auto_open_app_boot"
if [[ -f "${boot_script}" ]]; then
    echo "移除开机自启动脚本..."
    rm -f "${boot_script}"
    echo -e "${GREEN}  已移除${NC}"
fi

# 5. 清理临时文件
rm -f "${SCRIPT_DIR}/job_runner.sh" 2>/dev/null || true
rm -f "${SCRIPT_DIR}/job_output.log" 2>/dev/null || true

echo ""
echo "=============================================="
echo "  卸载完成！"
echo "=============================================="
echo ""
echo "以下文件未被删除，如需彻底清除请手动删除："
echo "  脚本目录: ${SCRIPT_DIR}"
echo "  日志文件: ${HOME}/.auto_open_app.log"
echo ""
echo "手动删除命令:"
echo "  rm -rf ${SCRIPT_DIR}"
echo "  rm -f ${HOME}/.auto_open_app.log"