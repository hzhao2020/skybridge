# 由探测脚本 source；提供 CSV 转义与可审计来源信息（便于第三方核验）
# shellcheck shell=bash

csv_field() {
  local s="${1-}"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

# 将环境与工具链快照写入文本，便于证明「数据由何环境、何版本工具测得」
# 参数: $1=输出路径 $2=主脚本绝对路径(用于 sha256)  $3=可选配置文件路径(整份写入快照)
write_probe_provenance() {
  local dest="$1"
  local main_script="$2"
  local cfg="${3:-}"
  {
    echo "schema_version=1"
    echo "recorded_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "local_hostname=$(hostname 2>/dev/null || echo NA)"
    echo "local_fqdn=$(hostname -f 2>/dev/null || echo NA)"
    echo "login_user=$(whoami 2>/dev/null || echo NA)"
    echo "working_dir=$(pwd -P 2>/dev/null || pwd)"
    echo "uname=$(uname -a 2>/dev/null || echo NA)"
    echo "ping_bin=$(command -v ping 2>/dev/null || echo NA)"
    echo "ping_version_line=$(ping -V 2>&1 | head -1 || true)"
    echo "iperf3_bin=$(command -v iperf3 2>/dev/null || echo NA)"
    echo "iperf3_version_line=$(iperf3 -v 2>&1 | head -1 || true)"
    echo "main_script_path=${main_script}"
    if command -v sha256sum >/dev/null 2>&1 && [[ -f "$main_script" ]]; then
      echo "main_script_sha256=$(sha256sum "$main_script" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1 && [[ -f "$main_script" ]]; then
      echo "main_script_sha256=$(shasum -a 256 "$main_script" | awk '{print $1}')"
    fi
    if [[ -n "$cfg" && -f "$cfg" ]]; then
      echo "---- config_snapshot_begin ----"
      cat "$cfg"
      echo "---- config_snapshot_end ----"
    fi
  } > "$dest"
}

append_probe_provenance_end() {
  local dest="$1"
  [[ -z "$dest" || ! -f "$dest" ]] && return 0
  {
    echo "recorded_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "$dest"
}

