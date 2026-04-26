#!/usr/bin/env bash
# 跨区域 RTT + 带宽 24h（或可配置时长）周期性探测
# 依赖: ping, iperf3, awk
#
# 部署步骤（概要）:
#   1. 两台 VM安装: sudo apt-get update && sudo apt-get install -y iperf3
#   2. 在对端 VM 上常驻: iperf3 -s -p 5201   （或用 systemd 托管）
#   3. 本机: cp inter_region_probe.conf.example inter_region_probe.conf && 编辑
#   4. chmod +x inter_region_probe.sh
#   5. 前台: ./inter_region_probe.sh --config inter_region_probe.conf
#      后台: nohup ./inter_region_probe.sh --config inter_region_probe.conf >> probe.out 2>&1 &
#
# 防火墙/VPC: 需放行对端 TCP 5201（或你配置的 IPERF_PORT），ICMP 若被禁则 RTT 会失败。
# 同目录需有 probe_audit.sh（CSV 转义 + provenance 审计文件）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/inter_region_probe.conf"

usage() {
  echo "用法: $0 [--config PATH] [--help]"
  echo "  --config FILE  配置文件（默认: ${SCRIPT_DIR}/inter_region_probe.conf）"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "缺少配置文件: $CONFIG_FILE" >&2
  echo "请复制 inter_region_probe.conf.example 并修改。" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${PEER_HOST:?请在配置中设置 PEER_HOST}"
: "${IPERF_PORT:=5201}"
: "${IPERF_SECONDS:=10}"
: "${PING_COUNT:=10}"
: "${INTERVAL_SEC:=300}"
: "${DURATION:=24h}"
: "${LOG_DIR:=/var/log/inter_region_probe}"
: "${REGION_LOCAL:=local}"
: "${REGION_PEER:=peer}"
: "${CONTINUE_ON_ERROR:=1}"

duration_to_seconds() {
  local d="$1"
  if [[ "$d" =~ ^[0-9]+$ ]]; then
    echo "$d"
    return
  fi
  local num unit
  num="${d//[^0-9]/}"
  unit="${d//[^a-zA-Z]/}"
  unit="${unit,,}"
  case "$unit" in
    h|hr|hour|hours) echo $(( num * 3600 )) ;;
    m|min|minute|minutes) echo $(( num * 60 )) ;;
    s|sec|second|seconds) echo "$num" ;;
    d|day|days) echo $(( num * 86400 )) ;;
    *) echo "无法解析 DURATION: $d（示例: 24h, 86400）" >&2; exit 1 ;;
  esac
}

TOTAL_SEC="$(duration_to_seconds "$DURATION")"
START_EPOCH="$(date +%s)"
END_EPOCH=$(( START_EPOCH + TOTAL_SEC ))

mkdir -p "$LOG_DIR"
_run_id="$(date -u +%Y%m%dT%H%M%SZ)_${REGION_LOCAL}_to_${REGION_PEER}"
RTT_CSV="${LOG_DIR}/rtt_${_run_id}.csv"
BW_CSV="${LOG_DIR}/bandwidth_${_run_id}.csv"
PROVENANCE_FILE="${LOG_DIR}/provenance_${_run_id}.txt"
SCRIPT_SELF="${BASH_SOURCE[0]}"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/probe_audit.sh"
write_probe_provenance "$PROVENANCE_FILE" "$SCRIPT_SELF" "$CONFIG_FILE"

{
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "peer=${PEER_HOST}"
  echo "region_local=${REGION_LOCAL} region_peer=${REGION_PEER}"
  echo "interval_sec=${INTERVAL_SEC} total_sec=${TOTAL_SEC}"
  echo "ping_cmd=ping -n -c ${PING_COUNT} -W 2 ${PEER_HOST}"
  echo "iperf_cmd=iperf3 -c ${PEER_HOST} -p ${IPERF_PORT} -t ${IPERF_SECONDS} -f m --bidir"
  echo "rtt_csv=${RTT_CSV}"
  echo "bandwidth_csv=${BW_CSV}"
  echo "provenance=${PROVENANCE_FILE}"
} > "${LOG_DIR}/run_${_run_id}.meta"

echo "timestamp_utc,region_local,region_peer,peer_host,ping_count,rtt_min_ms,rtt_avg_ms,rtt_max_ms,rtt_mdev_ms,packet_loss_pct,ping_ok,rtt_notes,ping_stdout_stderr" > "$RTT_CSV"
echo "timestamp_utc,region_local,region_peer,peer_host,iperf_port,iperf_seconds,bw_out_mbits_per_sec,bw_in_mbits_per_sec,iperf_ok,bw_notes,iperf_stdout_stderr" > "$BW_CSV"

log_rtt_row() {
  local ts="$1" rmin="$2" ravg="$3" rmax="$4" rmdev="$5" loss="$6" pok="$7" n="$8" raw="$9"
  n="${n//,/;}"
  {
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
      "$ts" "$REGION_LOCAL" "$REGION_PEER" "$PEER_HOST" "$PING_COUNT" \
      "$rmin" "$ravg" "$rmax" "$rmdev" "$loss" "$pok" "$n"
    csv_field "$raw"
    printf '\n'
  } >> "$RTT_CSV"
}

log_bw_row() {
  local ts="$1" bw_out="$2" bw_in="$3" iok="$4" n="$5" raw="$6"
  n="${n//,/;}"
  {
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
      "$ts" "$REGION_LOCAL" "$REGION_PEER" "$PEER_HOST" "$IPERF_PORT" "$IPERF_SECONDS" \
      "$bw_out" "$bw_in" "$iok" "$n"
    csv_field "$raw"
    printf '\n'
  } >> "$BW_CSV"
}

parse_ping_stats() {
  local out="$1"
  local loss="NA"
  local loss_line
  loss_line="$(echo "$out" | grep -F 'packet loss' | tail -1 || true)"
  if [[ -n "$loss_line" ]]; then
    loss="$(echo "$loss_line" | sed -n 's/.* \([0-9.]*\)% packet loss.*/\1/p')"
    [[ -z "$loss" ]] && loss="NA"
  fi

  local line
  line="$(echo "$out" | grep -E 'rtt min/avg/max' || true)"
  if [[ -z "$line" ]]; then
    echo "NA NA NA NA $loss"
    return
  fi
  local vals
  vals="$(echo "$line" | sed -e 's/.*= *//' -e 's/ ms$//' | tr '/' ' ')"
  echo "$vals $loss"
}

# --bidir 下客户端：一条流 sender 非零为本地→对端，另一条流 receiver 非零为对端→本地
parse_iperf_bidir_mbps() {
  local out="$1"
  echo "$out" | awk '
    function to_mbps(val, unit,   r) {
      r = val + 0
      if (unit == "Kbits/sec") return r / 1000
      if (unit == "Mbits/sec") return r
      if (unit == "Gbits/sec") return r * 1000
      return -1
    }
    function rate_from_line(   i, mb) {
      for (i = 1; i <= NF; i++) {
        if (($i == "Kbits/sec" || $i == "Mbits/sec" || $i == "Gbits/sec") && i > 1) {
          mb = to_mbps($(i - 1), $i)
          if (mb >= 0) return mb
        }
      }
      return -1
    }
    !/\[SUM\]/ && /sender/ && /bits\/sec/ {
      mb = rate_from_line()
      if (mb > 0) bw_out = mb
    }
    !/\[SUM\]/ && /receiver/ && /bits\/sec/ {
      mb = rate_from_line()
      if (mb > 0) bw_in = mb
    }
    END {
      if (bw_out == "" && bw_in == "") { print "NA NA"; exit }
      if (bw_out == "") bw_out = "NA"
      if (bw_in == "") bw_in = "NA"
      if (bw_out != "NA" && bw_in != "NA")
        printf "%.6f %.6f\n", bw_out, bw_in
      else if (bw_out != "NA")
        printf "%.6f NA\n", bw_out
      else
        printf "NA %.6f\n", bw_in
    }'
}

probe_once() {
  local ts rtt_notes="" bw_notes="" ping_ok=0 iperf_ok=0
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local pout pstat
  set +e
  pout="$(ping -n -c "$PING_COUNT" -W 2 "$PEER_HOST" 2>&1)"
  pstat=$?
  set -e

  local rtt_min="NA" rtt_avg="NA" rtt_max="NA" rtt_mdev="NA" loss="NA"
  read -r rtt_min rtt_avg rtt_max rtt_mdev loss <<< "$(parse_ping_stats "$pout")"
  if [[ $pstat -eq 0 ]]; then
    ping_ok=1
  else
    rtt_notes="ping_failed"
  fi
  log_rtt_row "$ts" "$rtt_min" "$rtt_avg" "$rtt_max" "$rtt_mdev" "$loss" "$ping_ok" "$rtt_notes" "$pout"

  local iout istat bw_out="NA" bw_in="NA"
  set +e
  iout="$(iperf3 -c "$PEER_HOST" -p "$IPERF_PORT" -t "$IPERF_SECONDS" -f m --bidir 2>&1)"
  istat=$?
  set -e

  if [[ $istat -eq 0 ]]; then
    iperf_ok=1
    read -r bw_out bw_in <<< "$(parse_iperf_bidir_mbps "$iout")"
    [[ -z "$bw_out" ]] && bw_out="NA"
    [[ -z "$bw_in" ]] && bw_in="NA"
    if [[ "$bw_out" == "NA" && "$bw_in" == "NA" ]]; then
      iperf_ok=0
      bw_notes="iperf_parse_failed"
    elif [[ "$bw_out" == "NA" || "$bw_in" == "NA" ]]; then
      bw_notes="iperf_bidir_partial_parse"
    fi
  else
    bw_notes="iperf_failed"
  fi
  log_bw_row "$ts" "$bw_out" "$bw_in" "$iperf_ok" "$bw_notes" "$iout"

  echo "[${ts}] RTT avg=${rtt_avg} ms loss=${loss}% BW_out=${bw_out} BW_in=${bw_in} Mbit/s (ping_ok=${ping_ok} iperf_ok=${iperf_ok})"
}

cleanup() {
  append_probe_provenance_end "$PROVENANCE_FILE"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) 探测结束，RTT: $RTT_CSV  带宽: $BW_CSV  审计: $PROVENANCE_FILE" >&2
}
trap cleanup EXIT

echo "开始探测 ${TOTAL_SEC}s，间隔 ${INTERVAL_SEC}s，对端 ${PEER_HOST}" >&2
echo "  RTT 日志: $RTT_CSV" >&2
echo "  带宽日志: $BW_CSV" >&2
echo "  来源审计: $PROVENANCE_FILE" >&2

while (( $(date +%s) < END_EPOCH )); do
  set +e
  probe_once
  rc=$?
  set -e
  if [[ $rc -ne 0 && "$CONTINUE_ON_ERROR" != "1" ]]; then
    echo "probe_once 失败且 CONTINUE_ON_ERROR=0，退出" >&2
    exit "$rc"
  fi

  now="$(date +%s)"
  (( now >= END_EPOCH )) && break

  sleep_sec="$INTERVAL_SEC"
  if (( now + sleep_sec > END_EPOCH )); then
    sleep_sec=$(( END_EPOCH - now ))
  fi
  if (( sleep_sec > 0 )); then
    sleep "$sleep_sec"
  fi
done

exit 0
