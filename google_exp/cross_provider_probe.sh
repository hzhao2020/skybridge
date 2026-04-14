#!/usr/bin/env bash
# 跨云服务商：两台 VM 之间 RTT（ping）与 TCP 带宽（iperf3 上行 + 下行）
# 部署在本脚本所在的一台 VM；对端运行 iperf3 -s。
# 上行：本机 -> 对端（iperf3 默认，本机 sender）
# 下行：对端 -> 本机（iperf3 -R，本机 receiver）
#
# 依赖：ping、iperf3、awk；同目录 probe_audit.sh（审计与 CSV 转义）。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/cross_provider_probe.conf"
SCRIPT_SELF="${BASH_SOURCE[0]}"

usage() {
  echo "用法: $0 [--config PATH] [--help]"
  echo "  --config FILE  配置文件（默认: ${SCRIPT_DIR}/cross_provider_probe.conf）"
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
  echo "请复制 cross_provider_probe.conf.example 并修改。" >&2
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
: "${LOG_DIR:=${HOME}/logs_cross_provider_probe}"
: "${PROVIDER_LOCAL:=local_provider}"
: "${PROVIDER_PEER:=peer_provider}"
: "${REGION_LOCAL:=NA}"
: "${REGION_PEER:=NA}"
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
_run_id="$(date -u +%Y%m%dT%H%M%SZ)_${PROVIDER_LOCAL}_to_${PROVIDER_PEER}"
RTT_CSV="${LOG_DIR}/rtt_${_run_id}.csv"
BW_CSV="${LOG_DIR}/bandwidth_${_run_id}.csv"
META_FILE="${LOG_DIR}/run_${_run_id}.meta"
PROVENANCE_FILE="${LOG_DIR}/provenance_${_run_id}.txt"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/probe_audit.sh"
write_probe_provenance "$PROVENANCE_FILE" "$SCRIPT_SELF" "$CONFIG_FILE"

{
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "probe_role=client_vm (iperf3/ping initiator)"
  echo "peer_host=${PEER_HOST} (iperf3 server / ping target)"
  echo "provider_local=${PROVIDER_LOCAL} provider_peer=${PROVIDER_PEER}"
  echo "region_local=${REGION_LOCAL} region_peer=${REGION_PEER}"
  echo "interval_sec=${INTERVAL_SEC} total_sec=${TOTAL_SEC}"
  echo "iperf_upload_cmd=iperf3 -c ${PEER_HOST} -p ${IPERF_PORT} -t ${IPERF_SECONDS} -f m"
  echo "iperf_download_cmd=iperf3 -c ${PEER_HOST} -p ${IPERF_PORT} -t ${IPERF_SECONDS} -f m -R"
  echo "ping_cmd=ping -n -c ${PING_COUNT} -W 2 ${PEER_HOST}"
  echo "rtt_csv=${RTT_CSV}"
  echo "bandwidth_csv=${BW_CSV}"
  echo "provenance=${PROVENANCE_FILE}"
} > "$META_FILE"

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

parse_iperf_mbps_role() {
  local out="$1"
  local role="$2"
  local line
  line="$(echo "$out" | grep -F "$role" | tail -1 || true)"
  if [[ -z "$line" ]]; then
    echo "NA"
    return
  fi
  echo "$line" | awk -v role="$role" '
    $0 ~ role {
      for (i = 1; i <= NF; i++) {
        if ($i == "Kbits/sec" && i > 1) { printf "%.6f\n", $(i-1) / 1000; exit }
        if ($i == "Mbits/sec" && i > 1) { print $(i-1); exit }
        if ($i == "Gbits/sec" && i > 1) { printf "%.6f\n", $(i-1) * 1000; exit }
      }
      exit
    }'
}

log_rtt_row() {
  local ts="$1" rmin="$2" ravg="$3" rmax="$4" rmdev="$5" loss="$6" pok="$7" n="$8" raw="$9"
  n="${n//,/;}"
  {
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
      "$ts" "$PROVIDER_LOCAL" "$PROVIDER_PEER" "$REGION_LOCAL" "$REGION_PEER" "$PEER_HOST" \
      "$PING_COUNT" "$rmin" "$ravg" "$rmax" "$rmdev" "$loss" "$pok" "$n"
    csv_field "$raw"
    printf '\n'
  } >> "$RTT_CSV"
}

log_bw_row() {
  local ts="$1" up="$2" down="$3" up_ok="$4" down_ok="$5" n="$6" raw_up="$7" raw_down="$8"
  n="${n//,/;}"
  {
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
      "$ts" "$PROVIDER_LOCAL" "$PROVIDER_PEER" "$REGION_LOCAL" "$REGION_PEER" "$PEER_HOST" \
      "$IPERF_PORT" "$IPERF_SECONDS" "$up" "$down" "$up_ok" "$down_ok" "$n"
    csv_field "$raw_up"
    printf ','
    csv_field "$raw_down"
    printf '\n'
  } >> "$BW_CSV"
}

echo "timestamp_utc,provider_local,provider_peer,region_local,region_peer,peer_host,ping_count,rtt_min_ms,rtt_avg_ms,rtt_max_ms,rtt_mdev_ms,packet_loss_pct,ping_ok,rtt_notes,ping_stdout_stderr" > "$RTT_CSV"
echo "timestamp_utc,provider_local,provider_peer,region_local,region_peer,peer_host,iperf_port,iperf_seconds,bw_upload_mbits_per_sec,bw_download_mbits_per_sec,iperf_upload_ok,iperf_download_ok,bw_notes,iperf_upload_stdout_stderr,iperf_download_stdout_stderr" > "$BW_CSV"

probe_once() {
  local ts rtt_notes="" bw_notes="" ping_ok=0 up_ok=0 down_ok=0
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

  local up_out up_stat bw_up="NA"
  set +e
  up_out="$(iperf3 -c "$PEER_HOST" -p "$IPERF_PORT" -t "$IPERF_SECONDS" -f m 2>&1)"
  up_stat=$?
  set -e
  if [[ $up_stat -eq 0 ]]; then
    up_ok=1
    bw_up="$(parse_iperf_mbps_role "$up_out" sender)"
    if [[ -z "$bw_up" || "$bw_up" == "NA" ]]; then
      bw_up="NA"
      up_ok=0
      bw_notes="iperf_upload_parse_failed"
    fi
  else
    bw_notes="iperf_upload_failed"
  fi

  local down_out down_stat bw_down="NA"
  set +e
  down_out="$(iperf3 -c "$PEER_HOST" -p "$IPERF_PORT" -t "$IPERF_SECONDS" -f m -R 2>&1)"
  down_stat=$?
  set -e
  if [[ $down_stat -eq 0 ]]; then
    down_ok=1
    bw_down="$(parse_iperf_mbps_role "$down_out" receiver)"
    if [[ -z "$bw_down" || "$bw_down" == "NA" ]]; then
      bw_down="NA"
      down_ok=0
      [[ -n "$bw_notes" ]] && bw_notes="${bw_notes};"
      bw_notes="${bw_notes}iperf_download_parse_failed"
    fi
  else
    [[ -n "$bw_notes" ]] && bw_notes="${bw_notes};"
    bw_notes="${bw_notes}iperf_download_failed"
  fi

  log_bw_row "$ts" "$bw_up" "$bw_down" "$up_ok" "$down_ok" "$bw_notes" "$up_out" "$down_out"
  echo "[${ts}] ${PROVIDER_LOCAL}/${REGION_LOCAL} -> ${PROVIDER_PEER}/${REGION_PEER} @${PEER_HOST} RTT avg=${rtt_avg} ms loss=${loss}% 上行=${bw_up} 下行=${bw_down} Mbit/s"
}

cleanup() {
  append_probe_provenance_end "$PROVENANCE_FILE"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) 探测结束。RTT: $RTT_CSV  带宽: $BW_CSV  审计: $PROVENANCE_FILE" >&2
}
trap cleanup EXIT

echo "跨云探测 ${TOTAL_SEC}s，间隔 ${INTERVAL_SEC}s，对端 ${PEER_HOST} (${PROVIDER_PEER} ${REGION_PEER})" >&2
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
