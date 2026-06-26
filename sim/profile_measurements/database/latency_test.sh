#!/bin/bash

# === 配置区域 ===
HOST="35.197.105.4"
PORT="5432"
DB="postgres"
USER="postgres"
PASSWORD="你的密码"       # <--- 请确保这里是你的实际密码
DURATION=10               # 每次压力测试持续 10 秒
INTERVAL=30               # 每两次测试的起始间隔为 30 秒
TOTAL_RUNS=120            # 1小时共运行 120 次 (120 * 30s = 3600s)
OUTPUT_FILE="alloydb_latency.csv"

export PGPASSWORD=$PASSWORD

# 初始化 CSV 表头
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Timestamp,Latency_Avg_ms,TPS" > "$OUTPUT_FILE"
fi

echo "开始 1 小时连续监控测试..."
echo "计划任务: 每 $INTERVAL 秒运行一次，共执行 $TOTAL_RUNS 次。"

for ((i=1; i<=TOTAL_RUNS; i++))
do
    START_TIME=$(date +%s)
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    
    echo "[$i/$TOTAL_RUNS] 测试中... $TIMESTAMP"
    
    # 执行 pgbench 检索测试 (-S 只读模式)
    RESULT=$(pgbench -S -c 10 -T $DURATION "host=$HOST port=$PORT dbname=$DB user=$USER sslmode=require" 2>&1)
    
    # 提取数据（pgbench 输出为 "latency average = 1.234 ms"，awk 的 $3 是 "=" 而非数值，需用正则）
    LATENCY=$(echo "$RESULT" | grep -E 'latency average[[:space:]]*=' | sed -n 's/.*average[[:space:]]*=[[:space:]]*\([0-9.]*\)[[:space:]]*ms.*/\1/p' | head -1)
    TPS=$(echo "$RESULT" | grep -F 'including connections establishing' | sed -n 's/^[[:space:]]*tps[[:space:]]*=[[:space:]]*\([0-9.]*\).*/\1/p' | head -1)
    
    if [ -z "$LATENCY" ] && [ -z "$TPS" ]; then
        echo "[$i/$TOTAL_RUNS] 警告: 未能从 pgbench 输出解析到指标，请检查连接与 pgbench 日志" >&2
    fi
    
    # 写入文件
    echo "$TIMESTAMP,$LATENCY,$TPS" >> "$OUTPUT_FILE"
    
    # 计算需要休眠的时间，以确保每 30 秒准时开始下一次测试
    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    SLEEP_TIME=$(( INTERVAL - ELAPSED ))
    
    if [ $i -lt $TOTAL_RUNS ] && [ $SLEEP_TIME -gt 0 ]; then
        sleep $SLEEP_TIME
    fi
done

echo "测试圆满完成！结果保存在: $OUTPUT_FILE"