sudo apt-get update
sudo apt-get install -y postgresql-client


# 格式：psql "连接字符串"
# psql "host=35.197.105.4 port=5432 dbname=postgres user=postgres sslmode=require"


pgbench -i -s 50 \
  "host=35.197.105.4 port=5432 dbname=postgres user=postgres sslmode=require"