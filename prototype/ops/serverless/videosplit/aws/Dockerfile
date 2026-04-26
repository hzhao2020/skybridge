FROM public.ecr.aws/lambda/python:3.11

# 1. 安装基础工具 (tar, xz, wget)
# 注意：这里我们只安装基础解压工具，通常 AWS 源都有这些
RUN yum install -y tar xz wget && yum clean all

# 2. 下载并配置静态 FFmpeg (最稳健的方法)
# 直接从 johnvansickle 下载编译好的二进制文件，避开 yum 源缺失 ffmpeg 的问题
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    && tar -xf ffmpeg-release-amd64-static.tar.xz \
    && mv ffmpeg-*-amd64-static/ffmpeg /usr/bin/ffmpeg \
    && rm -rf ffmpeg-release-amd64-static.tar.xz ffmpeg-*-amd64-static

# 3. 复制依赖
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt -t ${LAMBDA_TASK_ROOT}

# 4. 复制代码
COPY main.py ${LAMBDA_TASK_ROOT}

CMD [ "main.lambda_handler" ]