FROM public.ecr.aws/lambda/python:3.11

# 安装 ffmpeg
RUN yum install -y ffmpeg

COPY main.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt -t ${LAMBDA_TASK_ROOT}

CMD [ "main.lambda_handler" ]

