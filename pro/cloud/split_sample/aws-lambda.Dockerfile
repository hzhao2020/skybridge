FROM public.ecr.aws/lambda/python:3.11

RUN microdnf install -y ffmpeg && microdnf clean all

COPY app.py lambda_function.py ${LAMBDA_TASK_ROOT}/

ENV SAMPLES_PER_SHOT=3

CMD ["lambda_function.handler"]

