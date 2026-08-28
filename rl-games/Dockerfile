# Dockerfile —— 黑白棋/贪吃蛇 RL 平台(web + 训练 worker 共用同一镜像)
#
# 一个镜像两用:
#   · web   : uvicorn 起 FastAPI 服务(见 docker-compose.yml 的 command)
#   · worker: python 跑训练(同样由 command 指定)
# 训练写 models/ 与 data/(注册表、曲线),web 只读——通过挂载同一个卷共享,
# 完全沿用"训练/Web 解耦"的设计,容器间不共享内存、不互相依赖。

FROM python:3.12-slim

WORKDIR /app

# 先装依赖再拷代码:依赖层几乎不变,能吃到 Docker 层缓存,重建飞快
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 拷代码(排除 .dockerignore 里的大文件/缓存)
COPY . .

# 运行期需要的目录(挂载卷时会被覆盖,这里保证首次启动不报错)
RUN mkdir -p /app/models /app/data

# web 默认服务黑白棋;具体端口/进程由 docker-compose 的 command 覆盖
EXPOSE 8001 8002
CMD ["uvicorn", "othello.serve:app", "--host", "0.0.0.0", "--port", "8001"]
