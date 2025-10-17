# 服装属性识别服务 Docker镜像

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY ../requirements.txt /app/requirements.txt
COPY service/requirements_service.txt /app/service/requirements_service.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r service/requirements_service.txt

# 复制代码
COPY . /app/

# 切换到服务目录
WORKDIR /app/service

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
