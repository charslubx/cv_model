# 服装属性多标签分类服务

基于GAT+GCN的服装属性多标签分类模型推理服务，使用FastAPI构建。

## 功能特性

- ✅ 单张图片多标签分类
- ✅ 批量图片分类
- ✅ 支持26个服装属性的多标签分类
- ✅ 可调节分类阈值
- ✅ RESTful API接口
- ✅ 异步处理
- ✅ CORS支持

## 关于多标签分类

本模型执行的是**多标签分类**任务，即一张图片可以同时具有多个属性。例如，一件衣服可以同时是"黑色"、"纯色"、"棉质"等。

- **输入**：服装图片
- **输出**：26个属性的分类结果（每个属性0或1）和置信度（0-1之间的概率值）
- **阈值**：默认使用0.5作为分类阈值，置信度≥0.5的属性被分类为1（正类），否则为0（负类）

## 环境要求

- Python 3.8+
- CUDA 11.8+ (可选，用于GPU加速)
- PyTorch 2.4.1+

## 安装依赖

```bash
# 安装主项目依赖
pip install -r ../requirements.txt

# 安装服务依赖
pip install -r requirements_service.txt
```

## 配置

服务配置可以通过环境变量设置：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| HOST | 0.0.0.0 | 服务监听地址 |
| PORT | 8000 | 服务端口 |
| DEVICE | cuda | 推理设备 (cuda/cpu) |
| MODEL_PATH | ../checkpoints/best_model.pth | 模型权重路径 |
| NUM_CLASSES | 26 | 属性类别数量 |
| MAX_BATCH_SIZE | 32 | 最大批量大小 |

## 启动服务

### 方式1: 直接运行

```bash
cd service
python app.py
```

### 方式2: 使用uvicorn

```bash
cd service
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 方式3: 使用启动脚本

```bash
bash start_service.sh
```

## API接口

### 1. 健康检查

```bash
curl http://localhost:8000/health
```

响应:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. 单张图片分类

```bash
# 使用默认阈值0.5
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"

# 指定自定义阈值
curl -X POST "http://localhost:8000/predict?threshold=0.6" \
  -F "file=@/path/to/image.jpg"
```

响应:
```json
{
  "success": true,
  "filename": "image.jpg",
  "predictions": {
    "attributes": {
      "black": 0.95,
      "blue": 0.12,
      "pattern_solid": 0.89,
      ...
    },
    "classifications": {
      "black": 1,
      "blue": 0,
      "pattern_solid": 1,
      ...
    },
    "positive_attributes": [
      {"attribute": "black", "confidence": 0.95},
      {"attribute": "pattern_solid", "confidence": 0.89},
      ...
    ],
    "top_k_attributes": [
      {"attribute": "black", "confidence": 0.95, "classification": 1},
      {"attribute": "pattern_solid", "confidence": 0.89, "classification": 1},
      ...
    ],
    "statistics": {
      "total_attributes": 26,
      "positive_count": 5,
      "threshold": 0.5
    }
  }
}
```

### 3. 批量图片分类

```bash
# 使用默认阈值
curl -X POST "http://localhost:8000/predict_batch" \
  -F "files=@/path/to/image1.jpg" \
  -F "files=@/path/to/image2.jpg"

# 指定自定义阈值
curl -X POST "http://localhost:8000/predict_batch?threshold=0.6" \
  -F "files=@/path/to/image1.jpg" \
  -F "files=@/path/to/image2.jpg"
```

### 4. 获取属性列表

```bash
curl http://localhost:8000/attributes
```

响应:
```json
{
  "attributes": [
    "black", "blue", "brown", "collar", "cyan",
    "gray", "green", "many_colors", "necktie", 
    ...
  ],
  "num_classes": 26
}
```

## 测试

运行测试脚本：

```bash
python test_service.py
```

## 性能优化

1. **GPU加速**: 设置 `DEVICE=cuda` 使用GPU加速
2. **批量处理**: 使用 `/predict_batch` 接口进行批量预测
3. **多worker**: 生产环境可以增加worker数量
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
   ```

## Docker部署

```bash
# 构建镜像
docker build -t fashion-attr-service .

# 运行容器
docker run -d -p 8000:8000 \
  -v /path/to/checkpoints:/app/checkpoints \
  --gpus all \
  fashion-attr-service
```

## 故障排查

### 1. 模型加载失败

- 检查模型文件路径是否正确
- 确认模型文件存在
- 如果没有模型文件，服务会使用预训练权重初始化

### 2. CUDA错误

- 检查CUDA是否正确安装
- 设置 `DEVICE=cpu` 使用CPU推理

### 3. 内存不足

- 减少 `MAX_BATCH_SIZE`
- 使用CPU推理
- 减少worker数量

## 开发

### 修改属性列表

编辑 `model_loader.py` 中的 `_get_attribute_names` 方法。

### 自定义模型

修改 `model_loader.py` 中的 `ModelInference.__init__` 方法。

## 许可证

MIT License
