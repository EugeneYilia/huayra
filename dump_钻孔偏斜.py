# embed_borehole_deviation.py

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# === 配置 ===
EMBED_MODEL = "BAAI/bge-large-zh"
EMBED_DIM = 1024
COLLECTION_NAME = "borehole_deviation_data"
FILE_PATH = "sources/钻孔偏斜数据_语义增强版_清洁版_分块增强版.txt"
BATCH_SIZE = 1000

# === 初始化 Qdrant ===
client = QdrantClient(host="localhost", port=6333)

# 安全创建 collection
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
)

# === 加载嵌入模型 ===
model = SentenceTransformer(EMBED_MODEL)

# === 构建向量 + 提取孔号/深度 ===
points = []
with open(FILE_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        vector = model.encode(line).tolist()
        payload = {
            "source": "钻孔偏斜数据",
            "text": line
        }
        if "钻孔孔号为" in line:
            payload["孔号"] = line.split("钻孔孔号为")[1].split("，")[0].strip()
        if "钻孔深度为" in line:
            payload["深度"] = line.split("钻孔深度为")[1].split("米")[0].strip()
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

# === 分批写入 Qdrant ===
print(f"🚀 开始写入 {len(points)} 条钻孔偏斜数据向量到 Qdrant...")
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i + BATCH_SIZE]
    client.upsert(collection_name=COLLECTION_NAME, points=batch)
    print(f"✅ 第 {i // BATCH_SIZE + 1} 批写入成功，共 {len(batch)} 条")

print(f"🎉 所有数据成功写入 Qdrant（collection: {COLLECTION_NAME}）")
