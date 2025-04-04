# embed_neighbor_spacing.py

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import os

# === 配置 ===
EMBED_MODEL = "BAAI/bge-large-zh"
EMBED_DIM = 1024
COLLECTION_NAME = "neighbor_spacing_data"
FILE_PATH = "sources/邻孔间距数据_语义增强版_清洁版_分块增强版.txt"
BATCH_SIZE = 1000  # 每批写入上限，避免 payload 超限

# === 初始化 Qdrant ===
client = QdrantClient(host="localhost", port=6333)

# 创建或重建 collection（替代 recreate_collection）
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
)

# === 加载嵌入模型 ===
model = SentenceTransformer(EMBED_MODEL)

# === 读取文本并准备数据 ===
points = []
with open(FILE_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        vec = model.encode(line).tolist()
        payload = {
            "source": "邻孔间距数据",
            "text": line
        }
        if "孔号为" in line:
            payload["孔号"] = line.split("孔号为")[1].split("，")[0].strip()
        if "深度" in line:
            payload["深度"] = line.split("深度")[1].split("米")[0].replace("处", "").strip()
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

# === 分批写入 Qdrant，防止 Payload Too Large ===
print(f"🚀 开始写入 {len(points)} 条邻孔间距数据向量至 Qdrant...")
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i + BATCH_SIZE]
    client.upsert(collection_name=COLLECTION_NAME, points=batch)
    print(f"✅ 已写入第 {i // BATCH_SIZE + 1} 批（{len(batch)} 条）")

print(f"🎉 所有数据成功写入 Qdrant（collection: {COLLECTION_NAME}）")
