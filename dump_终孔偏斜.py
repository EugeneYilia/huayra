# embed_terminal_deviation.py

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# === 嵌入配置 ===
EMBED_MODEL = "BAAI/bge-large-zh"
EMBED_DIM = 1024
COLLECTION_NAME = "terminal_deviation_data"
FILE_PATH = "sources/终孔偏斜数据_语义增强版_清洁版_分块增强版.txt"

# === 初始化 Qdrant ===
client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
)

# === 加载嵌入模型 ===
model = SentenceTransformer(EMBED_MODEL)

# === 处理文本并写入向量库 ===
points = []
with open(FILE_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 嵌入
        vec = model.encode(line).tolist()
        # 提取元数据
        meta = {"source": "终孔偏斜数据"}
        if "终孔孔号为" in line:
            meta["孔号"] = line.split("终孔孔号为")[1].split("，")[0].strip()
        if "设计孔深为" in line:
            meta["设计孔深"] = line.split("设计孔深为")[1].split("米")[0].strip()
        meta["text"] = line

        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=meta))

# 写入 Qdrant
client.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"✅ 成功写入 {len(points)} 条‘终孔偏斜’数据到 Qdrant（collection: {COLLECTION_NAME}）")
