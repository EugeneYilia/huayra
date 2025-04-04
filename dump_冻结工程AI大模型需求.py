from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import re
import uuid

# 初始化嵌入模型
model = SentenceTransformer("BAAI/bge-large-zh")

# Step 1：文档结构切分（“【模块名】+子段落”）
def load_structured_chunks(filepath, source_name):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    raw_chunks = re.split(r"【(.*?)】", text)
    chunks = []
    for i in range(1, len(raw_chunks), 2):
        title = raw_chunks[i].strip()
        content = raw_chunks[i + 1].strip()
        sub_chunks = re.split(r"(?:\n\s*\n)|(?:\n\d+\.|\n\(.\))", content)
        for sub in sub_chunks:
            sub = sub.strip()
            if len(sub) < 30:
                continue
            chunks.append({
                "id": str(uuid.uuid4()),
                "section": title,
                "content": sub,
                "source": source_name
            })
    return chunks

# Step 2：连接 Qdrant（可远程也可本地）
client = QdrantClient("localhost", port=6333)  # 修改为远程地址或 API 密钥也可以

collection_name = "enterprise_knowledge"

# Step 3：初始化 collection（只需执行一次）
def init_collection():
    if collection_name not in [col.name for col in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)  # bge-large-zh 输出维度为 1024
        )

# Step 4：嵌入并写入 Qdrant 向量数据库
def write_to_qdrant(chunks):
    texts = [f"{c['section']}\n{c['content']}" for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    points = [
        PointStruct(
            id=chunk["id"],
            vector=embedding,
            payload={
                "section": chunk["section"],
                "content": chunk["content"],
                "source": chunk["source"]
            }
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    client.upsert(collection_name=collection_name, points=points)
    print(f"成功写入 Qdrant：{len(points)} 条")

# === 主流程 ===
init_collection()
chunks = load_structured_chunks("sources/井筒冻结工程AI大模型需求_增强版.md", source_name="井筒冻结AI需求文档")
write_to_qdrant(chunks)
