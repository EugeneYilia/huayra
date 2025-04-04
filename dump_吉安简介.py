from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import json
import re
import uuid

# === 初始化嵌入模型（bge-large-zh） ===
model = SentenceTransformer("BAAI/bge-large-zh")

# === 加载文档并结构化分段 ===
def load_markdown_sections(file_path, source_name):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = content.split("【")
    chunks = []
    for sec in sections:
        if not sec.strip():
            continue
        title_end = sec.find("】")
        if title_end == -1:
            continue
        title = sec[:title_end].strip()
        body = sec[title_end + 1:].strip()

        # 可选：按换行分段（控制每段内容不宜太长）
        sub_chunks = re.split(r"\n\s*\n", body)
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

# === Qdrant 初始化 ===
client = QdrantClient("localhost", port=6333)  # 如果是远程服务请替换为对应 host 和 port

collection_name = "enterprise_knowledge"

def init_qdrant():
    if collection_name not in [col.name for col in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

# === 向量化并写入 Qdrant ===
def embed_and_upsert(chunks):
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
    print(f"写入完成：{len(points)} 条记录")

# === 主流程 ===
init_qdrant()
chunks = load_markdown_sections("sources/烟台吉安简介文档_增强版.md", source_name="烟台吉安简介文档")
embed_and_upsert(chunks)
