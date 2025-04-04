from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

# 初始化嵌入模型
model = SentenceTransformer("BAAI/bge-large-zh")

# 读取整个文档作为单一段落
def load_config_text(filepath, section="项目参数", source="智能钻孔配置文档_东欢坨"):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return [{
        "id": str(uuid.uuid4()),
        "section": section,
        "content": text,
        "source": source
    }]

# Qdrant 初始化
client = QdrantClient("localhost", port=6333)
collection_name = "enterprise_knowledge"

def init_qdrant():
    if collection_name not in [col.name for col in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

# 写入向量
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

# 主流程
init_qdrant()
chunks = load_config_text("sources/智能钻孔静态文档_东欢坨项目版.txt")
embed_and_upsert(chunks)
