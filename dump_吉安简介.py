from sentence_transformers import SentenceTransformer
import faiss
import json
import os

# 中文嵌入模型（可替换为 text2vec）
model = SentenceTransformer("BAAI/bge-large-zh")

# 读取 Markdown 文件并按模块切分
def load_markdown_sections(file_path):
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
        chunks.append({
            "title": title,
            "text": body
        })
    return chunks

# 嵌入 + 构建向量库
def build_vector_index(chunks, save_path="vector.index"):
    texts = [f"{chunk['title']}\n{chunk['text']}" for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 保存索引和原文内容
    faiss.write_index(index, save_path)
    with open("vector_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"向量库已保存，共 {len(texts)} 条记录")

# 使用示例
chunks = load_markdown_sections("sources/烟台吉安简介文档_增强版.md")
build_vector_index(chunks)
