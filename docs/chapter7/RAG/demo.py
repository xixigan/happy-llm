import os
import sys
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# 没有保存数据库
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割
vector = VectorStore(docs)
embedding = OpenAIEmbedding() # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

# vector.load_vector('./storage') # 加载本地的数据库

question = 'RAG的原理是什么？'

results = vector.query(question, EmbeddingModel=embedding, k=1)
if not results:
    print("没有检索到相似文档，query 返回空。请确认是否已生成向量数据库。")
    content = ""
else:
    content = results[0]

chat = OpenAIChat(model='Qwen/Qwen3-VL-8B-Instruct')
print("答案:", chat.chat(question, [], content))