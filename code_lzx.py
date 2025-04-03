import json
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
import jieba
import jieba.analyse


tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    texts = []
    for hexagram_content in data['周易'].values():
        texts.extend(hexagram_content)
    return texts

file_path = 'n001.json'
texts = load_data(file_path)

inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()


n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit_predict(embeddings)


cluster_texts = {i: [] for i in range(n_clusters)}
for i, label in enumerate(labels):
    cluster_texts[label].append(texts[i])


def extract_keywords_from_cluster(texts, topK=5):
    combined_text = " ".join(texts)
    keywords = jieba.analyse.textrank(combined_text, topK=topK)
    return keywords

cluster_keywords = {}
for cluster_id, cluster_text_list in cluster_texts.items():
    keywords = extract_keywords_from_cluster(cluster_text_list)
    cluster_keywords[cluster_id] = keywords


for cluster_id, keywords in cluster_keywords.items():
    print(f"聚类 {cluster_id} 的核心概念: {', '.join(keywords)}")


#数据加载与编码：使用 load_data 函数读取 n001.json 文件中的文本内容，接着利用预训练的 BERT 模型将文本编码为向量。
#聚类操作：运用 K-Means 算法对文本向量进行聚类，把文本划分成 n_clusters 个组。
#分组文本：依据聚类结果，将文本按照聚类标签分组，存储在 cluster_texts 字典中。
#提取关键词：定义 extract_keywords_from_cluster 函数，把每个聚类中的所有文本合并成一个长文本，然后使用 jieba.analyse.textrank 算法从中提取前 topK 个关键词。
#打印核心概念：遍历每个聚类，打印出对应的核心概念（关键词）