import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import re
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei' 

class TextSimilarityAnalyzer:
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        """初始化文本相似度分析器"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # 存储处理后的文本数据
        self.zhouyi_data = None
        self.lunyu_data = None
        self.laozi_data = None
        self.liezi_data = None
        self.wenzi_data = None
        self.mengzi_data = None
        self.zhuangzi_data = None
        self.xunzi_data = None
        
        # 存储句向量
        self.zhouyi_vectors = None
        self.lunyu_vectors = None
        self.laozi_vectors = None
        self.liezi_vectors = None
        self.wenzi_vectors = None
        self.mengzi_vectors = None
        self.zhuangzi_vectors = None
        self.xunzi_vectors = None

    def read_text_file(self, file_path: str) -> str:
        """读取文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""

    def extract_sentences(self, text: str) -> List[str]:
        """从文本中提取句子"""
        # 简单的句子分割，根据中文标点符号
        sentences = re.split(r'[。！？；\n]', text)
        # 过滤空句子
        return [s.strip() for s in sentences if s.strip()]

    def locate_keyword_sentences(self, text: str, keywords: List[str]) -> Dict[str, List[str]]:
        """定位能够提炼出关键词的句子（基于语义相似度）"""
        # 1. 先提取文本中的所有句子
        sentences = self.extract_sentences(text)
        keyword_sentences = {kw: [] for kw in keywords}
    
        # 2. 为每个关键词创建一个"语义查询向量"
        keyword_vectors = {}
        for kw in keywords:
            # 将关键词转换为向量
            encoded_kw = self.tokenizer(kw, padding=True, truncation=True, max_length=128, return_tensors='pt')
            encoded_kw = {k: v.to(self.device) for k, v in encoded_kw.items()}
        
            with torch.no_grad():
                model_output = self.model(**encoded_kw)
        
            # 应用平均池化获取关键词向量
            kw_vector = self.mean_pooling(model_output, encoded_kw['attention_mask'])
            kw_vector = torch.nn.functional.normalize(kw_vector, p=2, dim=1)
            keyword_vectors[kw] = kw_vector.cpu().numpy()[0]  # 转为numpy数组
    
        # 3. 为所有句子生成向量
        print(f"为 {len(sentences)} 个句子生成向量...")
        sentence_vectors = self.get_sentence_vectors(sentences)
    
        # 4. 计算每个关键词与所有句子的相似度
        for kw, kw_vector in keyword_vectors.items():
            # 计算关键词向量与所有句子向量的余弦相似度
            similarities = cosine_similarity([kw_vector], sentence_vectors)[0]
        
            # 找出相似度最高的前N个句子（这里取前5个）
            top_indices = np.argsort(-similarities)[:2]
        
            # 只保留相似度大于阈值的句子（这里设为0.4，可以调整）
            for idx in top_indices:
                if similarities[idx] > 0.4:
                    keyword_sentences[kw].append({
                        "sentence": sentences[idx],
                        "similarity": float(similarities[idx])
                    })
    
        return keyword_sentences

    def mean_pooling(self, model_output, attention_mask):
        """对模型输出进行平均池化，获取句子向量"""
        token_embeddings = model_output[0]  # 第一个元素是token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_sentence_vectors(self, sentences: List[str]) -> np.ndarray:
        """获取句子的向量表示"""
        vectors = []
        batch_size = 16
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="生成句向量"):
            batch = sentences[i:i+batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # 应用平均池化获取句向量
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            # 归一化
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            vectors.extend(sentence_embeddings.cpu().numpy())
        
        return np.array(vectors)

    def calculate_similarity(self, vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
        """计算两组向量之间的相似度矩阵"""
        return cosine_similarity(vectors_a, vectors_b)

    def analyze_concept_relations(self, keywords: List[str], zhouyi_path: str, lunyu_path: str, laozi_path: str, 
                                  liezi_path: str, wenzi_path: str, mengzi_path: str, zhuangzi_path: str, xunzi_path: str,
                                  output_path: Optional[str] = None, thresholds: List[float] = [0.5]) -> Dict:
        """分析《周易》与《老子》《论语》《列子》《文子》《孟子》《庄子》《荀子》的概念关系"""
        # 读取文本
        zhouyi_text = self.read_text_file(zhouyi_path)
        lunyu_text = self.read_text_file(lunyu_path)
        laozi_text = self.read_text_file(laozi_path)
        liezi_text = self.read_text_file(liezi_path)
        wenzi_text = self.read_text_file(wenzi_path)
        mengzi_text = self.read_text_file(mengzi_path)
        zhuangzi_text = self.read_text_file(zhuangzi_path)
        xunzi_text = self.read_text_file(xunzi_path)
        
        # 定位关键词所在句子
        print("定位《周易》中关键词所在句子...")
        zhouyi_keyword_sentences = self.locate_keyword_sentences(zhouyi_text, keywords)
        
        # 提取各书籍的所有句子
        print("提取各书籍的句子...")
        lunyu_sentences = self.extract_sentences(lunyu_text)
        laozi_sentences = self.extract_sentences(laozi_text)
        liezi_sentences = self.extract_sentences(liezi_text)
        wenzi_sentences = self.extract_sentences(wenzi_text)
        mengzi_sentences = self.extract_sentences(mengzi_text)
        zhuangzi_sentences = self.extract_sentences(zhuangzi_text)
        xunzi_sentences = self.extract_sentences(xunzi_text)
        
        # 收集所有需要计算向量的《周易》句子
        all_zhouyi_sentences = []
        keyword_sentence_map = {}
        for kw, sentences in zhouyi_keyword_sentences.items():
            if sentences:  # 只处理找到句子的关键词
                all_zhouyi_sentences.extend([s["sentence"] for s in sentences])
                keyword_sentence_map[kw] = len(all_zhouyi_sentences) - len(sentences)
        
        # 生成句向量
        print("生成句向量...")
        zhouyi_vectors = self.get_sentence_vectors(all_zhouyi_sentences)
        lunyu_vectors = self.get_sentence_vectors(lunyu_sentences)
        laozi_vectors = self.get_sentence_vectors(laozi_sentences)
        liezi_vectors = self.get_sentence_vectors(liezi_sentences)
        wenzi_vectors = self.get_sentence_vectors(wenzi_sentences)
        mengzi_vectors = self.get_sentence_vectors(mengzi_sentences)
        zhuangzi_vectors = self.get_sentence_vectors(zhuangzi_sentences)
        xunzi_vectors = self.get_sentence_vectors(xunzi_sentences)
        
        # 计算相似度
        print("计算相似度...")
        zhouyi_lunyu_sim = self.calculate_similarity(zhouyi_vectors, lunyu_vectors)
        zhouyi_laozi_sim = self.calculate_similarity(zhouyi_vectors, laozi_vectors)
        zhouyi_liezi_sim = self.calculate_similarity(zhouyi_vectors, liezi_vectors)
        zhouyi_wenzi_sim = self.calculate_similarity(zhouyi_vectors, wenzi_vectors)
        zhouyi_mengzi_sim = self.calculate_similarity(zhouyi_vectors, mengzi_vectors)
        zhouyi_zhuangzi_sim = self.calculate_similarity(zhouyi_vectors, zhuangzi_vectors)
        zhouyi_xunzi_sim = self.calculate_similarity(zhouyi_vectors, xunzi_vectors)
        
        results = {}
        for kw, start_idx in keyword_sentence_map.items():
            end_idx = start_idx + len(zhouyi_keyword_sentences[kw])
            kw_vectors = zhouyi_vectors[start_idx:end_idx]
            
            # 获取当前关键词对应的所有相似度分数
            lunyu_scores = zhouyi_lunyu_sim[start_idx:end_idx]
            laozi_scores = zhouyi_laozi_sim[start_idx:end_idx]
            liezi_scores = zhouyi_liezi_sim[start_idx:end_idx]
            wenzi_scores = zhouyi_wenzi_sim[start_idx:end_idx]
            mengzi_scores = zhouyi_mengzi_sim[start_idx:end_idx]
            zhuangzi_scores = zhouyi_zhuangzi_sim[start_idx:end_idx]
            xunzi_scores = zhouyi_xunzi_sim[start_idx:end_idx]
            
            keyword_results = []
            for threshold in thresholds:
                # 计算各书籍中高于阈值的比例
                lunyu_high_score = np.mean(lunyu_scores > threshold)
                laozi_high_score = np.mean(laozi_scores > threshold)
                liezi_high_score = np.mean(liezi_scores > threshold)
                wenzi_high_score = np.mean(wenzi_scores > threshold)
                mengzi_high_score = np.mean(mengzi_scores > threshold)
                zhuangzi_high_score = np.mean(zhuangzi_scores > threshold)
                xunzi_high_score = np.mean(xunzi_scores > threshold)
                
                keyword_results.append({
                    "threshold": threshold,
                    "lunyu_similarity_ratio": float(lunyu_high_score),
                    "laozi_similarity_ratio": float(laozi_high_score),
                    "liezi_similarity_ratio": float(liezi_high_score),
                    "wenzi_similarity_ratio": float(wenzi_high_score),
                    "mengzi_similarity_ratio": float(mengzi_high_score),
                    "zhuangzi_similarity_ratio": float(zhuangzi_high_score),
                    "xunzi_similarity_ratio": float(xunzi_high_score)
                })
            
            results[kw] = keyword_results
        
        # 保存结果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到 {output_path}")
        
        return results

    def visualize_results(self, results: Dict, thresholds: List[float]):
        """可视化结果"""
        for keyword, keyword_results in results.items():
            lunyu_ratios = [result["lunyu_similarity_ratio"] for result in keyword_results]
            laozi_ratios = [result["laozi_similarity_ratio"] for result in keyword_results]
            liezi_ratios = [result["liezi_similarity_ratio"] for result in keyword_results]
            wenzi_ratios = [result["wenzi_similarity_ratio"] for result in keyword_results]
            mengzi_ratios = [result["mengzi_similarity_ratio"] for result in keyword_results]
            zhuangzi_ratios = [result["zhuangzi_similarity_ratio"] for result in keyword_results]
            xunzi_ratios = [result["xunzi_similarity_ratio"] for result in keyword_results]
            
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, lunyu_ratios, label="《论语》")
            plt.plot(thresholds, laozi_ratios, label="《老子》")
            plt.plot(thresholds, liezi_ratios, label="《列子》")
            plt.plot(thresholds, wenzi_ratios, label="《文子》")
            plt.plot(thresholds, mengzi_ratios, label="《孟子》")
            plt.plot(thresholds, zhuangzi_ratios, label="《庄子》")
            plt.plot(thresholds, xunzi_ratios, label="《荀子》")
            
            plt.xlabel("相似度率的阈值")
            plt.ylabel("相似度率")
            plt.title(f"关键词 '{keyword}' 的相似度率随阈值变化")
            plt.legend()
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    # 使用示例
    analyzer = TextSimilarityAnalyzer()
    
    # 定义《周易》关键词
    keywords = ['仁者', '悔吝者', '不見','吉凶者']
    
    # 假设文本文件已准备好
    zhouyi_path = "n001.json"
    lunyu_path = "n007.json"
    laozi_path = "n006.json"
    liezi_path = "n013.json"
    wenzi_path = "n014.json"
    mengzi_path = "n019.json"
    zhuangzi_path = "n020.json"
    xunzi_path = "n022.json"
    
    # 定义阈值列表
    thresholds = [i / 100 for i in range(60, 95, 5)]
    
    # 分析概念关系
    results = analyzer.analyze_concept_relations(
        keywords=keywords,
        zhouyi_path=zhouyi_path,
        lunyu_path=lunyu_path,
        laozi_path=laozi_path,
        liezi_path=liezi_path,
        wenzi_path=wenzi_path,
        mengzi_path=mengzi_path,
        zhuangzi_path=zhuangzi_path,
        xunzi_path=xunzi_path,
        output_path="concept_relations.json",
        thresholds=thresholds
    )
    
    # 可视化结果
    analyzer.visualize_results(results, thresholds)