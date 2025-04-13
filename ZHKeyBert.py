import json
from zhkeybert import KeyBERT
import hanlp

file_path = 'n001.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

model_path = "D:/culture/zhkeybert_model"
# 初始化 KeyBERT 模型
model = KeyBERT(model=model_path)
HanLP = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

# 合并所有卦象的文本
all_texts = []
for texts in data['周易'].values():
    all_texts.extend(texts)
combined_text = " ".join(all_texts)

# 使用 HanLP 进行分词
term_list = HanLP(combined_text)
# 处理分词结果，确保只对具有 word 属性的对象访问 word 属性
tokenized_text = " ".join([term.word if hasattr(term, 'word') else term for term in term_list])

# 对分词后的文本进行关键词提取
keywords = model.extract_keywords(tokenized_text, keyphrase_ngram_range=(1, 2), use_mmr=True, diversity=0.25, top_n=10)

print("《周易》全文的抽象概念（关键词）:")
for keyword, score in keywords:
    print(f"  - {keyword}: {score}")