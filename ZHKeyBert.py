import json
from zhkeybert import KeyBERT

file_path = 'n001.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

model_path = "/ceph/home/luozixiang/culture/zh_model"


# 初始化 KeyBERT 模型
model = KeyBERT(model=model_path)

for hexagram, texts in data['周易'].items():
    combined_text = " ".join(texts)
    
    keywords = model.extract_keywords(combined_text, keyphrase_ngram_range=(1, 2), use_mmr=True, diversity=0.25,top_n=5)
    
    print(f"卦象 {hexagram} 的抽象概念（关键词）:")
    for keyword, score in keywords:
        print(f"  - {keyword}: {score}")
    print()