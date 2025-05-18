import os
import torch
from intertextuality_keyword import load_txt, parse_keywords_to_single_list
from intertextuality_sentence_vec import extract_sentences_with_paths, load_json_data
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# 加载BERT模型
model_path = '../guwenbert'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)

# 提取BERT向量函数
def get_bert_embedding(texts, tokenizer, model, device, batch_size=64):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
            torch.cuda.empty_cache()  # 清理显存
    return embeddings

path_list = ["n001", "n006", "n007", "n013", "n014", "n019", "n020", "n022"]
name_list = ["周易", "老子", "论语", "列子", "文子", "孟子", "庄子", "荀子"]
script_dir = os.path.dirname(os.path.abspath(__file__))

print("-" * 50)
for i, target_file_relative_1 in enumerate(path_list):
    file_path_1 = os.path.join(script_dir, "keyword_txt/", target_file_relative_1+".txt")
    file_path_2 = os.path.join(script_dir, "base/", target_file_relative_1+".json")

    # 加载关键词和句子
    keyword_text_content_1 = load_txt(file_path_1)
    merged_keyword_list_text1 = parse_keywords_to_single_list(keyword_text_content_1, unique=True)
    data = load_json_data(file_path_2)
    sents_with_paths = extract_sentences_with_paths(data, os.path.basename(file_path_2))
    texts1 = [s[1] for s in sents_with_paths if len(s[1])>6]

    # 计算句子和关键词向量
    sentence_embeddings = get_bert_embedding(texts1, tokenizer, model, device)

    print(f"从《{name_list[i]}》提取关键词与最相关句子：")
    for keyword in merged_keyword_list_text1:
        keyword_embedding = get_bert_embedding([keyword], tokenizer, model, device)
        similarities = cosine_similarity(keyword_embedding, sentence_embeddings)[0]
        top_indices = similarities.argsort()[-3:][::-1]
        print(f"\n关键词：{keyword}")
        for idx in top_indices:
            print(f"相关句：{texts1[idx]} (相似度：{similarities[idx]:.4f})")

    print("-" * 50)