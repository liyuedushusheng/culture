from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

def parse_keywords_to_single_list(text_data, unique=False):
    all_keywords = []
    lines = text_data.strip().split('\n') 
    pattern = re.compile(r"^\s*-\s*(.+?)(?:\s*:.*|$)")

    for line in lines:
        line = line.strip() 
        if not line: 
            continue
        match = pattern.match(line)
        if match:
            keywords_section = match.group(1).strip() 
            current_line_keywords = keywords_section.split()
            
            if current_line_keywords:
                all_keywords.extend(current_line_keywords)       
    if unique:
        return list(set(all_keywords))
    return all_keywords

def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def keyword_jaccard_similarity(keywords1, keywords2):
    """基于Jaccard相似度的关键词分析"""
    set1 = set(keywords1)
    set2 = set(keywords2)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0


def keyword_cosine_similarity(keywords1, keywords2):
    """基于余弦相似度的关键词分析"""
    corpus = [' '.join(keywords1), ' '.join(keywords2)]
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(corpus)
    return cosine_similarity(X[0], X[1])[0][0]


if __name__ == "__main__":
    path_list = ["keyword_txt/n001.txt", "keyword_txt/n006.txt", "keyword_txt/n007.txt", 
                 "keyword_txt/n013.txt", "keyword_txt/n014.txt", "keyword_txt/n019.txt",
                 "keyword_txt/n020.txt", "keyword_txt/n022.txt"]
    name_list = ["周易", "老子", "论语", "列子", "文子", "孟子", "庄子", "荀子"]    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("-" * 50)
    for i, target_file_relative_1 in enumerate(path_list):
        for j in range(len(path_list) - i - 1):
            file_path_1 = os.path.join(script_dir, target_file_relative_1)
            keyword_text_content_1 = load_txt(file_path_1)
            merged_keyword_list_text1 = parse_keywords_to_single_list(keyword_text_content_1, unique=True)
            print(f"从《{name_list[i]}》解析并合并的关键词列表:")
            print(merged_keyword_list_text1)

            target_file_relative_2 = path_list[i+j+1]
            file_path_2 = os.path.join(script_dir, target_file_relative_2)
            keyword_text_content_2 = load_txt(file_path_2)
            merged_keyword_list_text2 = parse_keywords_to_single_list(keyword_text_content_2, unique=True)
            print(f"\n从《{name_list[i+j+1]}》解析并合并的关键词列表:")
            print(merged_keyword_list_text2)

            print("\n关键词方法:")
            print(f"Jaccard相似度: {keyword_jaccard_similarity(merged_keyword_list_text1, merged_keyword_list_text2):.2f}")
            print(f"余弦相似度: {keyword_cosine_similarity(merged_keyword_list_text1, merged_keyword_list_text2):.2f}")
            print("-" * 50)