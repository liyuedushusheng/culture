import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt
import numpy as np
import scienceplots


def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_sentences_with_paths(data, doc_id_prefix="doc"):
    """
    从JSON数据中提取所有句子，并为每个句子生成一个唯一的路径标识符。
    返回一个列表，其中每个元素是 (path_id, sentence_text) 的元组。
    """
    sentences_with_paths = []
    for book_title, chapters in data.items():
        for chapter_name, sentences_list in chapters.items():
            for i, sentence in enumerate(sentences_list):
                path_id = f"{doc_id_prefix}|{book_title}|{chapter_name}|sentence_{i}"
                sentences_with_paths.append((path_id, sentence))
    return sentences_with_paths

def find_intertextual_relations(file1_path, file2_path, model, similarity_threshold=0.6):
    data1 = load_json_data(file1_path)
    data2 = load_json_data(file2_path)

    sents_with_paths1 = extract_sentences_with_paths(data1, os.path.basename(file1_path))
    sents_with_paths2 = extract_sentences_with_paths(data2, os.path.basename(file2_path))

    texts1 = [s[1] for s in sents_with_paths1]
    texts2 = [s[1] for s in sents_with_paths2]
    pair_number = len(texts1) * len(texts2)
  
    embeddings1 = model.encode(texts1, show_progress_bar=True, convert_to_numpy=True)
    embeddings2 = model.encode(texts2, show_progress_bar=True, convert_to_numpy=True)
    sim_matrix = cosine_similarity(embeddings1, embeddings2)

    results = []
    bound_number = [0] * 36
    for i in range(len(texts1)):
        for j in range(len(texts2)):
            similarity = sim_matrix[i, j]
            if similarity >= similarity_threshold:
                path1, sent1 = sents_with_paths1[i]
                path2, sent2 = sents_with_paths2[j]
                
                result_entry = {
                    "file1_path_id": path1,
                    "file1_sentence": sent1,
                    "file2_path_id": path2,
                    "file2_sentence": sent2,
                    "similarity": float(similarity) 
                }
                results.append(result_entry)
                
                '''print(f"\n相似度: {similarity:.4f}")
                print(f"  来源1 ({path1}): {sent1}")
                print(f"  来源2 ({path2}): {sent2}")'''
            
            for bound in range(60, 96, 1):
                if similarity >= bound / 100:
                    bound_number[bound-60] += 1
    if not results:
        print(f"未找到相似度高于 {similarity_threshold} 的句子对。")

    bound_rate = [0] * 36
    for i in range(len(bound_number)):
        bound_rate[i] = bound_number[i] / pair_number

    return results, bound_rate


if __name__ == '__main__':
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['Songti SC']  #此行代码为MacOS中显示中文的设置，其它系统可能需要不同的设置。
    plt.rcParams['axes.unicode_minus'] = False  
    index = np.array(range(60, 96, 1))

    model_name = 'paraphrase-multilingual-MiniLM-L12-v2' 
    print(f"\n\n正在加载模型: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"模型加载完成。")

    similarity_param = 0.6

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_directory = "base/"
    file_name_list = ["base/n001.json", "base/n006.json", "base/n007.json", 
                      "base/n013.json", "base/n014.json", "base/n019.json",
                      "base/n020.json", "base/n022.json"]
    name_list = ["周易", "Laozi", "Lunyu", "Liezi", "Wenzi", "Mengzi", "Zhuangzi", "Xunzi"]    

    for i, file1 in enumerate(["base/n001.json"]): #现在设置的仅仅是《周易》和其他几本书之间的互文关系。
        for j in range(len(file_name_list) - i - 1):
            file1_full_path = os.path.join(script_dir, file1)
            file2_full_path = os.path.join(script_dir, file_name_list[i+j+1])

            print(f"\n\n《{name_list[i]}》和《{name_list[i+j+1]}》之间的互文关系分析:")

            intertextual_results, bound_rate = find_intertextual_relations(
                file1_full_path, 
                file2_full_path,
                model=model,
                similarity_threshold=similarity_param
            )

            plt.plot(index, bound_rate, 'o-', label=f"{name_list[i+j+1]}")

            if intertextual_results:
                #print(f"\n\n总共找到 {len(intertextual_results)} 组互文关系 (相似度 >= {similarity_param})。")
                
                file1_basename_no_ext = os.path.splitext(os.path.basename(file1_full_path))[0]
                file2_basename_no_ext = os.path.splitext(os.path.basename(file2_full_path))[0]
                output_filename = f"intertextual_results/《{name_list[i]}》and《{name_list[i+j+1]}》_thresh{str(similarity_param).replace('.', 'p')}.json"
                
                with open(output_filename, 'w', encoding='utf-8') as f_out:
                    json.dump(intertextual_results, f_out, ensure_ascii=False, indent=2)
                #print(f"结果已保存到 {output_filename}")

    plt.xlabel('Cosine Similarity Threshold(%)')
    plt.ylabel('Number')
    plt.title('Intertextuality rate between Zhouyi and other books')
    plt.legend()
    plt.tight_layout()
    plt.savefig("curves/intertextuality_rate_curve.pdf", format="pdf", bbox_inches="tight")