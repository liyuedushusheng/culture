from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")


def extract_keywords(text):
    """ 使用 KeyBART 模型进行关键词抽取 """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    keywords = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return keywords

text = "Chinese traditional culture, rooted in five millennia of history, embodies profound philosophical systems and artistic expressions. The teachings of Confucianism emphasizing ren (benevolence) and li (ritual propriety) shaped social harmony, while Daoist principles of wuwei (non-action) and yin-yang balance influenced medical practices like acupuncture. Iconic elements include the intricate brushwork of calligraphy, the symbolic patterns of porcelain from Jingdezhen, and the dynamic movements of Peking Opera performers wearing liandan facial makeup. Traditional festivals like the Lunar New Year feature nian gao rice cakes and dragon dances to ward off nian beasts. The ancient Silk Road facilitated cultural exchanges, introducing Chinese innovations such as papermaking and gunpowder to the world. Architectural marvels like the Forbidden City demonstrate perfect integration of feng shui principles and imperial cosmology. Traditional Chinese Medicine (TCM) utilizes herbal formulations based on the Five Elements theory, while practices like tai chi cultivate both physical health and spiritual alignment."
print(extract_keywords(text))

text = "中国传统文化植根于五千年的历史长河，蕴含着深邃的哲学体系与艺术表现形式。儒家思想以'仁'与'礼'为核心构建社会伦理秩序，道家'无为而治'的理念与阴阳平衡的哲学则深刻影响着针灸等医学实践。书法中精妙的笔触、景德镇瓷器上的象征性纹样、京剧演员绘着脸谱的灵动身段，共同构成独特的文化符号体系。春节年糕的香甜与舞龙驱赶年兽的传统，承载着岁时节令的文化记忆。古老的丝绸之路见证着造纸术与火药等发明的跨文明传播，紫禁城的建筑布局完美融合风水理念与皇家宇宙观。中医运用五行理论配伍草药，太极拳则通过刚柔相济的招式实现身心合一的境界。"
print(extract_keywords(text))