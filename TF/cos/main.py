import jieba
import re
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PlagiarismChecker:

    # 内置停用词表
    STOPWORDS = {
        '是', '的', '了', '在', '和', '要', '我', '这', '那', '就', '也', '不',
        '，', '。', '！', '?', '、', '；', '：', '“', '”', '（', '）', '【', '】'
    }

    def __init__(self):
        #初始化查重器
        #直接使用内置停用词表
        self.stopwords = self.STOPWORDS

        #自动加载项目根目录的 synonyms.txt（同义词表）
        script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
        synonyms_path = os.path.join(script_dir, 'synonyms.txt')
        self.synonyms = self._load_synonyms(synonyms_path) if os.path.exists(synonyms_path) else {}

    def _load_synonyms(self, file_path):
        #加载同义词表
        synonyms = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if ',' in line:
                        src, dst = line.split(',', 1)
                        synonyms[src.strip()] = dst.strip()
        except Exception as e:
            print(f"[警告] 同义词表加载失败：{str(e)}")
        return synonyms

    def preprocess(self, text):
        #文本预处理
        #text:原始文本字符串
        #return:标准化后的空格分隔词序列

        #移除非中文字符（保留基本标点）
        text = re.sub(r'[^\u4e00-\u9fa5，。！？；：、\s]', '', text)
        #使用Jieba精确分词
        words = jieba.lcut(text)
        #同义词替换+停用词过滤
        processed_words = []
        for word in words:
            word = self.synonyms.get(word, word)
            if word not in self.stopwords and len(word) > 0:
                processed_words.append(word)

        print(f"[预处理结果] {processed_words}")  # 添加此行
        return ' '.join(processed_words)

    def calculate_similarity(self, text_a, text_b):
        #计算余弦相似度
        #text_a:预处理后的文本A
        #text_b:预处理后的文本B
        #return:相似度值（0~1）
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            ngram_range=(1, 2),  #同时考虑单个词和双词组合
            use_idf=False,
            norm=None,#如果抄袭版比原文多出大量无关内容，启用L2归一化，相似度可能被稀释（因向量被拉长后归一化）
            max_features = 5000  #限制特征维度防止稀释
        )
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def main():
    #配置命令行参数解析器
    parser = argparse.ArgumentParser(description='中文论文查重系统')
    parser.add_argument('orig_path', help='论文原文绝对路径')
    parser.add_argument('plag_path', help='抄袭版论文绝对路径')
    parser.add_argument('output_path', help='输出答案文件绝对路径')
    args = parser.parse_args()

    try:
        output_path = os.path.abspath(args.output_path)
        output_dir = os.path.dirname(output_path)

        #确保输出目录存在
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        #验证输入文件存在
        if not os.path.isfile(args.orig_path):
            raise FileNotFoundError(f"原文文件不存在：{args.orig_path}")
        if not os.path.isfile(args.plag_path):
            raise FileNotFoundError(f"抄袭版文件不存在：{args.plag_path}")

        #初始化查重器（不再传递停用词表参数）
        checker = PlagiarismChecker()

        #读取并处理文件
        with open(args.orig_path, 'r', encoding='utf-8') as f:
            orig_text = checker.preprocess(f.read())
        with open(args.plag_path, 'r', encoding='utf-8') as f:
            plag_text = checker.preprocess(f.read())

        #计算相似度
        similarity = checker.calculate_similarity(orig_text, plag_text)
        result = round(similarity * 100, 2)

        #写入结果文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"论文查重率：{result}%")
        print(f"[完成] 结果已保存至：{output_path}")

    except Exception as e:
        print(f"[错误] {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()

#在终端输入：python main.py "orig.txt" "orig_add.txt" "output.txt"