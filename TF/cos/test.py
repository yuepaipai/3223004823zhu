import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
import sys
import numpy as np

# 修复路径问题
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from main import PlagiarismChecker


class TestPlagiarismChecker(unittest.TestCase):
    """PlagiarismChecker 完整单元测试"""

    def setUp(self):
        # 创建临时同义词表文件
        self.synonyms_file = tempfile.NamedTemporaryFile(
            mode='w+',
            delete=False,
            encoding='utf-8',
            suffix='.txt'
        )
        self.synonyms_file.write("周天,星期天\n天气晴朗,天气晴\nML,机器学习")
        self.synonyms_file.close()

        # 初始化带同义词表的检查器
        self.checker_with_syn = PlagiarismChecker()
        self.checker_with_syn.synonyms = self.checker_with_syn._load_synonyms(self.synonyms_file.name)

        # 初始化无同义词表的检查器
        self.checker_no_syn = PlagiarismChecker()
        self.checker_no_syn.synonyms = {}

    def tearDown(self):
        """清理临时文件"""
        try:
            os.unlink(self.synonyms_file.name)
        except:
            pass

    def test_load_synonyms(self):
        """测试同义词表加载功能"""
        self.assertEqual(self.checker_with_syn.synonyms.get("周天"), "星期天")
        self.assertIsNone(self.checker_with_syn.synonyms.get("不存在的词"))

    def test_preprocess_stopwords(self):
        """测试停用词过滤功能"""
        test_text = "这是一个的测试文本，包含无效的停用词！"
        processed = self.checker_no_syn.preprocess(test_text).split()
        self.assertNotIn("是", processed)

    def test_preprocess_synonym_replacement(self):
        """测试同义词替换功能"""
        test_text = "周天天气晴朗，学习ML"
        processed = self.checker_with_syn.preprocess(test_text)
        self.assertIn("星期天", processed)

    def test_preprocess_special_characters(self):
        """测试特殊字符过滤功能"""
        test_text = "Hello! 这是@带有特殊#字符的文本￥%……&"
        processed = self.checker_no_syn.preprocess(test_text)
        self.assertNotIn("Hello", processed)

    def test_identical_text_similarity(self):
        """测试完全相同的文本"""
        text = "机器学习需要大量数据训练模型"
        similarity = self.checker_no_syn.calculate_similarity(text, text)
        self.assertAlmostEqual(similarity, 1.0, delta=0.01)

    def test_different_text_similarity(self):
        """测试完全不相关的文本"""
        text_a = "深度学习依赖神经网络结构"
        text_b = "数据库管理需要SQL语言技能"
        similarity = self.checker_no_syn.calculate_similarity(text_a, text_b)
        self.assertLess(similarity, 0.2)

    def test_synonym_impact_on_similarity(self):
        """测试同义词对相似度的影响"""
        original = "周天学习ML课程"
        paraphrased = "星期天研究机器学习教程"
        processed_orig = self.checker_with_syn.preprocess(original)
        processed_para = self.checker_with_syn.preprocess(paraphrased)
        similarity = self.checker_with_syn.calculate_similarity(processed_orig, processed_para)
        self.assertGreaterEqual(similarity, 0.9)

    def test_empty_input_handling(self):
        """测试空输入处理"""
        # 空文本预处理
        processed = self.checker_no_syn.preprocess("")
        self.assertEqual(processed, "")

        # 空文本相似度计算
        similarity = self.checker_no_syn.calculate_similarity("", "")
        self.assertTrue(np.isnan(similarity))  # 根据实际逻辑调整

    def test_invalid_synonyms_file(self):
        """测试无效同义词表路径处理"""
        with self.assertLogs(level='WARNING') as log:
            synonyms = self.checker_no_syn._load_synonyms("non_existent_file.txt")
            self.assertEqual(synonyms, {})
            self.assertIn("同义词表加载失败", log.output[0])

    def test_large_text_processing(self):
        """测试长文本处理能力（性能+内存）"""
        long_text = "大数据分析" * 5000  # 生成50KB文本
        processed = self.checker_no_syn.preprocess(long_text)
        self.assertLessEqual(len(processed.split()), 10000)  # 验证分词数量合理

    def test_order_preservation(self):
        """测试预处理后的词序保留"""
        text = "机器学习需要先学习数学基础"
        processed = self.checker_no_syn.preprocess(text)
        self.assertTrue(processed.startswith("机器学习 需要 学习 数学 基础"))

    def test_cosine_similarity_edge_cases(self):
        """测试余弦相似度边界条件"""
        # 完全正交向量
        text_a = "apple orange banana"
        text_b = "car truck plane"
        similarity = self.checker_no_syn.calculate_similarity(text_a, text_b)
        self.assertEqual(similarity, 0)

        # 零向量处理
        similarity = self.checker_no_syn.calculate_similarity("", "非空文本")
        self.assertTrue(np.isnan(similarity))


class TestMainFunction(unittest.TestCase):
    """main函数命令行测试"""

    def test_main_output(self):
        """测试完整的命令行工作流"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试文件
            orig_path = os.path.join(tmpdir, "orig.txt")
            plag_path = os.path.join(tmpdir, "plag.txt")
            output_path = os.path.join(tmpdir, "result.txt")

            with open(orig_path, 'w', encoding='utf-8') as f:
                f.write("机器学习需要数据")
            with open(plag_path, 'w', encoding='utf-8') as f:
                f.write("机器学习需要数据")

            # 模拟命令行参数
            with patch('sys.argv', ['main.py', orig_path, plag_path, output_path]):
                from main import main
                main()

            # 验证输出结果
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn("100.00%", content)

    def test_missing_arguments(self):
        """测试缺少命令行参数时的报错"""
        with patch('sys.argv', ['main.py']), \
                self.assertRaises(SystemExit) as cm, \
                self.assertLogs(level='ERROR') as log:
            from main import main
            main()

        self.assertEqual(cm.exception.code, 2)
        self.assertIn("缺少必要参数", log.output[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)

    # 运行全部测试（显示详细结果）
     # python -m unittest test.py -v

    # 运行单个测试类
     # python -m unittest test.py.TestPlagiarismChecker -v

    # 运行单个测试用例
   # python -m unittest test.py.TestPlagiarismChecker.test_large_text_processing -v