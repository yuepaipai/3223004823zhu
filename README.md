# 中文论文查重系统

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)![License](https://img.shields.io/badge/License-MIT-green)

一个基于语义相似度的中文论文查重系统，采用TF-IDF向量化和余弦相似度算法，支持同义词替换和停用词过滤。
## 使用
Way1：  在cmd输入： python main.py "论文地址" "抄袭版论文地址" "查看查重率TXT文档"
Way2:   把原版论文和抄袭版论文分别复制粘贴在“origin.txt”和“origin_add.txt”
## 功能特性

- ✅ **多级文本预处理**：字符清洗 + Jieba精确分词 + 同义词替换 + 停用词过滤
- 🚀 **高效相似度计算**：动态特征裁剪 + Bigram语义捕捉
- 📦 **开箱即用**：命令行接口支持批量处理
- 🔧 **灵活扩展**：支持动态加载同义词表

## 技术栈

- **核心算法**：TF-IDF + 余弦相似度
- **分词工具**：Jieba中文分词
- **向量化库**：scikit-learn
- **性能优化**：Joblib并行计算 + LRU缓存

## 安装指南

### 环境要求
- Python 3.8+
- pip包管理器

### 安装依赖(首次使用需要安装Jieba库）
```bash
python -m jieba.downloader all
