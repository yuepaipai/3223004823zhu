import cProfile
import pstats
from main import PlagiarismChecker  # 导入主代码中的类


def test_performance():
    # 初始化查重器
    checker = PlagiarismChecker()

    # 测试数据（可替换为实际文件路径）
    orig_text = "深度学习依赖神经网络结构"
    plag_text = "数据库管理需要SQL语言技能"

    # 执行预处理和相似度计算（多次调用以放大性能问题）
    for _ in range(10):  # 循环执行以增加分析准确性
        processed_orig = checker.preprocess(orig_text)
        processed_plag = checker.preprocess(plag_text)
        similarity = checker.calculate_similarity(processed_orig, processed_plag)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    test_performance()

    profiler.disable()
    profiler.dump_stats("performance.prof")

    # 打印耗时最高的10个函数
    stats = pstats.Stats("performance.prof")
    stats.sort_stats(pstats.SortKey.TIME).print_stats(10)

    # 生成可视化报告（需安装 snakeviz）
    import subprocess

    subprocess.run(["snakeviz", "performance.prof"])

    # python profile_text.py