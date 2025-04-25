#!/usr/bin/env python3
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# 添加库目录到Python路径
bindings_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../build/python"
)
if not os.path.exists(bindings_path):
    print(f"错误: 绑定库路径不存在: {bindings_path}")
    sys.exit(1)
sys.path.append(bindings_path)

# 导入C++绑定库
try:
    import inception3_bindings as inc3
except ImportError as e:
    print(f"无法导入inception3_bindings: {e}")
    print("请确保已编译Python绑定库，并设置正确的Python路径")
    sys.exit(1)


def test_single_inference(model_path, batch_size=1, verbose=True):
    """测试单次推理"""
    if verbose:
        print(f"\n--- 测试单次推理 (批次大小={batch_size}) ---")

    # 创建模型实例
    model_id = inc3.create_model()

    # 加载模型
    if verbose:
        print(f"加载模型: {model_path}")
    success = inc3.load_model(model_id, model_path)
    if not success:
        print("模型加载失败！")
        return False

    # 创建随机输入数据
    # input_data = np.random.rand(batch_size, 7, 100, 221).astype(np.float32)
    input_data = np.zeros((batch_size, 7, 100, 221), dtype=np.float32)
    for i in range(batch_size):
        input_data[i, :, :, :] = 0.1 * i

    # 执行推理
    if verbose:
        print("执行推理...")
    start_time = time.time()
    result = inc3.infer(model_id, input_data)
    end_time = time.time()

    if verbose:
        print(f"推理完成，用时: {(end_time - start_time) * 1000:.2f} ms")
        print(f"输出形状: {result.shape}")
        print(f"输出示例: {result[0][:3]}...")  # 显示第一个样本的前3个分类结果
        for i in range(result.shape[0]):
            # 格式化输出结果，保留4位小数
            formatted_result = [f"{value:.6f}" for value in result[i]]
            print(f"第{i}个:\t {formatted_result}")

    # 释放资源
    inc3.free_model(model_id)
    if verbose:
        print("模型资源已释放")

    return True


def test_multi_threading(model_path, num_threads=4, batch_size=1):
    """测试多线程并行推理"""
    print(f"\n--- 测试多线程并行推理 ({num_threads}线程) ---")

    # 创建线程池
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交多个推理任务
        futures = [
            executor.submit(test_single_inference, model_path, batch_size, False)
            for _ in range(num_threads)
        ]

        # 等待所有任务完成
        results = [future.result() for future in futures]

    success_count = sum(results)
    print(f"多线程测试完成: {success_count}/{num_threads} 成功")

    # 检查剩余模型数量
    remaining_models = inc3.get_model_count()
    print(f"剩余模型实例数: {remaining_models} (应为0)")

    return success_count == num_threads and remaining_models == 0


def worker_process(model_path, batch_size, process_id):
    """多进程工作函数"""
    print(f"进程 {process_id} 启动")
    result = test_single_inference(model_path, batch_size, False)
    print(f"进程 {process_id} 完成: {'成功' if result else '失败'}")
    return result


def test_multi_processing(model_path, num_processes=2, batch_size=1):
    """测试多进程并行推理"""
    print(f"\n--- 测试多进程并行推理 ({num_processes}进程) ---")

    # 创建进程池
    with mp.Pool(processes=num_processes) as pool:
        # 提交多个推理任务
        results = [
            pool.apply_async(worker_process, (model_path, batch_size, i))
            for i in range(num_processes)
        ]

        # 等待所有任务完成并获取结果
        success_count = sum([result.get() for result in results])

    print(f"多进程测试完成: {success_count}/{num_processes} 成功")

    # 检查当前模型数量（每个进程有自己的内存空间，主进程中应为0）
    remaining_models = inc3.get_model_count()
    print(f"主进程中剩余模型实例数: {remaining_models} (应为0)")

    return success_count == num_processes


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python test_inception3_bindings.py <model_path> [batch_size]")
        sys.exit(1)

    model_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"使用模型: {model_path}")
    print(f"批次大小: {batch_size}")

    # 运行测试
    test_single_inference(model_path, batch_size)
    # test_multi_threading(model_path, num_threads=4, batch_size=batch_size)
    # test_multi_processing(model_path, num_processes=2, batch_size=batch_size)

    print("\n所有测试完成！")


if __name__ == "__main__":
    main()
