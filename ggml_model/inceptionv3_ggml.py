import os
import sys

import numpy as np

# 添加库目录到Python路径
bindings_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "./build/python"
)
sys.path.append(bindings_path)

# 导入C++绑定库
try:
    import inception3_bindings as inc3
except ImportError as e:
    print(f"Loading inception3_bindings failed: {e}")
    print(
        "Please ensure the Python bindings are compiled and the path is set correctly."
    )
    sys.exit(1)


class InceptionV3GGML:
    """
    InceptionV3 GGML模型的Python包装类，提供简洁的NumPy接口。
    """

    def __init__(self, model_path: str):
        """
        初始化InceptionV3 GGML模型

        Args:
            model_path: GGUF模型文件的路径
        """
        self.model_path = model_path

        # 使用Python绑定加载模型
        try:
            # 创建模型实例
            self.model_id = inc3.create_model()
            self.is_loaded = inc3.load_model(self.model_id, model_path)
            if not self.is_loaded:
                raise RuntimeError("Loading model failed")
        except Exception as e:
            raise RuntimeError(f"Loading model failed: {e}")

    def __del__(self):
        """析构函数，确保资源被释放"""
        self.free()

    def free(self):
        """释放模型资源"""
        if self.is_loaded:
            inc3.free_model(self.model_id)
            self.is_loaded = False

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源被释放"""
        self.free()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        执行前向推理

        Args:
            x: 输入NumPy数组

        Returns:
            推理结果作为NumPy数组
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        if x.ndim != 4:
            raise ValueError(f"Invalid input shape: {x.shape}. Expected 4D array.")
        if x.shape[1:] != (7, 100, 221):
            raise ValueError(
                f"Invalid input shape: {x.shape}. Expected shape (batch_size, 7, 100, 221)."
            )

        # 确保输入是正确的类型
        if x.dtype != np.float32:
            x_np = x.astype(np.float32)
        else:
            x_np = x

        # 调用绑定库的推理方法
        try:
            output = inc3.infer(self.model_id, x_np)
            return output
        except Exception as e:
            raise RuntimeError(f"inference failed: {e}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    @property
    def device(self):
        """兼容接口，返回CPU设备"""
        return "cpu"
