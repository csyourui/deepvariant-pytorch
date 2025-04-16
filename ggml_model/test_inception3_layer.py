import argparse
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from pytorch_model.inception import Inception3

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class InceptionShapeTracker(Inception3):
    """
    扩展Inception3类，用于跟踪和打印每一层的输出形状和计算结果
    """

    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logger.info(f"Input shape: {x.shape}")
        if self.verbose:
            logger.info(f"Input values (sample): {self._format_tensor_output(x)}")

        # Conv2d_1a_3x3
        x = self.Conv2d_1a_3x3(x)
        logger.info(f"Conv2d_1a_3x3 output shape: {x.shape}")
        if self.verbose:
            logger.info(
                f"Conv2d_1a_3x3 values (sample): {self._format_tensor_output(x)}"
            )

        # Conv2d_2a_3x3
        x = self.Conv2d_2a_3x3(x)
        logger.info(f"Conv2d_2a_3x3 output shape: {x.shape}")
        if self.verbose:
            logger.info(
                f"Conv2d_2a_3x3 values (sample): {self._format_tensor_output(x)}"
            )

        # Conv2d_2b_3x3
        x = self.Conv2d_2b_3x3(x)
        logger.info(f"Conv2d_2b_3x3 output shape: {x.shape}")
        if self.verbose:
            logger.info(
                f"Conv2d_2b_3x3 values (sample): {self._format_tensor_output(x)}"
            )

        # maxpool1
        x = self.maxpool1(x)
        logger.info(f"maxpool1 output shape: {x.shape}")
        if self.verbose:
            logger.info(f"maxpool1 values (sample): {self._format_tensor_output(x)}")

        # Conv2d_3b_1x1
        x = self.Conv2d_3b_1x1(x)
        logger.info(f"Conv2d_3b_1x1 output shape: {x.shape}")
        if self.verbose:
            logger.info(
                f"Conv2d_3b_1x1 values (sample): {self._format_tensor_output(x)}"
            )

        # Conv2d_4a_3x3
        x = self.Conv2d_4a_3x3(x)
        logger.info(f"Conv2d_4a_3x3 output shape: {x.shape}")
        if self.verbose:
            logger.info(
                f"Conv2d_4a_3x3 values (sample): {self._format_tensor_output(x)}"
            )

        # maxpool2
        x = self.maxpool2(x)
        logger.info(f"maxpool2 output shape: {x.shape}")
        if self.verbose:
            logger.info(f"maxpool2 values (sample): {self._format_tensor_output(x)}")

        # Mixed_5b
        x = self.Mixed_5b(x)
        logger.info(f"Mixed_5b output shape: {x.shape}")
        if self.verbose:
            logger.info(f"Mixed_5b values (sample): {self._format_tensor_output(x)}")

        # Mixed_5c
        x = self.Mixed_5c(x)
        logger.info(f"Mixed_5c output shape: {x.shape}")
        if self.verbose:
            logger.info(f"Mixed_5c values (sample): {self._format_tensor_output(x)}")

        # Mixed_5d
        x = self.Mixed_5d(x)
        logger.info(f"Mixed_5d output shape: {x.shape}")
        if self.verbose:
            logger.info(f"Mixed_5d values (sample): {self._format_tensor_output(x)}")

        # Mixed_6a
        x = self.Mixed_6a(x)
        logger.info(f"Mixed_6a output shape: {x.shape}")
        if self.verbose:
            logger.info(f"Mixed_6a values (sample): {self._format_tensor_output(x)}")

        # Mixed_6b
        x = self.Mixed_6b(x)
        logger.info(f"Mixed_6b output shape: {x.shape}")
        if self.verbose:
            logger.info(f"Mixed_6b values (sample): {self._format_tensor_output(x)}")

        # Mixed_6c
        x = self.Mixed_6c(x)
        logger.info(f"Mixed_6c output shape: {x.shape}")
        if self.verbose:
            logger.info(f"Mixed_6c values (sample): {self._format_tensor_output(x)}")

        # Mixed_6d
        x = self.Mixed_6d(x)
        logger.info(f"Mixed_6d output shape: {x.shape}")
        if self.verbose:
            logger.info(f"Mixed_6d values (sample): {self._format_tensor_output(x)}")

        # Mixed_6e
        x = self.Mixed_6e(x)
        logger.info(f"Mixed_6e output shape: {x.shape}")
        if self.verbose:
            logger.info(f"Mixed_6e values (sample): {self._format_tensor_output(x)}")

        # AuxLogits
        aux = None
        if self.AuxLogits is not None and self.training:
            aux = self.AuxLogits(x)
            logger.info(f"AuxLogits output shape: {aux.shape}")
            if self.verbose:
                logger.info(
                    f"AuxLogits values (sample): {self._format_tensor_output(aux)}"
                )

        # Mixed_7a
        x = self.Mixed_7a(x)
        logger.info(f"Mixed_7a output shape: {x.shape}")
        if self.verbose:
            logger.info(f"Mixed_7a values (sample): {self._format_tensor_output(x)}")

        # Mixed_7b
        x = self.Mixed_7b(x)
        logger.info(f"Mixed_7b output shape: {x.shape}")
        if self.verbose:
            logger.info(f"Mixed_7b values (sample): {self._format_tensor_output(x)}")

        # Mixed_7c
        x = self.Mixed_7c(x)
        logger.info(f"Mixed_7c output shape: {x.shape}")
        if self.verbose:
            logger.info(f"Mixed_7c values (sample): {self._format_tensor_output(x)}")

        # avgpool
        x = self.avgpool(x)
        logger.info(f"avgpool output shape: {x.shape}")
        if self.verbose:
            logger.info(f"avgpool values (sample): {self._format_tensor_output(x)}")

        # dropout
        x = self.dropout(x)
        logger.info(f"dropout output shape: {x.shape}")
        if self.verbose:
            logger.info(f"dropout values (sample): {self._format_tensor_output(x)}")

        # flatten
        x = torch.flatten(x, 1)
        logger.info(f"flatten output shape: {x.shape}")
        if self.verbose:
            logger.info(f"flatten values (sample): {self._format_tensor_output(x)}")

        # fc
        x = self.fc(x)
        logger.info(f"fc output shape: {x.shape}")
        if self.verbose:
            logger.info(f"fc values (sample): {self._format_tensor_output(x)}")

        return x, aux

    def __init__(self, *args, verbose=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    def _format_tensor_output(self, tensor: torch.Tensor) -> str:
        """将张量格式化为可读性更好的字符串表示"""
        # 将张量转为numpy以便格式化
        tensor_np = tensor.detach().cpu().numpy()

        # 对于大张量，只显示少量元素
        if tensor.numel() > 10:
            # 如果是4D张量(B, C, H, W)，显示第一个样本的第一个通道的前10个元素
            if len(tensor.shape) == 4:
                # 获取第一个样本的第一个通道并展平
                first_channel = tensor_np[0, 0].flatten()
                # 取前10个元素
                sample = first_channel[:10]
                return f"CH1[:9]\t: {np.array2string(sample, precision=4, suppress_small=True)}, \nmin: {tensor_np.min():.4f}, max: {tensor_np.max():.4f}, mean: {tensor_np.mean():.4f} \n"
            # 对于1D或2D张量，显示前10个元素
            else:
                sample = tensor_np.flatten()[:10]
                return f"[:9]\t: {np.array2string(sample, precision=4, suppress_small=True)}, \nmin: {tensor_np.min():.4f}, max: {tensor_np.max():.4f}, mean: {tensor_np.mean():.4f} \n"
        else:
            # 小张量完整显示
            return f"{np.array2string(tensor_np, precision=4, suppress_small=True)} \n"

    def test_conv2d_1a_layer(self, x: torch.Tensor) -> dict:
        """
        详细测试Conv2d_1a_3x3层的卷积和batch_norm计算过程

        Args:
            x: 输入张量

        Returns:
            包含各计算阶段结果的字典
        """
        results = {}

        # 保存输入
        results["input"] = x.clone()

        # 获取Conv2d_1a_3x3层
        conv_layer = self.Conv2d_1a_3x3.conv
        bn_layer = self.Conv2d_1a_3x3.bn

        # 执行卷积
        conv_output = conv_layer(x)
        results["conv_output"] = conv_output.clone()

        # 获取BatchNorm参数
        running_mean = bn_layer.running_mean.clone()
        running_var = bn_layer.running_var.clone()
        weight = bn_layer.weight.clone()
        bias = bn_layer.bias.clone()
        eps = bn_layer.eps

        results["bn_running_mean"] = running_mean.clone()
        results["bn_running_var"] = running_var.clone()
        results["bn_weight"] = weight.clone()
        results["bn_bias"] = bias.clone()
        results["bn_eps"] = eps

        # BatchNorm计算中间结果
        # x_norm = (x - running_mean) / sqrt(running_var + eps)
        # output = weight * x_norm + bias

        # 手动计算BatchNorm中间结果
        x_centered = conv_output - running_mean.view(1, -1, 1, 1)
        results["bn_centered"] = x_centered.clone()

        x_inv_std = 1.0 / torch.sqrt(running_var.view(1, -1, 1, 1) + eps)
        results["bn_inv_std"] = x_inv_std.clone()

        x_normalized = x_centered * x_inv_std
        results["bn_normalized"] = x_normalized.clone()

        bn_output = x_normalized * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
        results["bn_output"] = bn_output.clone()

        # 实际的BatchNorm输出
        actual_bn_output = bn_layer(conv_output)
        results["actual_bn_output"] = actual_bn_output.clone()

        # 检查我们的手动计算与PyTorch实现是否一致
        is_close = torch.allclose(bn_output, actual_bn_output, rtol=1e-5, atol=1e-5)
        results["calculation_matches"] = is_close

        relu = torch.nn.ReLU(inplace=True)
        relu_output = relu(bn_output)
        results["relu_output"] = relu_output.clone()

        return results


def tensor_to_serializable(tensor):
    """将张量转换为可序列化的格式"""
    if isinstance(tensor, torch.Tensor):
        # 转换为numpy，再转为列表
        return tensor.detach().cpu().numpy().tolist()
    return tensor


def save_results_to_yaml(results: Dict[str, Any], filename: str):
    """
    保存结果到YAML文件

    Args:
        results: 测试结果字典
        filename: 输出文件名
    """
    # 创建可序列化的结果副本
    serializable_results = {}

    # 处理每个键值对
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            # 对于张量，我们只保存形状和摘要统计信息
            tensor_np = value.detach().cpu().numpy()
            serializable_results[key] = {
                "shape": value.shape,
                "min": float(tensor_np.min()),
                "max": float(tensor_np.max()),
                "mean": float(tensor_np.mean()),
                "std": float(tensor_np.std()),
                # 对于小张量或示例值，保存部分内容
                "sample": tensor_np.flatten()[:20].tolist(),
            }
        else:
            # 其他类型直接保存
            serializable_results[key] = value

    # 写入YAML文件
    with open(filename, "w") as f:
        yaml.dump(serializable_results, f, default_flow_style=False)

    logger.info(f"结果已保存到: {filename}")


def test_inception_conv2d_1a_layer(args):
    """
    单元测试函数，专门测试Inception模型的Conv2d_1a_3x3层
    """
    # 设置设备
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    logger.info(f"使用设备: {device}")

    # 创建输入数据
    input_shape = (args.batch_size, args.channels, args.height, args.width)
    input_tensor = 0.5 * torch.ones(input_shape).to(device)
    logger.info(f"创建值为0.1的输入张量，形状为 {input_shape}")

    # 创建模型
    model_shape = (args.height, args.width, args.channels)
    model = InceptionShapeTracker(
        input_shape=model_shape,
        num_classes=args.num_classes,
        aux_logits=args.aux_logits,
        transform_input=args.transform_input,
        verbose=False,
    ).to(device)

    # 加载权重文件（如果有）
    if args.weights_path and os.path.exists(args.weights_path):
        loaded_model = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(loaded_model.state_dict())
        logger.info(f"成功加载权重文件: {args.weights_path}")
        weights_info = "with_pretrained_weights"
    else:
        if args.weights_path:
            logger.warning(f"指定的权重文件不存在: {args.weights_path}")
        logger.info("使用随机初始化的权重")
        weights_info = "random_weights"

    # 设置为评估模式
    model.eval()

    logger.info("\n" + "=" * 80)
    logger.info("开始测试 Conv2d_1a_3x3 层")
    logger.info("=" * 80 + "\n")

    # 执行Conv2d_1a_3x3层测试
    with torch.no_grad():
        results = model.test_conv2d_1a_layer(input_tensor)

    # 创建用于YAML的结果字典
    yaml_results = {
        "test_info": {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "input_shape": input_shape,
            "model_config": {
                "height": args.height,
                "width": args.width,
                "channels": args.channels,
                "num_classes": args.num_classes,
                "weights": weights_info,
            },
        },
        "conv2d_1a_test": {},
    }

    # 打印测试结果并添加到YAML结果中
    logger.info(f"输入张量形状: {results['input'].shape}")
    logger.info(f"输入张量示例: {model._format_tensor_output(results['input'])}")
    yaml_results["conv2d_1a_test"]["input"] = {
        "shape": list(results["input"].shape),
        "sample": results["input"][0, 0, 0, :10].cpu().numpy().tolist(),
    }

    logger.info("\n" + "-" * 40 + "卷积层结果" + "-" * 40)
    logger.info(f"卷积输出形状: {results['conv_output'].shape}")
    logger.info(f"卷积输出示例: {model._format_tensor_output(results['conv_output'])}")
    yaml_results["conv2d_1a_test"]["conv_output"] = {
        "shape": list(results["conv_output"].shape),
        "sample": results["conv_output"][0, 0, 0, :10].cpu().numpy().tolist(),
        "stats": {
            "min": float(results["conv_output"].min().item()),
            "max": float(results["conv_output"].max().item()),
            "mean": float(results["conv_output"].mean().item()),
            "std": float(results["conv_output"].std().item()),
        },
    }

    logger.info("\n" + "-" * 40 + "BatchNorm参数" + "-" * 40)
    bn_mean = results["bn_running_mean"].cpu().numpy()
    bn_var = results["bn_running_var"].cpu().numpy()
    bn_weight = results["bn_weight"].cpu().numpy()
    bn_bias = results["bn_bias"].cpu().numpy()

    logger.info(f"Running Mean: {np.array2string(bn_mean, precision=4)}")
    logger.info(f"Running Var: {np.array2string(bn_var, precision=4)}")
    logger.info(f"Weight: {np.array2string(bn_weight, precision=4)}")
    logger.info(f"Bias: {np.array2string(bn_bias, precision=4)}")
    logger.info(f"Epsilon: {results['bn_eps']}")

    yaml_results["conv2d_1a_test"]["bn_params"] = {
        "running_mean": bn_mean.tolist(),
        "running_var": bn_var.tolist(),
        "weight": bn_weight.tolist(),
        "bias": bn_bias.tolist(),
        "epsilon": results["bn_eps"],
    }

    logger.info("\n" + "-" * 40 + "BatchNorm计算过程" + "-" * 40)
    # 为每个BatchNorm计算步骤添加YAML数据
    for step_name, step_key in [
        ("中心化", "bn_centered"),
        ("标准差倒数", "bn_inv_std"),
        ("归一化", "bn_normalized"),
        ("缩放和偏移", "bn_output"),
    ]:
        step_num = ["中心化", "标准差倒数", "归一化", "缩放和偏移"].index(step_name) + 1
        logger.info(f"{step_num}. {step_name}后形状: {results[step_key].shape}")
        logger.info(f"   示例值: {model._format_tensor_output(results[step_key])}")

        yaml_results["conv2d_1a_test"][step_key] = {
            "shape": list(results[step_key].shape),
            "sample": results[step_key][0, 0, 0, :10].cpu().numpy().tolist(),
            "stats": {
                "min": float(results[step_key].min().item()),
                "max": float(results[step_key].max().item()),
                "mean": float(results[step_key].mean().item()),
                "std": float(results[step_key].std().item()),
            },
        }

    logger.info("\n" + "-" * 40 + "最终结果" + "-" * 40)
    logger.info(f"PyTorch BatchNorm输出形状: {results['actual_bn_output'].shape}")
    logger.info(
        f"PyTorch BatchNorm输出示例: {model._format_tensor_output(results['actual_bn_output'])}"
    )

    yaml_results["conv2d_1a_test"]["actual_bn_output"] = {
        "shape": list(results["actual_bn_output"].shape),
        "sample": results["actual_bn_output"][0, 0, 0, :10].cpu().numpy().tolist(),
        "stats": {
            "min": float(results["actual_bn_output"].min().item()),
            "max": float(results["actual_bn_output"].max().item()),
            "mean": float(results["actual_bn_output"].mean().item()),
            "std": float(results["actual_bn_output"].std().item()),
        },
    }

    logger.info(f"\n手动计算与PyTorch实现是否一致: {results['calculation_matches']}")
    yaml_results["conv2d_1a_test"]["calculation_matches"] = results[
        "calculation_matches"
    ]

    if not results["calculation_matches"]:
        # 如果结果不一致，计算差异
        diff = torch.abs(results["bn_output"] - results["actual_bn_output"])
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        logger.info(f"最大差异: {max_diff:.6e}")
        logger.info(f"平均差异: {mean_diff:.6e}")

        yaml_results["conv2d_1a_test"]["diff"] = {"max": max_diff, "mean": mean_diff}

    logger.info("\n" + "-" * 40 + "ReLU激活" + "-" * 40)
    logger.info(f"ReLU输出形状: {results['relu_output'].shape}")
    logger.info(f"ReLU输出示例: {model._format_tensor_output(results['relu_output'])}")
    yaml_results["conv2d_1a_test"]["relu_output"] = {
        "shape": list(results["relu_output"].shape),
        "sample": results["relu_output"][0, 0, 0, :10].cpu().numpy().tolist(),
        "stats": {
            "min": float(results["relu_output"].min().item()),
            "max": float(results["relu_output"].max().item()),
            "mean": float(results["relu_output"].mean().item()),
            "std": float(results["relu_output"].std().item()),
        },
    }

    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"conv2d_1a_test_{weights_info}_{timestamp}.yaml"
    )

    # 保存YAML结果
    with open(output_file, "w") as f:
        yaml.dump(yaml_results, f, default_flow_style=False)

    logger.info(f"\n测试结果已保存到YAML文件: {output_file}")

    logger.info("\n" + "=" * 80)
    logger.info("Conv2d_1a_3x3 层测试完成")
    logger.info("=" * 80)


def run_inception_shape_test(args):
    """
    运行Inception形状测试，记录并打印每一层的输出形状和计算结果
    """
    # 设置设备
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    logger.info(f"Using device: {device}")

    # 创建输入数据 - 全1张量
    input_shape = (args.batch_size, args.channels, args.height, args.width)
    input_tensor = 0.1 * torch.ones(input_shape).to(device)
    logger.info(f"Created all-ones input tensor with shape {input_shape}")

    # 设置是否加载权重（如果提供了权重路径）
    load_weights = False
    if args.weights_path and os.path.exists(args.weights_path):
        load_weights = True
        logger.info(f"将加载权重文件: {args.weights_path}")
        weights_info = "with_pretrained_weights"
    else:
        if args.weights_path:
            logger.warning(f"指定的权重文件不存在: {args.weights_path}")
        logger.info("将使用随机初始化的权重")
        weights_info = "random_weights"

    # 创建模型
    model_shape = (args.height, args.width, args.channels)
    model = InceptionShapeTracker(
        input_shape=model_shape,
        num_classes=args.num_classes,
        aux_logits=args.aux_logits,
        transform_input=args.transform_input,
        verbose=load_weights,  # 只有当加载了权重时才打印详细信息
    ).to(device)

    # 加载权重文件（如果有）
    if load_weights:
        loaded_model = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(loaded_model.state_dict())
        logger.info("成功加载权重文件")

    # 设置为评估模式
    model.eval()

    # 用于收集所有层输出形状的字典
    shapes_data = {
        "test_info": {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "input_shape": input_shape,
            "model_config": {
                "height": args.height,
                "width": args.width,
                "channels": args.channels,
                "num_classes": args.num_classes,
                "weights": weights_info,
            },
        },
        "layer_shapes": {},
    }

    # 为了收集所有层的输出形状，我们需要在前向传播过程中记录
    def record_layer_output(name, tensor):
        shapes_data["layer_shapes"][name] = {
            "shape": list(tensor.shape),
            "stats": {
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()),
            },
        }

    # 修改一个临时的_forward方法来记录形状
    original_forward = model._forward

    def capturing_forward(x):
        record_layer_output("input", x)

        # Conv2d_1a_3x3
        x = model.Conv2d_1a_3x3(x)
        record_layer_output("Conv2d_1a_3x3", x)

        # Conv2d_2a_3x3
        x = model.Conv2d_2a_3x3(x)
        record_layer_output("Conv2d_2a_3x3", x)

        # Conv2d_2b_3x3
        x = model.Conv2d_2b_3x3(x)
        record_layer_output("Conv2d_2b_3x3", x)

        # maxpool1
        x = model.maxpool1(x)
        record_layer_output("maxpool1", x)

        # Conv2d_3b_1x1
        x = model.Conv2d_3b_1x1(x)
        record_layer_output("Conv2d_3b_1x1", x)

        # Conv2d_4a_3x3
        x = model.Conv2d_4a_3x3(x)
        record_layer_output("Conv2d_4a_3x3", x)

        # # Conv2d_4a_3x3
        # conv_4a_3x3_layer = model.Conv2d_4a_3x3.conv
        # bn_4a_3x3_layer = model.Conv2d_4a_3x3.bn
        # x = conv_4a_3x3_layer(x)
        # record_layer_output('Conv2d_4a_3x3_conv', x)
        # x = bn_4a_3x3_layer(x)
        # record_layer_output('Conv2d_4a_3x3_bn', x)

        # maxpool2
        x = model.maxpool2(x)
        record_layer_output("maxpool2", x)

        # Mixed_5b
        x = model.Mixed_5b(x)
        record_layer_output("Mixed_5b", x)

        # Mixed_5c
        x = model.Mixed_5c(x)
        record_layer_output("Mixed_5c", x)

        # Mixed_5d
        x = model.Mixed_5d(x)
        record_layer_output("Mixed_5d", x)

        # Mixed_6a
        x = model.Mixed_6a(x)
        record_layer_output("Mixed_6a", x)

        # Mixed_6b
        x = model.Mixed_6b(x)
        record_layer_output("Mixed_6b", x)

        # Mixed_6c
        x = model.Mixed_6c(x)
        record_layer_output("Mixed_6c", x)

        # Mixed_6d
        x = model.Mixed_6d(x)
        record_layer_output("Mixed_6d", x)

        # Mixed_6e
        x = model.Mixed_6e(x)
        record_layer_output("Mixed_6e", x)

        # AuxLogits
        aux = None
        if model.AuxLogits is not None and model.training:
            aux = model.AuxLogits(x)
            record_layer_output("AuxLogits", aux)

        # Mixed_7a
        x = model.Mixed_7a(x)
        record_layer_output("Mixed_7a", x)

        # Mixed_7b
        x = model.Mixed_7b(x)
        record_layer_output("Mixed_7b", x)

        # Mixed_7c
        x = model.Mixed_7c(x)
        record_layer_output("Mixed_7c", x)

        # avgpool
        x = model.avgpool(x)
        record_layer_output("avgpool", x)

        # dropout
        x = model.dropout(x)
        record_layer_output("dropout", x)

        # flatten
        x = torch.flatten(x, 1)
        record_layer_output("flatten", x)

        # fc
        x = model.fc(x)
        record_layer_output("fc", x)

        return x, aux

    # 替换_forward方法
    model._forward = capturing_forward

    # 运行推理
    logger.info("\n" + "=" * 50)
    if load_weights:
        logger.info("RUNNING INCEPTION SHAPE TRACKING WITH LOADED WEIGHTS")
    else:
        logger.info("RUNNING INCEPTION SHAPE TRACKING WITH RANDOM WEIGHTS")
    logger.info("=" * 50 + "\n")

    with torch.no_grad():
        outputs = model(input_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # 如果有辅助输出，取主输出
        softmax_output = torch.nn.functional.softmax(outputs, dim=1)

        # 添加softmax结果到YAML数据
        shapes_data["softmax_output"] = {
            "shape": list(softmax_output.shape),
            "values": softmax_output[0].cpu().numpy().tolist(),
        }

        # 打印 softmax 结果
        logger.info("\nSoftmax 概率分布:")
        for i in range(softmax_output.shape[1]):
            prob = softmax_output[0][i].item()
            logger.info(f"类别 {i}: {prob:.6f} ({prob * 100:.2f}%)")

    # 恢复原始_forward方法
    model._forward = original_forward

    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"inception_shapes_{weights_info}_{timestamp}.yaml"
    )

    # 保存YAML结果
    with open(output_file, "w") as f:
        yaml.dump(shapes_data, f, default_flow_style=False)

    logger.info(f"\n所有层形状数据已保存到YAML文件: {output_file}")

    logger.info("\n" + "=" * 50)
    logger.info("SHAPE TRACKING COMPLETED")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inception3模型形状追踪工具")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--channels", type=int, default=7, help="输入通道数")
    parser.add_argument("--height", type=int, default=100, help="输入高度")
    parser.add_argument("--width", type=int, default=221, help="输入宽度")
    parser.add_argument("--num_classes", type=int, default=3, help="类别数量")
    parser.add_argument(
        "--aux_logits", type=bool, default=False, help="是否使用辅助输出"
    )
    parser.add_argument(
        "--transform_input", type=bool, default=False, help="是否转换输入"
    )
    parser.add_argument(
        "--weights_path", type=str, default=None, help="预训练权重文件路径"
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default="full",
        help="测试模式：full(全模型) 或 conv2d_1a(只测试第一层)",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="输出结果保存目录")

    args = parser.parse_args()

    # 如果指定了输出目录则使用，否则使用默认目录
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    if args.test_mode.lower() == "conv2d_1a":
        test_inception_conv2d_1a_layer(args)
    else:
        run_inception_shape_test(args)
