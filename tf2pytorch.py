import torch

from pytorch_model.inception import inception_v3 as pt_inception_v3
from tensorflow_model.keras_modeling import inceptionv3 as tf_inception_v3

NUM_CLASSES = 3
DEFAULT_WEIGHT_DECAY = 0.00004
DEFAULT_BACKBONE_DROPOUT_RATE = 0.2

INPUT_SHAPE = (100, 221, 7)

TF_TENSOR_2_PT_TENSOR = {
    "conv2d/kernel:0": "Conv2d_1a_3x3.conv.weight",
    "batch_normalization/beta:0": "Conv2d_1a_3x3.bn.bias",
    "conv2d_1/kernel:0": "Conv2d_2a_3x3.conv.weight",
    "batch_normalization_1/beta:0": "Conv2d_2a_3x3.bn.bias",
    "conv2d_2/kernel:0": "Conv2d_2b_3x3.conv.weight",
    "batch_normalization_2/beta:0": "Conv2d_2b_3x3.bn.bias",
    "conv2d_3/kernel:0": "Conv2d_3b_1x1.conv.weight",
    "batch_normalization_3/beta:0": "Conv2d_3b_1x1.bn.bias",
    "conv2d_4/kernel:0": "Conv2d_4a_3x3.conv.weight",
    "batch_normalization_4/beta:0": "Conv2d_4a_3x3.bn.bias",
    "conv2d_5/kernel:0": "Mixed_5b.branch1x1.conv.weight",
    "batch_normalization_5/beta:0": "Mixed_5b.branch1x1.bn.bias",
    "conv2d_6/kernel:0": "Mixed_5b.branch5x5_1.conv.weight",
    "batch_normalization_6/beta:0": "Mixed_5b.branch5x5_1.bn.bias",
    "conv2d_7/kernel:0": "Mixed_5b.branch5x5_2.conv.weight",
    "batch_normalization_7/beta:0": "Mixed_5b.branch5x5_2.bn.bias",
    "conv2d_8/kernel:0": "Mixed_5b.branch3x3dbl_1.conv.weight",
    "batch_normalization_8/beta:0": "Mixed_5b.branch3x3dbl_1.bn.bias",
    "conv2d_9/kernel:0": "Mixed_5b.branch3x3dbl_2.conv.weight",
    "batch_normalization_9/beta:0": "Mixed_5b.branch3x3dbl_2.bn.bias",
    "conv2d_10/kernel:0": "Mixed_5b.branch3x3dbl_3.conv.weight",
    "batch_normalization_10/beta:0": "Mixed_5b.branch3x3dbl_3.bn.bias",
    "conv2d_11/kernel:0": "Mixed_5b.branch_pool.conv.weight",
    "batch_normalization_11/beta:0": "Mixed_5b.branch_pool.bn.bias",
    "conv2d_12/kernel:0": "Mixed_5c.branch1x1.conv.weight",
    "batch_normalization_12/beta:0": "Mixed_5c.branch1x1.bn.bias",
    "conv2d_13/kernel:0": "Mixed_5c.branch5x5_1.conv.weight",
    "batch_normalization_13/beta:0": "Mixed_5c.branch5x5_1.bn.bias",
    "conv2d_14/kernel:0": "Mixed_5c.branch5x5_2.conv.weight",
    "batch_normalization_14/beta:0": "Mixed_5c.branch5x5_2.bn.bias",
    "conv2d_15/kernel:0": "Mixed_5c.branch3x3dbl_1.conv.weight",
    "batch_normalization_15/beta:0": "Mixed_5c.branch3x3dbl_1.bn.bias",
    "conv2d_16/kernel:0": "Mixed_5c.branch3x3dbl_2.conv.weight",
    "batch_normalization_16/beta:0": "Mixed_5c.branch3x3dbl_2.bn.bias",
    "conv2d_17/kernel:0": "Mixed_5c.branch3x3dbl_3.conv.weight",
    "batch_normalization_17/beta:0": "Mixed_5c.branch3x3dbl_3.bn.bias",
    "conv2d_18/kernel:0": "Mixed_5c.branch_pool.conv.weight",
    "batch_normalization_18/beta:0": "Mixed_5c.branch_pool.bn.bias",
    "conv2d_19/kernel:0": "Mixed_5d.branch1x1.conv.weight",
    "batch_normalization_19/beta:0": "Mixed_5d.branch1x1.bn.bias",
    "conv2d_20/kernel:0": "Mixed_5d.branch5x5_1.conv.weight",
    "batch_normalization_20/beta:0": "Mixed_5d.branch5x5_1.bn.bias",
    "conv2d_21/kernel:0": "Mixed_5d.branch5x5_2.conv.weight",
    "batch_normalization_21/beta:0": "Mixed_5d.branch5x5_2.bn.bias",
    "conv2d_22/kernel:0": "Mixed_5d.branch3x3dbl_1.conv.weight",
    "batch_normalization_22/beta:0": "Mixed_5d.branch3x3dbl_1.bn.bias",
    "conv2d_23/kernel:0": "Mixed_5d.branch3x3dbl_2.conv.weight",
    "batch_normalization_23/beta:0": "Mixed_5d.branch3x3dbl_2.bn.bias",
    "conv2d_24/kernel:0": "Mixed_5d.branch3x3dbl_3.conv.weight",
    "batch_normalization_24/beta:0": "Mixed_5d.branch3x3dbl_3.bn.bias",
    "conv2d_25/kernel:0": "Mixed_5d.branch_pool.conv.weight",
    "batch_normalization_25/beta:0": "Mixed_5d.branch_pool.bn.bias",
    "conv2d_26/kernel:0": "Mixed_6a.branch3x3.conv.weight",
    "batch_normalization_26/beta:0": "Mixed_6a.branch3x3.bn.bias",
    "conv2d_27/kernel:0": "Mixed_6a.branch3x3dbl_1.conv.weight",
    "batch_normalization_27/beta:0": "Mixed_6a.branch3x3dbl_1.bn.bias",
    "conv2d_28/kernel:0": "Mixed_6a.branch3x3dbl_2.conv.weight",
    "batch_normalization_28/beta:0": "Mixed_6a.branch3x3dbl_2.bn.bias",
    "conv2d_29/kernel:0": "Mixed_6a.branch3x3dbl_3.conv.weight",
    "batch_normalization_29/beta:0": "Mixed_6a.branch3x3dbl_3.bn.bias",
    "conv2d_30/kernel:0": "Mixed_6b.branch1x1.conv.weight",
    "batch_normalization_30/beta:0": "Mixed_6b.branch1x1.bn.bias",
    "conv2d_31/kernel:0": "Mixed_6b.branch7x7_1.conv.weight",
    "batch_normalization_31/beta:0": "Mixed_6b.branch7x7_1.bn.bias",
    "conv2d_32/kernel:0": "Mixed_6b.branch7x7_2.conv.weight",
    "batch_normalization_32/beta:0": "Mixed_6b.branch7x7_2.bn.bias",
    "conv2d_33/kernel:0": "Mixed_6b.branch7x7_3.conv.weight",
    "batch_normalization_33/beta:0": "Mixed_6b.branch7x7_3.bn.bias",
    "conv2d_34/kernel:0": "Mixed_6b.branch7x7dbl_1.conv.weight",
    "batch_normalization_34/beta:0": "Mixed_6b.branch7x7dbl_1.bn.bias",
    "conv2d_35/kernel:0": "Mixed_6b.branch7x7dbl_2.conv.weight",
    "batch_normalization_35/beta:0": "Mixed_6b.branch7x7dbl_2.bn.bias",
    "conv2d_36/kernel:0": "Mixed_6b.branch7x7dbl_3.conv.weight",
    "batch_normalization_36/beta:0": "Mixed_6b.branch7x7dbl_3.bn.bias",
    "conv2d_37/kernel:0": "Mixed_6b.branch7x7dbl_4.conv.weight",
    "batch_normalization_37/beta:0": "Mixed_6b.branch7x7dbl_4.bn.bias",
    "conv2d_38/kernel:0": "Mixed_6b.branch7x7dbl_5.conv.weight",
    "batch_normalization_38/beta:0": "Mixed_6b.branch7x7dbl_5.bn.bias",
    "conv2d_39/kernel:0": "Mixed_6b.branch_pool.conv.weight",
    "batch_normalization_39/beta:0": "Mixed_6b.branch_pool.bn.bias",
    "conv2d_40/kernel:0": "Mixed_6c.branch1x1.conv.weight",
    "batch_normalization_40/beta:0": "Mixed_6c.branch1x1.bn.bias",
    "conv2d_41/kernel:0": "Mixed_6c.branch7x7_1.conv.weight",
    "batch_normalization_41/beta:0": "Mixed_6c.branch7x7_1.bn.bias",
    "conv2d_42/kernel:0": "Mixed_6c.branch7x7_2.conv.weight",
    "batch_normalization_42/beta:0": "Mixed_6c.branch7x7_2.bn.bias",
    "conv2d_43/kernel:0": "Mixed_6c.branch7x7_3.conv.weight",
    "batch_normalization_43/beta:0": "Mixed_6c.branch7x7_3.bn.bias",
    "conv2d_44/kernel:0": "Mixed_6c.branch7x7dbl_1.conv.weight",
    "batch_normalization_44/beta:0": "Mixed_6c.branch7x7dbl_1.bn.bias",
    "conv2d_45/kernel:0": "Mixed_6c.branch7x7dbl_2.conv.weight",
    "batch_normalization_45/beta:0": "Mixed_6c.branch7x7dbl_2.bn.bias",
    "conv2d_46/kernel:0": "Mixed_6c.branch7x7dbl_3.conv.weight",
    "batch_normalization_46/beta:0": "Mixed_6c.branch7x7dbl_3.bn.bias",
    "conv2d_47/kernel:0": "Mixed_6c.branch7x7dbl_4.conv.weight",
    "batch_normalization_47/beta:0": "Mixed_6c.branch7x7dbl_4.bn.bias",
    "conv2d_48/kernel:0": "Mixed_6c.branch7x7dbl_5.conv.weight",
    "batch_normalization_48/beta:0": "Mixed_6c.branch7x7dbl_5.bn.bias",
    "conv2d_49/kernel:0": "Mixed_6c.branch_pool.conv.weight",
    "batch_normalization_49/beta:0": "Mixed_6c.branch_pool.bn.bias",
    "conv2d_50/kernel:0": "Mixed_6d.branch1x1.conv.weight",
    "batch_normalization_50/beta:0": "Mixed_6d.branch1x1.bn.bias",
    "conv2d_51/kernel:0": "Mixed_6d.branch7x7_1.conv.weight",
    "batch_normalization_51/beta:0": "Mixed_6d.branch7x7_1.bn.bias",
    "conv2d_52/kernel:0": "Mixed_6d.branch7x7_2.conv.weight",
    "batch_normalization_52/beta:0": "Mixed_6d.branch7x7_2.bn.bias",
    "conv2d_53/kernel:0": "Mixed_6d.branch7x7_3.conv.weight",
    "batch_normalization_53/beta:0": "Mixed_6d.branch7x7_3.bn.bias",
    "conv2d_54/kernel:0": "Mixed_6d.branch7x7dbl_1.conv.weight",
    "batch_normalization_54/beta:0": "Mixed_6d.branch7x7dbl_1.bn.bias",
    "conv2d_55/kernel:0": "Mixed_6d.branch7x7dbl_2.conv.weight",
    "batch_normalization_55/beta:0": "Mixed_6d.branch7x7dbl_2.bn.bias",
    "conv2d_56/kernel:0": "Mixed_6d.branch7x7dbl_3.conv.weight",
    "batch_normalization_56/beta:0": "Mixed_6d.branch7x7dbl_3.bn.bias",
    "conv2d_57/kernel:0": "Mixed_6d.branch7x7dbl_4.conv.weight",
    "batch_normalization_57/beta:0": "Mixed_6d.branch7x7dbl_4.bn.bias",
    "conv2d_58/kernel:0": "Mixed_6d.branch7x7dbl_5.conv.weight",
    "batch_normalization_58/beta:0": "Mixed_6d.branch7x7dbl_5.bn.bias",
    "conv2d_59/kernel:0": "Mixed_6d.branch_pool.conv.weight",
    "batch_normalization_59/beta:0": "Mixed_6d.branch_pool.bn.bias",
    "conv2d_60/kernel:0": "Mixed_6e.branch1x1.conv.weight",
    "batch_normalization_60/beta:0": "Mixed_6e.branch1x1.bn.bias",
    "conv2d_61/kernel:0": "Mixed_6e.branch7x7_1.conv.weight",
    "batch_normalization_61/beta:0": "Mixed_6e.branch7x7_1.bn.bias",
    "conv2d_62/kernel:0": "Mixed_6e.branch7x7_2.conv.weight",
    "batch_normalization_62/beta:0": "Mixed_6e.branch7x7_2.bn.bias",
    "conv2d_63/kernel:0": "Mixed_6e.branch7x7_3.conv.weight",
    "batch_normalization_63/beta:0": "Mixed_6e.branch7x7_3.bn.bias",
    "conv2d_64/kernel:0": "Mixed_6e.branch7x7dbl_1.conv.weight",
    "batch_normalization_64/beta:0": "Mixed_6e.branch7x7dbl_1.bn.bias",
    "conv2d_65/kernel:0": "Mixed_6e.branch7x7dbl_2.conv.weight",
    "batch_normalization_65/beta:0": "Mixed_6e.branch7x7dbl_2.bn.bias",
    "conv2d_66/kernel:0": "Mixed_6e.branch7x7dbl_3.conv.weight",
    "batch_normalization_66/beta:0": "Mixed_6e.branch7x7dbl_3.bn.bias",
    "conv2d_67/kernel:0": "Mixed_6e.branch7x7dbl_4.conv.weight",
    "batch_normalization_67/beta:0": "Mixed_6e.branch7x7dbl_4.bn.bias",
    "conv2d_68/kernel:0": "Mixed_6e.branch7x7dbl_5.conv.weight",
    "batch_normalization_68/beta:0": "Mixed_6e.branch7x7dbl_5.bn.bias",
    "conv2d_69/kernel:0": "Mixed_6e.branch_pool.conv.weight",
    "batch_normalization_69/beta:0": "Mixed_6e.branch_pool.bn.bias",
    "conv2d_70/kernel:0": "Mixed_7a.branch3x3_1.conv.weight",
    "batch_normalization_70/beta:0": "Mixed_7a.branch3x3_1.bn.bias",
    "conv2d_71/kernel:0": "Mixed_7a.branch3x3_2.conv.weight",
    "batch_normalization_71/beta:0": "Mixed_7a.branch3x3_2.bn.bias",
    "conv2d_72/kernel:0": "Mixed_7a.branch7x7x3_1.conv.weight",
    "batch_normalization_72/beta:0": "Mixed_7a.branch7x7x3_1.bn.bias",
    "conv2d_73/kernel:0": "Mixed_7a.branch7x7x3_2.conv.weight",
    "batch_normalization_73/beta:0": "Mixed_7a.branch7x7x3_2.bn.bias",
    "conv2d_74/kernel:0": "Mixed_7a.branch7x7x3_3.conv.weight",
    "batch_normalization_74/beta:0": "Mixed_7a.branch7x7x3_3.bn.bias",
    "conv2d_75/kernel:0": "Mixed_7a.branch7x7x3_4.conv.weight",
    "batch_normalization_75/beta:0": "Mixed_7a.branch7x7x3_4.bn.bias",
    "conv2d_76/kernel:0": "Mixed_7b.branch1x1.conv.weight",
    "batch_normalization_76/beta:0": "Mixed_7b.branch1x1.bn.bias",
    "conv2d_77/kernel:0": "Mixed_7b.branch3x3_1.conv.weight",
    "batch_normalization_77/beta:0": "Mixed_7b.branch3x3_1.bn.bias",
    "conv2d_78/kernel:0": "Mixed_7b.branch3x3_2a.conv.weight",
    "batch_normalization_78/beta:0": "Mixed_7b.branch3x3_2a.bn.bias",
    "conv2d_79/kernel:0": "Mixed_7b.branch3x3_2b.conv.weight",
    "batch_normalization_79/beta:0": "Mixed_7b.branch3x3_2b.bn.bias",
    "conv2d_80/kernel:0": "Mixed_7b.branch3x3dbl_1.conv.weight",
    "batch_normalization_80/beta:0": "Mixed_7b.branch3x3dbl_1.bn.bias",
    "conv2d_81/kernel:0": "Mixed_7b.branch3x3dbl_2.conv.weight",
    "batch_normalization_81/beta:0": "Mixed_7b.branch3x3dbl_2.bn.bias",
    "conv2d_82/kernel:0": "Mixed_7b.branch3x3dbl_3a.conv.weight",
    "batch_normalization_82/beta:0": "Mixed_7b.branch3x3dbl_3a.bn.bias",
    "conv2d_83/kernel:0": "Mixed_7b.branch3x3dbl_3b.conv.weight",
    "batch_normalization_83/beta:0": "Mixed_7b.branch3x3dbl_3b.bn.bias",
    "conv2d_84/kernel:0": "Mixed_7b.branch_pool.conv.weight",
    "batch_normalization_84/beta:0": "Mixed_7b.branch_pool.bn.bias",
    "conv2d_85/kernel:0": "Mixed_7c.branch1x1.conv.weight",
    "batch_normalization_85/beta:0": "Mixed_7c.branch1x1.bn.bias",
    "conv2d_86/kernel:0": "Mixed_7c.branch3x3_1.conv.weight",
    "batch_normalization_86/beta:0": "Mixed_7c.branch3x3_1.bn.bias",
    "conv2d_87/kernel:0": "Mixed_7c.branch3x3_2a.conv.weight",
    "batch_normalization_87/beta:0": "Mixed_7c.branch3x3_2a.bn.bias",
    "conv2d_88/kernel:0": "Mixed_7c.branch3x3_2b.conv.weight",
    "batch_normalization_88/beta:0": "Mixed_7c.branch3x3_2b.bn.bias",
    "conv2d_89/kernel:0": "Mixed_7c.branch3x3dbl_1.conv.weight",
    "batch_normalization_89/beta:0": "Mixed_7c.branch3x3dbl_1.bn.bias",
    "conv2d_90/kernel:0": "Mixed_7c.branch3x3dbl_2.conv.weight",
    "batch_normalization_90/beta:0": "Mixed_7c.branch3x3dbl_2.bn.bias",
    "conv2d_91/kernel:0": "Mixed_7c.branch3x3dbl_3a.conv.weight",
    "batch_normalization_91/beta:0": "Mixed_7c.branch3x3dbl_3a.bn.bias",
    "conv2d_92/kernel:0": "Mixed_7c.branch3x3dbl_3b.conv.weight",
    "batch_normalization_92/beta:0": "Mixed_7c.branch3x3dbl_3b.bn.bias",
    "conv2d_93/kernel:0": "Mixed_7c.branch_pool.conv.weight",
    "batch_normalization_93/beta:0": "Mixed_7c.branch_pool.bn.bias",
    "classification/kernel:0": "fc.weight",
    "classification/bias:0": "fc.bias",
}

BN_MAP = {
    "batch_normalization/moving_variance:0": "Conv2d_1a_3x3.bn",
    "batch_normalization_1/moving_variance:0": "Conv2d_2a_3x3.bn",
    "batch_normalization_2/moving_variance:0": "Conv2d_2b_3x3.bn",
    "batch_normalization_3/moving_variance:0": "Conv2d_3b_1x1.bn",
    "batch_normalization_4/moving_variance:0": "Conv2d_4a_3x3.bn",
    "batch_normalization_5/moving_variance:0": "Mixed_5b.branch1x1.bn",
    "batch_normalization_6/moving_variance:0": "Mixed_5b.branch5x5_1.bn",
    "batch_normalization_7/moving_variance:0": "Mixed_5b.branch5x5_2.bn",
    "batch_normalization_8/moving_variance:0": "Mixed_5b.branch3x3dbl_1.bn",
    "batch_normalization_9/moving_variance:0": "Mixed_5b.branch3x3dbl_2.bn",
    "batch_normalization_10/moving_variance:0": "Mixed_5b.branch3x3dbl_3.bn",
    "batch_normalization_11/moving_variance:0": "Mixed_5b.branch_pool.bn",
    "batch_normalization_12/moving_variance:0": "Mixed_5c.branch1x1.bn",
    "batch_normalization_13/moving_variance:0": "Mixed_5c.branch5x5_1.bn",
    "batch_normalization_14/moving_variance:0": "Mixed_5c.branch5x5_2.bn",
    "batch_normalization_15/moving_variance:0": "Mixed_5c.branch3x3dbl_1.bn",
    "batch_normalization_16/moving_variance:0": "Mixed_5c.branch3x3dbl_2.bn",
    "batch_normalization_17/moving_variance:0": "Mixed_5c.branch3x3dbl_3.bn",
    "batch_normalization_18/moving_variance:0": "Mixed_5c.branch_pool.bn",
    "batch_normalization_19/moving_variance:0": "Mixed_5d.branch1x1.bn",
    "batch_normalization_20/moving_variance:0": "Mixed_5d.branch5x5_1.bn",
    "batch_normalization_21/moving_variance:0": "Mixed_5d.branch5x5_2.bn",
    "batch_normalization_22/moving_variance:0": "Mixed_5d.branch3x3dbl_1.bn",
    "batch_normalization_23/moving_variance:0": "Mixed_5d.branch3x3dbl_2.bn",
    "batch_normalization_24/moving_variance:0": "Mixed_5d.branch3x3dbl_3.bn",
    "batch_normalization_25/moving_variance:0": "Mixed_5d.branch_pool.bn",
    "batch_normalization_26/moving_variance:0": "Mixed_6a.branch3x3.bn",
    "batch_normalization_27/moving_variance:0": "Mixed_6a.branch3x3dbl_1.bn",
    "batch_normalization_28/moving_variance:0": "Mixed_6a.branch3x3dbl_2.bn",
    "batch_normalization_29/moving_variance:0": "Mixed_6a.branch3x3dbl_3.bn",
    "batch_normalization_30/moving_variance:0": "Mixed_6b.branch1x1.bn",
    "batch_normalization_31/moving_variance:0": "Mixed_6b.branch7x7_1.bn",
    "batch_normalization_32/moving_variance:0": "Mixed_6b.branch7x7_2.bn",
    "batch_normalization_33/moving_variance:0": "Mixed_6b.branch7x7_3.bn",
    "batch_normalization_34/moving_variance:0": "Mixed_6b.branch7x7dbl_1.bn",
    "batch_normalization_35/moving_variance:0": "Mixed_6b.branch7x7dbl_2.bn",
    "batch_normalization_36/moving_variance:0": "Mixed_6b.branch7x7dbl_3.bn",
    "batch_normalization_37/moving_variance:0": "Mixed_6b.branch7x7dbl_4.bn",
    "batch_normalization_38/moving_variance:0": "Mixed_6b.branch7x7dbl_5.bn",
    "batch_normalization_39/moving_variance:0": "Mixed_6b.branch_pool.bn",
    "batch_normalization_40/moving_variance:0": "Mixed_6c.branch1x1.bn",
    "batch_normalization_41/moving_variance:0": "Mixed_6c.branch7x7_1.bn",
    "batch_normalization_42/moving_variance:0": "Mixed_6c.branch7x7_2.bn",
    "batch_normalization_43/moving_variance:0": "Mixed_6c.branch7x7_3.bn",
    "batch_normalization_44/moving_variance:0": "Mixed_6c.branch7x7dbl_1.bn",
    "batch_normalization_45/moving_variance:0": "Mixed_6c.branch7x7dbl_2.bn",
    "batch_normalization_46/moving_variance:0": "Mixed_6c.branch7x7dbl_3.bn",
    "batch_normalization_47/moving_variance:0": "Mixed_6c.branch7x7dbl_4.bn",
    "batch_normalization_48/moving_variance:0": "Mixed_6c.branch7x7dbl_5.bn",
    "batch_normalization_49/moving_variance:0": "Mixed_6c.branch_pool.bn",
    "batch_normalization_50/moving_variance:0": "Mixed_6d.branch1x1.bn",
    "batch_normalization_51/moving_variance:0": "Mixed_6d.branch7x7_1.bn",
    "batch_normalization_52/moving_variance:0": "Mixed_6d.branch7x7_2.bn",
    "batch_normalization_53/moving_variance:0": "Mixed_6d.branch7x7_3.bn",
    "batch_normalization_54/moving_variance:0": "Mixed_6d.branch7x7dbl_1.bn",
    "batch_normalization_55/moving_variance:0": "Mixed_6d.branch7x7dbl_2.bn",
    "batch_normalization_56/moving_variance:0": "Mixed_6d.branch7x7dbl_3.bn",
    "batch_normalization_57/moving_variance:0": "Mixed_6d.branch7x7dbl_4.bn",
    "batch_normalization_58/moving_variance:0": "Mixed_6d.branch7x7dbl_5.bn",
    "batch_normalization_59/moving_variance:0": "Mixed_6d.branch_pool.bn",
    "batch_normalization_60/moving_variance:0": "Mixed_6e.branch1x1.bn",
    "batch_normalization_61/moving_variance:0": "Mixed_6e.branch7x7_1.bn",
    "batch_normalization_62/moving_variance:0": "Mixed_6e.branch7x7_2.bn",
    "batch_normalization_63/moving_variance:0": "Mixed_6e.branch7x7_3.bn",
    "batch_normalization_64/moving_variance:0": "Mixed_6e.branch7x7dbl_1.bn",
    "batch_normalization_65/moving_variance:0": "Mixed_6e.branch7x7dbl_2.bn",
    "batch_normalization_66/moving_variance:0": "Mixed_6e.branch7x7dbl_3.bn",
    "batch_normalization_67/moving_variance:0": "Mixed_6e.branch7x7dbl_4.bn",
    "batch_normalization_68/moving_variance:0": "Mixed_6e.branch7x7dbl_5.bn",
    "batch_normalization_69/moving_variance:0": "Mixed_6e.branch_pool.bn",
    "batch_normalization_70/moving_variance:0": "Mixed_7a.branch3x3_1.bn",
    "batch_normalization_71/moving_variance:0": "Mixed_7a.branch3x3_2.bn",
    "batch_normalization_72/moving_variance:0": "Mixed_7a.branch7x7x3_1.bn",
    "batch_normalization_73/moving_variance:0": "Mixed_7a.branch7x7x3_2.bn",
    "batch_normalization_74/moving_variance:0": "Mixed_7a.branch7x7x3_3.bn",
    "batch_normalization_75/moving_variance:0": "Mixed_7a.branch7x7x3_4.bn",
    "batch_normalization_76/moving_variance:0": "Mixed_7b.branch1x1.bn",
    "batch_normalization_77/moving_variance:0": "Mixed_7b.branch3x3_1.bn",
    "batch_normalization_78/moving_variance:0": "Mixed_7b.branch3x3_2a.bn",
    "batch_normalization_79/moving_variance:0": "Mixed_7b.branch3x3_2b.bn",
    "batch_normalization_80/moving_variance:0": "Mixed_7b.branch3x3dbl_1.bn",
    "batch_normalization_81/moving_variance:0": "Mixed_7b.branch3x3dbl_2.bn",
    "batch_normalization_82/moving_variance:0": "Mixed_7b.branch3x3dbl_3a.bn",
    "batch_normalization_83/moving_variance:0": "Mixed_7b.branch3x3dbl_3b.bn",
    "batch_normalization_84/moving_variance:0": "Mixed_7b.branch_pool.bn",
    "batch_normalization_85/moving_variance:0": "Mixed_7c.branch1x1.bn",
    "batch_normalization_86/moving_variance:0": "Mixed_7c.branch3x3_1.bn",
    "batch_normalization_87/moving_variance:0": "Mixed_7c.branch3x3_2a.bn",
    "batch_normalization_88/moving_variance:0": "Mixed_7c.branch3x3_2b.bn",
    "batch_normalization_89/moving_variance:0": "Mixed_7c.branch3x3dbl_1.bn",
    "batch_normalization_90/moving_variance:0": "Mixed_7c.branch3x3dbl_2.bn",
    "batch_normalization_91/moving_variance:0": "Mixed_7c.branch3x3dbl_3a.bn",
    "batch_normalization_92/moving_variance:0": "Mixed_7c.branch3x3dbl_3b.bn",
    "batch_normalization_93/moving_variance:0": "Mixed_7c.branch_pool.bn",
    "batch_normalization/moving_mean:0": "Conv2d_1a_3x3.bn",
    "batch_normalization_1/moving_mean:0": "Conv2d_2a_3x3.bn",
    "batch_normalization_2/moving_mean:0": "Conv2d_2b_3x3.bn",
    "batch_normalization_3/moving_mean:0": "Conv2d_3b_1x1.bn",
    "batch_normalization_4/moving_mean:0": "Conv2d_4a_3x3.bn",
    "batch_normalization_5/moving_mean:0": "Mixed_5b.branch1x1.bn",
    "batch_normalization_6/moving_mean:0": "Mixed_5b.branch5x5_1.bn",
    "batch_normalization_7/moving_mean:0": "Mixed_5b.branch5x5_2.bn",
    "batch_normalization_8/moving_mean:0": "Mixed_5b.branch3x3dbl_1.bn",
    "batch_normalization_9/moving_mean:0": "Mixed_5b.branch3x3dbl_2.bn",
    "batch_normalization_10/moving_mean:0": "Mixed_5b.branch3x3dbl_3.bn",
    "batch_normalization_11/moving_mean:0": "Mixed_5b.branch_pool.bn",
    "batch_normalization_12/moving_mean:0": "Mixed_5c.branch1x1.bn",
    "batch_normalization_13/moving_mean:0": "Mixed_5c.branch5x5_1.bn",
    "batch_normalization_14/moving_mean:0": "Mixed_5c.branch5x5_2.bn",
    "batch_normalization_15/moving_mean:0": "Mixed_5c.branch3x3dbl_1.bn",
    "batch_normalization_16/moving_mean:0": "Mixed_5c.branch3x3dbl_2.bn",
    "batch_normalization_17/moving_mean:0": "Mixed_5c.branch3x3dbl_3.bn",
    "batch_normalization_18/moving_mean:0": "Mixed_5c.branch_pool.bn",
    "batch_normalization_19/moving_mean:0": "Mixed_5d.branch1x1.bn",
    "batch_normalization_20/moving_mean:0": "Mixed_5d.branch5x5_1.bn",
    "batch_normalization_21/moving_mean:0": "Mixed_5d.branch5x5_2.bn",
    "batch_normalization_22/moving_mean:0": "Mixed_5d.branch3x3dbl_1.bn",
    "batch_normalization_23/moving_mean:0": "Mixed_5d.branch3x3dbl_2.bn",
    "batch_normalization_24/moving_mean:0": "Mixed_5d.branch3x3dbl_3.bn",
    "batch_normalization_25/moving_mean:0": "Mixed_5d.branch_pool.bn",
    "batch_normalization_26/moving_mean:0": "Mixed_6a.branch3x3.bn",
    "batch_normalization_27/moving_mean:0": "Mixed_6a.branch3x3dbl_1.bn",
    "batch_normalization_28/moving_mean:0": "Mixed_6a.branch3x3dbl_2.bn",
    "batch_normalization_29/moving_mean:0": "Mixed_6a.branch3x3dbl_3.bn",
    "batch_normalization_30/moving_mean:0": "Mixed_6b.branch1x1.bn",
    "batch_normalization_31/moving_mean:0": "Mixed_6b.branch7x7_1.bn",
    "batch_normalization_32/moving_mean:0": "Mixed_6b.branch7x7_2.bn",
    "batch_normalization_33/moving_mean:0": "Mixed_6b.branch7x7_3.bn",
    "batch_normalization_34/moving_mean:0": "Mixed_6b.branch7x7dbl_1.bn",
    "batch_normalization_35/moving_mean:0": "Mixed_6b.branch7x7dbl_2.bn",
    "batch_normalization_36/moving_mean:0": "Mixed_6b.branch7x7dbl_3.bn",
    "batch_normalization_37/moving_mean:0": "Mixed_6b.branch7x7dbl_4.bn",
    "batch_normalization_38/moving_mean:0": "Mixed_6b.branch7x7dbl_5.bn",
    "batch_normalization_39/moving_mean:0": "Mixed_6b.branch_pool.bn",
    "batch_normalization_40/moving_mean:0": "Mixed_6c.branch1x1.bn",
    "batch_normalization_41/moving_mean:0": "Mixed_6c.branch7x7_1.bn",
    "batch_normalization_42/moving_mean:0": "Mixed_6c.branch7x7_2.bn",
    "batch_normalization_43/moving_mean:0": "Mixed_6c.branch7x7_3.bn",
    "batch_normalization_44/moving_mean:0": "Mixed_6c.branch7x7dbl_1.bn",
    "batch_normalization_45/moving_mean:0": "Mixed_6c.branch7x7dbl_2.bn",
    "batch_normalization_46/moving_mean:0": "Mixed_6c.branch7x7dbl_3.bn",
    "batch_normalization_47/moving_mean:0": "Mixed_6c.branch7x7dbl_4.bn",
    "batch_normalization_48/moving_mean:0": "Mixed_6c.branch7x7dbl_5.bn",
    "batch_normalization_49/moving_mean:0": "Mixed_6c.branch_pool.bn",
    "batch_normalization_50/moving_mean:0": "Mixed_6d.branch1x1.bn",
    "batch_normalization_51/moving_mean:0": "Mixed_6d.branch7x7_1.bn",
    "batch_normalization_52/moving_mean:0": "Mixed_6d.branch7x7_2.bn",
    "batch_normalization_53/moving_mean:0": "Mixed_6d.branch7x7_3.bn",
    "batch_normalization_54/moving_mean:0": "Mixed_6d.branch7x7dbl_1.bn",
    "batch_normalization_55/moving_mean:0": "Mixed_6d.branch7x7dbl_2.bn",
    "batch_normalization_56/moving_mean:0": "Mixed_6d.branch7x7dbl_3.bn",
    "batch_normalization_57/moving_mean:0": "Mixed_6d.branch7x7dbl_4.bn",
    "batch_normalization_58/moving_mean:0": "Mixed_6d.branch7x7dbl_5.bn",
    "batch_normalization_59/moving_mean:0": "Mixed_6d.branch_pool.bn",
    "batch_normalization_60/moving_mean:0": "Mixed_6e.branch1x1.bn",
    "batch_normalization_61/moving_mean:0": "Mixed_6e.branch7x7_1.bn",
    "batch_normalization_62/moving_mean:0": "Mixed_6e.branch7x7_2.bn",
    "batch_normalization_63/moving_mean:0": "Mixed_6e.branch7x7_3.bn",
    "batch_normalization_64/moving_mean:0": "Mixed_6e.branch7x7dbl_1.bn",
    "batch_normalization_65/moving_mean:0": "Mixed_6e.branch7x7dbl_2.bn",
    "batch_normalization_66/moving_mean:0": "Mixed_6e.branch7x7dbl_3.bn",
    "batch_normalization_67/moving_mean:0": "Mixed_6e.branch7x7dbl_4.bn",
    "batch_normalization_68/moving_mean:0": "Mixed_6e.branch7x7dbl_5.bn",
    "batch_normalization_69/moving_mean:0": "Mixed_6e.branch_pool.bn",
    "batch_normalization_70/moving_mean:0": "Mixed_7a.branch3x3_1.bn",
    "batch_normalization_71/moving_mean:0": "Mixed_7a.branch3x3_2.bn",
    "batch_normalization_72/moving_mean:0": "Mixed_7a.branch7x7x3_1.bn",
    "batch_normalization_73/moving_mean:0": "Mixed_7a.branch7x7x3_2.bn",
    "batch_normalization_74/moving_mean:0": "Mixed_7a.branch7x7x3_3.bn",
    "batch_normalization_75/moving_mean:0": "Mixed_7a.branch7x7x3_4.bn",
    "batch_normalization_76/moving_mean:0": "Mixed_7b.branch1x1.bn",
    "batch_normalization_77/moving_mean:0": "Mixed_7b.branch3x3_1.bn",
    "batch_normalization_78/moving_mean:0": "Mixed_7b.branch3x3_2a.bn",
    "batch_normalization_79/moving_mean:0": "Mixed_7b.branch3x3_2b.bn",
    "batch_normalization_80/moving_mean:0": "Mixed_7b.branch3x3dbl_1.bn",
    "batch_normalization_81/moving_mean:0": "Mixed_7b.branch3x3dbl_2.bn",
    "batch_normalization_82/moving_mean:0": "Mixed_7b.branch3x3dbl_3a.bn",
    "batch_normalization_83/moving_mean:0": "Mixed_7b.branch3x3dbl_3b.bn",
    "batch_normalization_84/moving_mean:0": "Mixed_7b.branch_pool.bn",
    "batch_normalization_85/moving_mean:0": "Mixed_7c.branch1x1.bn",
    "batch_normalization_86/moving_mean:0": "Mixed_7c.branch3x3_1.bn",
    "batch_normalization_87/moving_mean:0": "Mixed_7c.branch3x3_2a.bn",
    "batch_normalization_88/moving_mean:0": "Mixed_7c.branch3x3_2b.bn",
    "batch_normalization_89/moving_mean:0": "Mixed_7c.branch3x3dbl_1.bn",
    "batch_normalization_90/moving_mean:0": "Mixed_7c.branch3x3dbl_2.bn",
    "batch_normalization_91/moving_mean:0": "Mixed_7c.branch3x3dbl_3a.bn",
    "batch_normalization_92/moving_mean:0": "Mixed_7c.branch3x3dbl_3b.bn",
    "batch_normalization_93/moving_mean:0": "Mixed_7c.branch_pool.bn",
}


def tf2pytorch(tf_model, pt_model, output, copy_tensor=False):
    pt_tensor_dict = {}
    pt_shape_dict = {}
    for name, param in pt_model.named_parameters():
        pt_tensor_dict[name] = param
        pt_shape_dict[name] = param.shape

    tf_tensor_map = {}
    tf_shape_map = {}
    for var in tf_model.variables:
        tf_tensor_map[var.name] = var.numpy()
        tf_shape_map[var.name] = var.shape
    for tf_tensor_name, tf_tensor_shape in tf_shape_map.items():
        tf_tensor = tf_tensor_map[tf_tensor_name]
        if tf_tensor_name in TF_TENSOR_2_PT_TENSOR:
            pt_tensor_name = TF_TENSOR_2_PT_TENSOR[tf_tensor_name]
            tf_shape_tuple = tuple(tf_tensor_shape)
            pt_shape_tuple = tuple(pt_shape_dict[pt_tensor_name])
            pt_tensor = pt_tensor_dict[pt_tensor_name]
            # Convert shapes for comparison
            if "conv" in tf_tensor_name and "kernel" in tf_tensor_name:
                # TensorFlow Conv: [height, width, in_channels, out_channels]
                # PyTorch Conv: [out_channels, in_channels, height, width]
                tf_converted_shape = (
                    tf_shape_tuple[3],
                    tf_shape_tuple[2],
                    tf_shape_tuple[0],
                    tf_shape_tuple[1],
                )
                print(
                    f"Conv: {tf_tensor_name}: {tf_shape_tuple} -> {tf_converted_shape}, {pt_tensor_name}: {pt_shape_tuple}"
                )
                assert tf_converted_shape == pt_shape_tuple, (
                    f"Shape mismatch: {tf_converted_shape} vs {pt_shape_tuple}"
                )

                if copy_tensor:
                    # Transpose from [h, w, in_c, out_c] to [out_c, in_c, h, w]
                    tf_tensor = tf_tensor.transpose(3, 2, 0, 1)
                    pt_tensor.data = torch.from_numpy(tf_tensor)

            elif "batch_normalization" in tf_tensor_name and "beta" in tf_tensor_name:
                # Bias shapes should match directly
                print(
                    f"BN: {tf_tensor_name}: {tf_shape_tuple}, {pt_tensor_name}: {pt_shape_tuple}"
                )
                assert tf_shape_tuple[-1] == pt_shape_tuple[0], (
                    f"Bias shape mismatch: {tf_shape_tuple[-1]} vs {pt_shape_tuple[0]}"
                )

                if copy_tensor:
                    # Copy the tensor directly
                    pt_tensor.data = torch.from_numpy(tf_tensor)

            elif "classification/kernel" in tf_tensor_name:
                # FC layer: TF [in, out] -> PT [out, in]
                tf_converted_shape = (tf_shape_tuple[1], tf_shape_tuple[0])
                print(
                    f"CL_Kernel: {tf_tensor_name}: {tf_shape_tuple} -> {tf_converted_shape}, {pt_tensor_name}: {pt_shape_tuple}"
                )
                assert tf_converted_shape == pt_shape_tuple, (
                    f"FC shape mismatch: {tf_converted_shape} vs {pt_shape_tuple}"
                )

                if copy_tensor:
                    # Transpose from [in, out] to [out, in]
                    tf_tensor = tf_tensor.transpose(1, 0)
                    pt_tensor.data = torch.from_numpy(tf_tensor)
            elif "classification/bias:0" in tf_tensor_name:
                # FC layer: TF [out] -> PT [out]
                print(
                    f"CL_Bias: {tf_tensor_name}: {tf_shape_tuple}, {pt_tensor_name}: {pt_shape_tuple}"
                )
                assert tf_shape_tuple[0] == pt_shape_tuple[0], (
                    f"FC bias shape mismatch: {tf_shape_tuple[0]} vs {pt_shape_tuple[0]}"
                )

                if copy_tensor:
                    # Copy the tensor directly
                    pt_tensor.data = torch.from_numpy(tf_tensor)
            else:
                raise ValueError(f"Wrong tensor name: {tf_tensor_name}")

        elif tf_tensor_name in BN_MAP:
            bn_layer = pt_model.get_submodule(BN_MAP[tf_tensor_name])
            if "moving_mean" in tf_tensor_name:
                bn_layer.running_mean = torch.from_numpy(tf_tensor)
            elif "moving_variance" in tf_tensor_name:
                bn_layer.running_var = torch.from_numpy(tf_tensor)
        else:
            raise ValueError(f"{tf_tensor_name} not found in TF_TENSOR_2_PT_TENSOR")

    # Save the PyTorch model with the copied weights
    if copy_tensor:
        torch.save(pt_model, output)
        print(f"Model conversion completed and saved to {output}")
    return pt_model


def run_tf2pytorch(args):
    tf_model = tf_inception_v3(weights=args.weights)

    pt_model = pt_inception_v3(
        num_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE,
        init_weights=False,
        aux_logits=False,
    )
    pt_model = tf2pytorch(
        tf_model=tf_model,
        pt_model=pt_model,
        output=args.output,
        copy_tensor=True,
    )
    total_trainable_params = sum(
        p.numel() for p in pt_model.parameters() if p.requires_grad
    )
    print(f"Total trainable parameters: {total_trainable_params}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, required=True, help="TensorFlow model weights"
    )
    parser.add_argument("--output", type=str, required=True, help="Output path")
    args = parser.parse_args()
    run_tf2pytorch(args)
