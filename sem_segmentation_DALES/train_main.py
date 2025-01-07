import torch
import numpy as np
from seg_trainer_boundary import TrainerSegmentation as Trainer

if __name__ == '__main__':
    try:
        # 设置随机种子以保证结果的可重复性
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        # 创建Trainer实例并训练
        trainer_instance = Trainer(dataparallel=False, more_aug=False, weight_decay_sgd=5e-4)
        trainer_instance.train_all()

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
