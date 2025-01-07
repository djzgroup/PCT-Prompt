import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser('training')

    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--checkpoint', default="../exprement_seg/dales", type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    # parser.add_argument('--batch_size', type=int, default=16,
    #                     help='batch size in training')
    parser.add_argument('--model', default='sem_seg', help='model_name')

    parser.add_argument('--scheduler', type=str, default='cos',
                        help='lr scheduler')
    # parser.add_argument('--epoch', default=250, type=int,
    #                     help='number of epoch in training')
    # parser.add_argument('--num_points', type=int,
    #                     default=1024, help='Point Number')
    parser.add_argument('--use_sgd', type=bool,
                        default=False, help='use sgd / adam')
    # parser.add_argument('--learning_rate', default=0.01,
    #                     type=float, help='learning rate in training')
    # parser.add_argument('--weight_decay', type=float,
    #                     default=2e-4, help='decay rate')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='min lr')
    parser.add_argument('--new_lr', default=False,
                        type=bool, help='use new lr')
    parser.add_argument('--cuda', type=bool,
                        default=True, help='enables CUDA training')

    # parser.add_argument('--dropout', type=float,
    #                     default=0.5, help='dropout rate')
    parser.add_argument('--new_model', default=True,
                        type=bool, help='choose new model')
    parser.add_argument(
        '--config',
        type=str,
        default='../cfgs/Mixup_models/PCT-DALES.yaml',
        help='yaml config file')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--num_workers', type=int, default=12)
    # seed
    parser.add_argument('--manual_seed', type=int, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--seed', type=int, default=1000,help='random seed')
    # parser.add_argument(
    #     '--deterministic',
    #     action='store_true',
    #     help='whether to set deterministic options for CUDNN backend.')
    # bn
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='train1', help = 'experiment name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False,
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model', 
        action='store_true', 
        default=False, 
        help = 'finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model', 
        action='store_true', 
        default=False, 
        help = 'training modelnet from scratch')
    parser.add_argument(
        '--label_smoothing', 
        action='store_true', 
        default=False, 
        help = 'use label smoothing loss trick')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')        
    parser.add_argument(
        '--way', type=int, default=-1)
    parser.add_argument(
        '--shot', type=int, default=-1)
    parser.add_argument(
        '--fold', type=int, default=-1)
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    # Augmentation
    parser.add_argument('--aug_scale', action='store_true', default=False,
                        help='Whether to augment by scaling [default: False]')
    parser.add_argument('--aug_rotate', type=str, default=None,
                        help='Type to augment by rotation [pert, pert_z, rot, rot_z]')
    parser.add_argument('--aug_jitter', action='store_true', default=False,
                        help='Whether to augment by shifting [default: False]')
    parser.add_argument('--aug_flip', action='store_true', default=False,
                        help='Whether to augment by flipping [default: False]')
    parser.add_argument('--aug_shift', action='store_true', default=False,
                        help='Whether to augment by shifting [default: False]')
    parser.add_argument('--color_contrast', action='store_true', default=False,
                        help='Whether to augment by RGB contrasting [default: False]')
    parser.add_argument('--color_shift', action='store_true', default=False,
                        help='Whether to augment by RGB shifting  [default: False]')
    parser.add_argument('--color_jitter', action='store_true', default=False,
                        help='Whether to augment by RGB jittering [default: False]')
    parser.add_argument('--hs_shift', action='store_true', default=False,
                        help='Whether to augment by HueSaturation shifting [default: False]')
    parser.add_argument('--color_drop', action='store_true', default=False,
                        help='Whether to augment by RGB Dropout [default: False]')
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while finetune_model mode')
    args.local_rank="0,1"
    if 'LOCAL_RANK' not in os.environ:
        # print(os.environ.keys())
        print(args.local_rank)
        os.environ['LOCAL_RANK'] = args.local_rank

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode
    args.experiment_path = os.path.join('/data/Point-adapter/exprement_part',  args.exp_name)
    args.tfboard_path = os.path.join('/data/Point-adapter/exprement_part','TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

