import numpy
import torch
import torch.nn as nn


from torch.nn import functional as F
from tqdm import tqdm
import os
from torch.utils import data

from DALESDataLoader import DALES
from modules.aug_utils import transform_point_cloud_coord, transform_point_cloud_rgb
from util.utils_surf import get_rgb_stat, get_class_weights, get_loss, get_dataset_description, intersectionAndUnionGPU, \
    AverageMeter
from util import transform
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

from sem_segmentation import get_semantic_model
import numpy as np
from utils.config import get_config
from utils import parser
from tensorboardX import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
classes = ['Ground', ' Buildings', ' Cars', 'Trucks', 'Poles', 'Power Lines', 'Fences', 'Vegetation']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def dice_loss(pred, target):
    """
    Dice 损失函数
    :param pred: 预测输出, 形状为 [batch_size, num_points, num_classes]
    :param target: 真实标签, 形状为 [batch_size, num_points]
    """
    smooth = 1.0
    num_classes = pred.shape[-1]

    pred = F.softmax(pred, dim=-1)
    pred_flat = pred.view(-1, num_classes)
    target_flat = F.one_hot(target, num_classes).view(-1, num_classes)

    intersection = (pred_flat * target_flat).sum(dim=0)
    dice_score = (2. * intersection + smooth) / (pred_flat.sum(dim=0) + target_flat.sum(dim=0) + smooth)

    return 1 - dice_score.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, label_smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = label_smoothing
        self.confidence = 1. - self.smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=1)
        nll_loss = -logprobs.gather(dim=1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class TrainerSegmentation(nn.Module):
    def __init__(self,  dataparallel=True,
                  more_aug=False, weight_decay_sgd=1e-4,
                 ):
        super(TrainerSegmentation, self).__init__()
        self.args = parser.get_args()
        config = get_config(self.args, logger=None)
        # n_layers: int, feat_dims: list, n_samples: list, n_class: int, in_feat_dim: int
        self.num_epochs = config.max_epoch
        batch_size=config.batch_size
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"当前系统中可用的GPU数量为：{device_count}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.NUM_CLASSES=config.class_num
        self.best_iou = 0
        self.best_OA=0
        self.best_mAcc=0
        self.args.test_area=config.test_area
        # self.device = torch.device("cuda:" + str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        # self.model_dir = self.args.checkpoint
        self.model_dir = '/home/PCT-Prompt/PCT-Prompt/exprement_seg/dales'
        # self.fea_num = fea_num
        self.more_aug = more_aug
        self.lr_scheduler = "step"
        self.epoch_checkpoints = {120: 0.1, 160: 0.01}
        resume=self.args.resume
        self.lr_mult_ratio = 0.9
        self.init_lr =config.optimizer.kwargs.learning_rate
        self.weight_decay = 1e-4
        self.weight_decay_sgd = weight_decay_sgd
        self.n_points = config.num_points
        self.num_gpu = 1
        self.args.dataset=config.dataset.NAME
        self.model =get_semantic_model(config).to(self.device)
        self.start_epoch = 0
        self.args.description =config.dataset.NAME+"_A5"
        self.label_weight = get_class_weights(self.args.description).cuda()
        self.args.ignore_label=255
        self.criterion = get_loss(self.label_weight,self.args.ignore_label).cuda()
        self.args.voxel_max, self.args.voxel_size, self.ignore_label = 10000, 0.5, 255
        optimizer_dict=None
        if resume==True:
            checkpoint_path = os.path.join("../exprement_semseg_test/prompt_sample40_group256_10000noresexprement_semantic_no_pretrain_pointnet_GF_adamw_colordrop_lr00005_4cross_res_knn_epoch1500/checkpoint_current.pth")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            optimizer_dict = checkpoint['optimizer_state_dict']
            base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
            self.model.load_state_dict(base_ckpt)
            self.best_iou = checkpoint['mIou']
            self.best_OA = checkpoint['acc']
            self.best_mAcc = checkpoint['class_avg_acc']

            self.start_epoch = checkpoint['epoch']
            print("Resume training model...")
            print(torch.load(checkpoint_path
                             ).keys())
        else:
            print('No existing model, starting training from scratch...')
            start_epoch = 0
            ckpt = torch.load(config.pretrain)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                    base_ckpt['model.' + k[len('transformer_q.'):]] = base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt['model.' + k[len('base_model.'):]] = base_ckpt[k]
                del base_ckpt[k]

            incompatible = self.model.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                    get_missing_parameters_message(incompatible.missing_keys)
                )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                    get_unexpected_parameters_message(incompatible.unexpected_keys)
                )

                print((f'[Transformer] Successful Loading the ckpt from {config.pretrain}'))
        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=[0])
        self.dataparallel = dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1

        self.args.aug_args=  {'scale_factor': 0.1, 'scale_ani': True, 'scale_prob': 1.,
                    'pert_factor': 0.03, 'pert_prob': 1., 'rot_prob': 0.5,
                    'shifts': [0.1, 0.1, 0.1], 'shift_prob': 1.}
        coord_transform = transform_point_cloud_coord(self.args)
        rgb_transform = transform_point_cloud_rgb(self.args)
        rgb_mean, rgb_std = get_rgb_stat(self.args)
        self.args.data_dir= config.dataset.DATA_PATH
        self.args.loop=1
        self.args.data_norm="mean"

        self.train_set = DALES(self.args, 'train', coord_transform, rgb_transform, rgb_mean, rgb_std, True)
        self.test_set = DALES(self.args, 'test', None, None, rgb_mean, rgb_std, False)

        self.use_sgd = self.args.use_sgd
        if not self.use_sgd:
            self.optimizer = torch.optim.AdamW([{'params': self.model.parameters(), 'initial_lr':config.optimizer.kwargs.learning_rate}],
                                    lr=config.optimizer.kwargs.learning_rate, weight_decay=1e-4)
            self.schduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,self.num_epochs
                                                                       ,
                                                                       eta_min=0.000001,
                                                                       )
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=config.optimizer.kwargs.learning_rate,
                                             momentum=0.9,
                                             weight_decay=1e-4,
                                             )

            self.schduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,self.num_epochs,eta_min=0.000001)
        if optimizer_dict  is not None:
            self.optimizer.load_state_dict(optimizer_dict)

        self.train_loader = data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True,drop_last=True)
        self.test_loader = data.DataLoader(self.test_set, batch_size=config.test_batch_size, shuffle=True,drop_last=True)



        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.model_dir =  self.model_dir


    def adjust_learning_rate(self, epoch):
        if epoch in self.epoch_checkpoints:
            new_lr = self.init_lr * self.epoch_checkpoints[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def calculate_acc(self, pred: torch.LongTensor, gt: torch.LongTensor):
        return (torch.sum(torch.max(pred.detach(), dim=1)[1] == gt).cpu().item())

    def _train_one_epoch(self, epoch):
        self.model.train()
        if self.use_sgd and self.lr_scheduler != "cosine":
            if epoch in self.epoch_checkpoints:
                self.adjust_learning_rate(epoch)

        tot_acc = 0
        tot_num = 0
        loss_list = []
        loss_nn = []
        step = 0
        labelweights=0
        # train_bar = tqdm(self.train_set)
        train_bar = tqdm(self.train_loader)
        for batch_data, batch_label in train_bar:
            points = batch_data.float().data.numpy()
            points = torch.Tensor(points)
            # # target = target[:, 0]
            points = points.float()
            batch_label = batch_label.long()
            # print(batch_label.size())
            if len(batch_label.size()) > 1:
                batch_label = batch_label.view(-1)
            if self.dataparallel:
                points =  points.to(self.device)
                # batch_pos = batch_pos.to(self.device)
                batch_label = batch_label.to(self.device)
            else:
                points =  points.cuda()
                # batch_pos = batch_pos.cuda()
                batch_label = batch_label.cuda()
            bz, N = points.size(0), points.size(1)
            batch_label_tmp = batch_label.view(bz, N).cpu().data.numpy()
            tmp, _ = np.histogram(batch_label_tmp, range(self.NUM_CLASSES + 1))
            labelweights += tmp
            points = points.transpose(2, 1)
            logits = self.model(points)
            logits = logits.contiguous().view(bz * N, -1)
            batch_label = batch_label.view(-1)
            # 计算主要损失
            main_loss = self.criterion(input=logits, target=batch_label)

            # 计算边界感知损失
            de_loss = dice_loss(logits,batch_label).cuda()

            # 总损失
            total_loss = main_loss + de_loss  # 你可以调整损失权重
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            pred_labels = self.calculate_acc(logits, batch_label)

            tot_acc += pred_labels
            tot_num += logits.size(0)
            loss_list += [total_loss.detach().cpu().item() * bz]
            loss_nn.append(bz)
            train_bar.set_description(
                'Train Epoch: [{}/{}] Loss:{:.3f} Acc@:{:.2f}%'.format(epoch+1, self.num_epochs,
                                                                      float(sum(loss_list) / sum(loss_nn)),
                                                                      float(tot_acc / tot_num) * 100))
        with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
            wf.write('Learning rate:%f\t' % self.optimizer.param_groups[0]['lr'])
            wf.write("Train Epoch: {:d}, loss: {:.4f}, Acc: {:.2f}%\t".format(epoch + 1,
                                                                              float(sum(loss_list) / sum(loss_nn)),
                                                                              float(tot_acc / tot_num) * 100))
            wf.close()
        self.schduler.step()

    def _test(self, epoch):
        self.model.eval()


        with torch.no_grad():
            loss_sum=0
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            labelweights = np.zeros(self.NUM_CLASSES)
            num_batches = len(self.test_loader)
            test_bar = tqdm(self.test_loader)
            for batch_data, batch_label in test_bar:
                batch_x = batch_data.float()
                batch_pos = batch_data.float()
                batch_label = batch_label.long()
                if len(batch_label.size()) > 1:
                    batch_label = batch_label.view(-1)
                if self.dataparallel:
                    batch_x = batch_x.to(self.device)
                    batch_pos = batch_pos.to(self.device)
                    batch_label = batch_label.to(self.device)
                else:
                    batch_x = batch_x.cuda()
                    batch_pos = batch_pos.cuda()
                    batch_label = batch_label.cuda()
                bz, N = batch_x.size(0), batch_x.size(1)
                batch_label_tmp = batch_label.view(bz, N).cpu().data.numpy()
                tmp, _ = np.histogram(batch_label_tmp, range(self.NUM_CLASSES + 1))
                labelweights += tmp
                points = batch_x.transpose(2, 1)
                logits = self.model(points)
                pred_val = logits.contiguous().cpu().data.numpy()
                logits = logits.contiguous().view(bz * N, -1)
                pred_val = np.argmax(pred_val, 2)
                # 计算主要损失
                loss = self.criterion(logits, batch_label)
                loss_sum += loss.item()

                # 计算边界损失
                de_loss = dice_loss(logits, batch_label).cuda()
                loss_sum += de_loss.item()  # 累加边界损失

                batch_label_s = batch_label.view(bz,N).cpu().data.numpy()
                batch_label = batch_label.view(-1)
                output = logits.max(1)[1]  # remove unclassified label
                intersection, union, target = intersectionAndUnionGPU(output, batch_label, self.NUM_CLASSES, self.args.ignore_label)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
                test_bar.set_description('Test Epoch: [{}/{}] Acc@:{:.2f}%'.format(epoch + 1, self.num_epochs,
                                                                                   (sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10))*100))
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            avg_class_iou = np.mean(iou_class)
            avg_class_acc = np.mean(accuracy_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
            if avg_class_iou >= self.best_iou:
                self.best_iou = avg_class_iou
                print('Save best mIou model...')
                savepath = self.args.checkpoint + "/best_mIou_model.pth"
                print('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'mIou': self.best_iou,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(state, savepath)
            # print('Best mIoU: %f' % self.best_iou)
            if allAcc*100 >= self.best_OA:
                self.best_OA = float(allAcc) * 100
                print('Save best OA model...')
                savepath = self.args.checkpoint + "/best_OA_model.pth"
                print('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'acc': self.best_OA ,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(state, savepath)
            if avg_class_acc >= self.best_mAcc:
                self.best_mAcc = avg_class_acc
                print('Save best mAcc model...')
                savepath = self.args.checkpoint + "/best_mAcc_model.pth"
                print('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_acc': self.best_mAcc,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(state, savepath)
            with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                wf.write("Test Epoch: {:d}, loss:{:.4f}, Acc: {:.4f} ----".format(epoch + 1,loss_sum / float(num_batches), allAcc*100))
                wf.write('eval best Acc: {:.4f}%\n'.format(self.best_OA))
                wf.write('eval point mIoU: {:.4f} ----' .format(avg_class_iou))
                wf.write('eval best mIou: {:.4f}%\n'.format(self.best_iou))
                wf.write('eval point mAcc: {:.4f} ----'.format(avg_class_acc))
                wf.write('eval best mAcc: {:.4f}%\n'.format(self.best_mAcc))
                iou_per_class_str = 'Test Epoch: {:d},---- IoU --------%\n'.format(epoch + 1)
                for l in range(self.NUM_CLASSES):
                    iou_per_class_str += 'weight:%.f,class %s, IoU: %.4f,Acc: %.4f \n' % (labelweights[l-1],
                        seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                        iou_class[l] * 100,accuracy_class[l] * 100)
                wf.write(iou_per_class_str )
                wf.close()

    def save_model(self, epoch):
        state = {
            'epoch': epoch,
            'acc': self.best_OA,
            'class_avg_acc': self.best_mAcc,
            'mIou': self.best_iou,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        savepath = self.args.checkpoint + "/checkpoint_current.pth"
        torch.save(state, savepath)

    def train_all(self):
        print("Start training.")
        for epoch in range(self.start_epoch,self.num_epochs):
            print('Learning rate:%f\t' % self.optimizer.param_groups[0]['lr'])
            self._train_one_epoch(epoch)

            self._test(epoch)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print("=== Saving model")
                self.save_model(epoch + 1)
                print("=== Saved")

