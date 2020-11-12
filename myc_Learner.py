from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt

plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz


def evaluation(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings1, embeddings2,
                                                        np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
    #     thresholds = np.arange(0, 4, 0.001)
    #     val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
    #                                       np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    #     return tpr, fpr, accuracy, best_thresholds, val, val_std, far
    return tpr, fpr, accuracy, best_thresholds


class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        # print("milestones:", conf.milestones) #[12, 15, 18]
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            # print('\n')
            # print("self.model:", self.model.linear.weight)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

        self.threshold = conf.threshold

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
                                     ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step,
                                                                                   extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                                        ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step,
                                                                                     extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                                             ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,
                                                                                               self.step, extra)))

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        self.model.load_state_dict(torch.load(save_path / 'model_{}'.format(fixed_str)))

        # state_dict = torch.load(save_path / 'model_{}'.format(fixed_str))
        # from collections import orderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]
        #     new_state_dict[name] = v
        # self.model.load_state_dict(new_state_dict)

        if not model_only:
            self.head.load_state_dict(torch.load(save_path / 'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))

    def extract(self, conf, carray, tta=False):
        # print("self.agedb_30.size:", np.shape(self.agedb_30))
        # carray-self.agedb_30 (12000, 3, 112, 112)
        # len(carray) = 12000
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])  # np.zeros(12000, 512)
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx: idx + conf.batch_size])  # carray[12000, 3, 112, 112]
                if tta:
                    # 验证调用的此处
                    # hflip_batch --> 添加了一些强化的技巧
                    fliped = hflip_batch(batch)
                    # print("fliped:", np.shape(fliped)) #fliped: torch.Size([200, 3, 112, 112])

                    # 进入模型进行训练
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    # print("emb_batch:", np.shape(emb_batch)) #emb_batch: torch.Size([200, 512])

                    # embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch).cpu()
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        # tpr, fpr, accuracy, best_thresholds = evaluation(embeddings, issame, nrof_folds)
        # buf = gen_plot(fpr, tpr)
        # roc_curve = Image.open(buf)
        # roc_curve_tensor = trans.ToTensor()(roc_curve)
        return embeddings

    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss, batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            # Do the SGD step
            # Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses

    def train(self, conf, epochs):
        self.model.train()
        # print("self.agedb_30.size:", np.shape(self.agedb_30))
        # print("self.agedb_30.len:",len(self.agedb_30)) 12000

        # conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        # self.model.to(conf.device)

        running_loss = 0.
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:  # 12
                self.schedule_lr()
            if e == self.milestones[1]:  # 15
                self.schedule_lr()
            if e == self.milestones[2]:  # 18
                self.schedule_lr()
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    # 原型 -> evaluate(self, conf, carray, issame, nrof_folds = 5, tta = False):
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30,
                                                                               self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1

        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)

    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum