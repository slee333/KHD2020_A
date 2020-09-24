
import os, sys
import argparse
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn

print('Pytorch Version: ' , torch.__version__)

from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from monai.config import print_config
from monai.transforms import \
    Compose, LoadPNG, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom
from monai.networks.nets import densenet121, DenseNet
from monai.metrics import compute_roc_auc

np.random.seed(0)
print_config()

import nsml
from nsml.constants import DATASET_PATH, GPU_NUM


IMSIZE = 256, 128
VAL_RATIO = 0.2
RANDOM_SEED = 1234

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')

    def infer(data):  ## test mode
        X = ImagePreprocessing(data)
        X = np.array(X)
        X = np.expand_dims(X, axis=1)
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(device)
            pred = model.forward(X)
            prob, pred_cls = torch.max(pred, 1)
            pred_cls = pred_cls.tolist()
            #pred_cls = pred_cls.data.cpu().numpy()
        print('Prediction done!\n Saving the result...')
        return pred_cls

    nsml.bind(save=save, load=load, infer=infer)


def DataLoad(imdir):
    impath = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(imdir) for f in files if all(s in f for s in ['.jpg'])]
    img = []
    lb = []
    print('Loading', len(impath), 'images ...')
    for i, p in enumerate(impath):
        img_whole = cv2.imread(p, 0)
        h, w = img_whole.shape
        h_, w_ = h, w//2
        l_img = img_whole[:, w_:2*w_]
        r_img = img_whole[:, :w_]
        _, l_cls, r_cls = os.path.basename(p).split('.')[0].split('_')
        if l_cls=='0' or l_cls=='1' or l_cls=='2' or l_cls=='3':
            img.append(l_img);      lb.append(int(l_cls))
        if r_cls=='0' or r_cls=='1' or r_cls=='2' or r_cls=='3':
            img.append(r_img);      lb.append(int(r_cls))
    print(len(img), 'data with label 0-3 loaded!')
    return img, lb


def ImagePreprocessing(img):
    # 자유롭게 작성
    h, w = IMSIZE
    print('Preprocessing ...')
    for i, im, in enumerate(img):
        tmp = cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA)
        #tmp = tmp / 255.
        img[i] = tmp
    print(len(img), 'images resized!')
    return img


def ParserArguments(args):
    # Setting Hyperparameters
    args.add_argument('--epoch', type=int, default=5)          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=8)      # batch size 설정
    args.add_argument('--learning_rate', type=float, default=1e-5)  # learning rate 설정
    args.add_argument('--num_classes', type=int, default=4)     # 분류될 클래스 수는 4개

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()
    return config.epoch, config.batch_size, config.num_classes, config.learning_rate, config.pause, config.mode

'''
class SampleModelTorch(nn.Module):
    def __init__(self, num_classes=4):
        super(SampleModelTorch, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(256 * 30 * 15, 2048),
                                nn.Linear(2048, 128),
                                nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class PNSDataset(Dataset):
    def __init(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
'''

def split_dataset( image_file_list, image_label_list, valid_frac = 0.1 ):
    valid_frac = 0.1
    trainX, trainY = [], []
    valX, valY = [], []

    for i in range( len(image_label_list) ):
        rann = np.random.random()
        if rann < valid_frac:
            valX.append(image_file_list[i])
            valY.append(image_label_list[i])
        else:
            trainX.append(image_file_list[i])
            trainY.append(image_label_list[i])
    
    return trainX, trainY, valX, valY

class PNSDataset(Dataset):

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
      tf_img = self.transforms( self.image_files[index] )
      return tf_img, self.labels[index]

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    print(GPU_NUM)
    nb_epoch, batch_size, num_classes, learning_rate, ifpause, ifmode = ParserArguments(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #####   Model   #####
    model = densenet121(
        spatial_dims=2,
        in_channels=1,
        out_channels= num_classes,
    ).to(device)
    #model.double()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    val_interval = 1

    # criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bind_model(model)

    if ifpause:  ## for test mode
        print('Inferring Start ...')

        # images, labels = DataLoad(imdir=os.path.join(DATASET_PATH, 'train'))
        # images = ImagePreprocessing(images)
        # images = np.array(images)
        # labels = np.array(labels)

        # print( np.shape(images) )
        # print( np.shape(labels) )

        # dataset = TensorDataset(torch.from_numpy(images).float(), torch.from_numpy(labels).long())
        # subset_size = [len(images) - int(len(images) * VAL_RATIO),int(len(images) * VAL_RATIO)]
        # tr_set, val_set = random_split(dataset, subset_size)
        # batch_train = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
        # batch_val = DataLoader(val_set, batch_size=1, shuffle=False)


        nsml.paused(scope=locals())

    if ifmode == 'train':  ## for train mode
        print('Training start ...')
        # 자유롭게 작성
        images, labels = DataLoad(imdir=os.path.join(DATASET_PATH, 'train'))
        images = ImagePreprocessing(images)
        images = np.array(images)
        labels = np.array(labels)

        ## Define transforms
        train_transforms = Compose([
            AddChannel(),
            ScaleIntensity(),
            RandRotate(degrees=15, prob=0.5, reshape =False),
            RandFlip(spatial_axis=0, prob=0.5),
            ToTensor()
        ])
            #RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),

        val_transforms = Compose([
            AddChannel(),
            ScaleIntensity(),
            ToTensor()
        ])
        
        # Split data
        x_train, y_train, x_test, y_test = split_dataset( images, labels )

        # dataset = PNSDataset(torch.from_numpy(images).float(), torch.from_numpy(labels).long())
        # subset_size = [len(images) - int(len(images) * VAL_RATIO),int(len(images) * VAL_RATIO)]
        # tr_set, val_set = random_split(dataset, subset_size)
        # batch_train = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
        # batch_val = DataLoader(val_set, batch_size=1, shuffle=False)

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        #train_ds = PNSDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long(), train_transforms)
        train_ds = PNSDataset(x_train, torch.from_numpy(y_train).long(), train_transforms)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=10)

        val_ds = PNSDataset(x_test, torch.from_numpy(y_test).long(), train_transforms)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=True, num_workers=10)

        #####   Training loop   #####
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = list()
        metric_values = list()
        for epoch in range(nb_epoch):
            print('-' * 10)
            print("epoch {}/{}".format(epoch + 1, nb_epoch))
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                print("{}/{}, train_loss: {}".format(step, len(train_ds) // train_loader.batch_size, "%.4f" % loss.item() ) )
                epoch_len = len(train_ds) // train_loader.batch_size
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print("epoch {} average loss: {}".format( epoch + 1,  "%.4f" % epoch_loss))

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y = torch.tensor([], dtype=torch.long, device=device)
                    for val_data in val_loader:
                        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                        
                        #val_images = val_images.type(torch.cuda.FloatTensor)
                        
                        y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                        y = torch.cat([y, val_labels], dim=0)
                    auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, add_softmax=True)
                    metric_values.append(auc_metric)
                    acc_value = torch.eq(y_pred.argmax(dim=1), y)
                    acc_metric = acc_value.sum().item() / len(acc_value)
                    if auc_metric > best_metric:
                        best_metric = auc_metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), 'best_metric_model.pth')
                        print('saved new best metric model')
                    
                    print("current epoch: {} current AUC: {} current accuracy: {} best AUC: {} at epoch {}".format(epoch + 1, "%.4f" % auc_metric, "%.4f" % acc_metric, "%.4f" % best_metric, best_metric_epoch ))
        print("train completed, best_metric: {} at epoch: {}".format( "%.4f" % best_metric, best_metric_epoch))