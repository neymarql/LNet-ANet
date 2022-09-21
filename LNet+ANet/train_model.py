# 作者：钱隆
# 时间：2022/9/20 23:10


import torch.nn as nn
from sklearn.svm import LinearSVC
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import os
from PIL import Image
from torch.utils import data
import time
import csv


class LNeto(nn.module):
    def __init__(self, num_class=1000):
        super(LNeto, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        identity = x
        out = self.downsample(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))
        out += identity
        return out


class LNets(nn.module):
    def __init__(self):
        super(LNets, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        identity = x
        out = self.downsample(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))
        out += identity
        return out


class ANet(nn.module):
    def __init__(self, num_class=1000):
        super(ANet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3, stride=3, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=80, kernel_size=3, stride=3, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(80, num_class)
        )

    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        out = out.view(80, -1)
        out = self.layer5(out)
        return out


class Dataset_Csv(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, folders, labels, transform=None):
        """Initialization"""
        # self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.folders)

    def read_images(self, path, use_transform):
        image = Image.open(path)
        if use_transform is not None:
            image = use_transform(image)
        return image

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(folder, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


def make_weights_for_balanced_classes(train_dataset, stage='train'):
    targets = []

    targets = torch.tensor(train_dataset)

    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight


def train_lnet(model, model_dir, criterion, optimizer, scheduler, num_epochs=10, current_epoch=0):
    best_logloss = 10.0
    best_epoch = 0
    for epoch in range(current_epoch, num_epochs):
        best_test_logloss = 10.0
        epoch_start = time.time()
        model_out_path = os.path.join(model_dir, str(epoch) + '_xception.ckpt')
        log.write('------------------------------------------------------------------------\n')
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_loss_train = 0.0

            y_scores, y_trues = [], []
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.cuda(), labels.to(torch.float32).cuda()

                if phase == 'train':
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        # labels = labels.unsqueeze(1)
                        loss = criterion(outputs, labels)
                        preds = torch.sigmoid(outputs)
                batch_loss = loss.data.item()
                running_loss += batch_loss
                running_loss_train += batch_loss

                y_true = labels.data.cpu().numpy()
                y_score = preds.data.cpu().numpy()

                if i % 100 == 0:
                    batch_acc = accuracy_score(y_true, np.where(y_score > 0.5, 1, 0))
                    log.write(
                        'Epoch {}/{} Batch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(epoch,
                                                                                                      num_epochs - 1,
                                                                                                      i, len(
                                dataloaders[phase]), phase, batch_loss, batch_acc))
                if (i + 1) % 500 == 0:
                    inter_loss = running_loss_train / 500.0
                    log.write('last phase train loss is {}\n'.format(inter_loss))
                    running_loss_train = 0.0
                    test_loss = val_models(model, criterion, num_epochs, test_list, epoch)
                    if test_loss < best_test_logloss:
                        best_test_logloss = test_loss
                        log.write('save current model {}, Now time is {}, best logloss is {}\n'.format(i, time.asctime(
                            time.localtime(time.time())), best_test_logloss))
                        model_out_paths = os.path.join(model_dir, str(epoch) + str(i) + '_xception.ckpt')
                        torch.save(model.module.state_dict(), model_out_paths)
                    model.train()
                    # scheduler.step()
                    log.write('now lr is : {}\n'.format(scheduler.get_lr()))

                if phase == 'test':
                    y_scores.extend(y_score)
                    y_trues.extend(y_true)
            if phase == 'test':
                epoch_loss = running_loss / (len(test_list) / batch_size)
                y_trues, y_scores = np.array(y_trues), np.array(y_scores)
                accuracy = accuracy_score(y_trues, np.where(y_scores > 0.5, 1, 0))

                log.write(
                    '**Epoch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(epoch, num_epochs - 1, phase,
                                                                                        epoch_loss,
                                                                                        accuracy))
            if phase == 'test' and epoch_loss < best_logloss:
                best_logloss = epoch_loss
                best_epoch = epoch
                torch.save(model.module.state_dict(), model_out_path)

        log.write('Epoch {}/{} Time {}s\n'.format(epoch, num_epochs - 1, time.time() - epoch_start))
    log.write('***************************************************')
    log.write('Best logloss {:.4f} and Best Epoch is {}\n'.format(best_logloss, best_epoch))


def val_models(model, criterion, num_epochs, test_list, current_epoch=0, phase='test'):
    log.write('------------------------------------------------------------------------\n')
    # Each epoch has a training and validation phase
    model.eval()
    running_loss_val = 0.0
    # print(phase)
    y_scores, y_trues = [], []
    for k, (inputs_val, labels_val) in enumerate(dataloaders[phase]):
        inputs_val, labels_val = inputs_val.cuda(), labels_val.to(torch.float32).cuda()
        with torch.no_grad():
            outputs_val = model(inputs_val)
            # labels = labels.unsqueeze(1)
            loss = criterion(outputs_val, labels_val)
            preds = torch.sigmoid(outputs_val)
        batch_loss = loss.data.item()
        running_loss_val += batch_loss

        y_true = labels_val.data.cpu().numpy()
        y_score = preds.data.cpu().numpy()

        if k % 100 == 0:
            batch_acc = accuracy_score(y_true, np.where(y_score > 0.5, 1, 0))
            log.write(
                'Epoch {}/{} Batch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(current_epoch,
                                                                                              num_epochs - 1,
                                                                                              k,
                                                                                              len(dataloaders[phase]),
                                                                                              phase, batch_loss,
                                                                                              batch_acc))
        y_scores.extend(y_score)
        y_trues.extend(y_true)

    epoch_loss = running_loss_val / (len(test_list) / batch_size)
    y_trues, y_scores = np.array(y_trues), np.array(y_scores)
    accuracy = accuracy_score(y_trues, np.where(y_scores > 0.5, 1, 0))
    # model_out_paths = os.path.join(model_dir, str(current_epoch) + '_xception.ckpt')
    # torch.save(model.module.state_dict(), model_out_paths)
    log.write(
        '**Epoch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(current_epoch, num_epochs - 1, phase,
                                                                            epoch_loss,
                                                                            accuracy))
    tn, fp, fn, tp = confusion_matrix(y_trues, np.where(y_scores > 0.5, 1, 0)).ravel()
    log.write(
        '**Epoch {}/{} Stage: {} TNR: {:.2f} FPR: {:.2f} FNR: {:.2f} TPR: {:.2f} \n'.format(current_epoch,
                                                                                            num_epochs - 1, phase,
                                                                                            tn / (fp + tn),
                                                                                            fp / (fp + tn),
                                                                                            fn / (tp + fn),
                                                                                            tp / (tp + fn)))
    log.write('***************************************************\n')
    # model.train()
    return epoch_loss


def base_data(csv_file):
    frame_reader = open(csv_file, 'r')
    csv_reader = csv.reader(frame_reader)

    for f in csv_reader:
        path = f[0]
        label = int(f[1])
        train_label.append(label)
        train_list.append(path)
    log.write(str(len(train_list)) + '\n')


def validation_data(csv_file):
    frame_reader = open(csv_file, 'r')
    fnames = csv.reader(frame_reader)
    for f in fnames:
        path = f[0]
        label = int(f[1])
        test_label.append(label)
        test_list.append(path)
    frame_reader.close()
    log.write(str(len(test_label)) + '\n')


if __name__ == '__main__':
    # Modify the following directories to yourselves
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.time()
    current_epoch = 0
    batch_size = 32
    train_csv = "/hd1/fanhongxing/csv/dfgc_train.csv"  # The train split file
    val_csv = "/hd1/fanhongxing/csv/dfgc_val.csv"  # The validation split file

    #  Output path
    model_dir = '/home/qianlong/DFGC_starterkit-master/Det_model_training/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_name = model_dir.split('/')[-2] + '.log'
    log_dir = os.path.join(model_dir, log_name)
    if os.path.exists(log_dir):
        os.remove(log_dir)
        print('The log file is exit!')

    log = Logger(log_dir, sys.stdout)
    log.write('model : xception   batch_size : 150 frames : 10 \n')
    log.write('pretrain : True   input_size : 299*299\n')

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # Data loading parameters
    params = {'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_list = []
    train_label = []
    log.write('loading train data' + '\n')
    base_data(train_csv)

    ziplist = list(zip(train_list, train_label))
    shuffle(ziplist)
    train_list[:], train_label[:] = zip(*ziplist)

    test_list = []
    test_label = []

    log.write('loading val data' + '\n')
    validation_data(val_csv)

    train_set, valid_set = Dataset_Csv(train_list, train_label, transform=xception_transforms), \
                           Dataset_Csv(test_list, test_label, transform=xception_transforms)

    images_datasets = {}
    images_datasets['train'] = train_label
    images_datasets['test'] = test_label

    weights = {x: make_weights_for_balanced_classes(images_datasets[x], stage=x) for
               x in ['train', 'test']}
    data_sampler = {x: WeightedRandomSampler(weights[x], len(images_datasets[x]), replacement=True) for x in
                    ['train', 'test']}

    image_datasets = {}
    # over sampling
    image_datasets['train'] = data.DataLoader(train_set, sampler=data_sampler['train'], batch_size=batch_size, **params)

    # image_datasets['train'] = data.DataLoader(train_set, batch_size=batch_size, **params)
    image_datasets['test'] = data.DataLoader(valid_set, batch_size=batch_size, **params)

    dataloaders = {x: image_datasets[x] for x in ['train', 'test']}
    datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    model = xception(pretrained=True)
    model.train()
    model = nn.DataParallel(model.cuda())
    criterion = nn.BCEWithLogitsLoss()
    criterion.cuda()

    optimizer_ft = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.5)

    train_model(model=model, model_dir=model_dir, criterion=criterion, optimizer=optimizer_ft,
                scheduler=exp_lr_scheduler,
                num_epochs=5,
                current_epoch=current_epoch)

    elapsed = (time.time() - start)
    log.write('Total time is {}.\n'.format(elapsed))