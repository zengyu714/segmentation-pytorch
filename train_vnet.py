import time
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from vnet import VNet
from utils import *
from inputs import DatasetFromFolder
from functools import partial

# Set net configuration
conf = config()
conf.prefix = 'vnet'
conf.checkpoint_dir += conf.prefix
conf.result_dir += conf.prefix
conf.learning_rate = 3e-7
conf.from_scratch = False
conf.resume_step = -1
# 'dice' or 'nll'
conf.criterion = 'nll'

# GPU configuration
if conf.cuda:
    torch.cuda.set_device(3)
    print('===> Current GPU device is', torch.cuda.current_device())

torch.manual_seed(conf.seed)
if conf.cuda:
    torch.cuda.manual_seed(conf.seed)

def training_data_loader():
    return DataLoader(dataset=DatasetFromFolder(), num_workers=conf.threads, batch_size=conf.batch_size, shuffle=True)

def validation_data_loader():
    return DataLoader(dataset=DatasetFromFolder('./data/val'), num_workers=conf.threads, batch_size=conf.batch_size, shuffle=False)

def get_resume_path():
    """Return latest checkpoints by default otherwise return the specified one."""

    names = [os.path.join(conf.checkpoint_dir, p) for p in os.listdir(conf.checkpoint_dir)]
    require = os.path.join(conf.checkpoint_dir, conf.prefix + '_' + str(conf.resume_step) + '.pth')
    if conf.resume_step == -1:
        return sorted(names, key=os.path.getmtime)[-1]
    elif os.path.isfile(require):
        return require
    raise Exception('\'%s\' dose not exist!' % require)

def save_checkpoints(model, step):
    # Save 20 checkpoints at most.
    names = os.listdir(conf.checkpoint_dir)
    if len(names) >= 20:
        os.remove(os.path.join(conf.checkpoint_dir, names[0]))
    # Recommand: save and load only the model parameters.
    filename = conf.prefix + '_' + str(step) + '.pth'
    torch.save(model.state_dict(), os.path.join(conf.checkpoint_dir, filename))
    print("===> ===> ===> Save checkpoint {} to {}".format(step, filename))

def main():
    print('===> Building vnet...')
    model = VNet(conf.criterion)
    if conf.cuda:
        model = model.cuda()

    if conf.criterion == 'nll':
        # To balance between foreground and backgound for NLL.
        pos_ratio = np.mean([label.float().mean() for image, label in validation_data_loader()])
        bg_weight =  pos_ratio / (1. + pos_ratio)
        fg_weight = 1. - bg_weight
        class_weight = torch.FloatTensor([bg_weight, fg_weight])
        if conf.cuda:
            class_weight = class_weight.cuda()
        print('---> Background weight:', bg_weight)

    print('===> Loss function: {}'.format(conf.criterion))
    print('===> Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    start_i = 1
    if conf.from_scratch:
        model.apply(weights_init)
    else:
        cp = get_resume_path()
        model.load_state_dict(torch.load(cp))
        cp_name = os.path.basename(cp)
        print('---> Loading checkpoint {}...'.format(cp_name))
        start_i = int(cp_name.split('_')[-1].split('.')[0]) + 1
    print('===> Begin training at epoch {}'.format(start_i))

    # Define optimizer, loss is related to conf.criterion.
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)
    total_i = conf.epochs * conf.augment_size

    if conf.criterion == 'nll':
        criterion = partial(F.nll_loss, weight=class_weight)
    else:
        criterion = dice_loss

    def train():
        epoch_loss = 0
        epoch_overlap = 0
        epoch_acc = 0

        # Sets the module in training mode.
        # This has any effect only on modules such as Dropout or BatchNorm.
        model.train()

        for partial_epoch, (image, label) in enumerate(training_data_loader(), 1):
            image, label = Variable(image).float(), Variable(label).float()
            if conf.cuda:
                image = image.cuda()
                label = label.cuda()

            optimizer.zero_grad()

            output = model(image).contiguous()
            target = label.view(-1).long()

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data[0]

            # Compute dice overlap by `argmax`
            pred = output.data.max(1)[1]
            true = target.data.long()

            dice_overlap = 2 * torch.sum(pred * true) / (torch.sum(pred) + torch.sum(true)) * 100
            epoch_overlap += dice_overlap

            # Compute accuracy
            accuracy = pred.eq(true).cpu().sum() / true.numel() * 100
            epoch_acc += accuracy

        avg_loss, avg_dice, avg_acc =np.array([epoch_loss, epoch_overlap, epoch_acc]) / conf.training_size
        print_format = [i, i // conf.augment_size + 1, conf.epochs, avg_loss, avg_dice, avg_acc]
        print('===> Training step {} ({}/{})\tLoss: {:.5f}\tDice Overlap {:.5f}\tAccuracy: {:.5f}'.format(*print_format))
        return avg_loss, avg_dice, avg_acc

    def validate():
        epoch_loss = 0
        epoch_overlap = 0
        epoch_acc = 0

        # Sets the module in evaluation mode
        # This has any effect only on modules such as Dropout or BatchNorm.
        model.eval()

        for image, label in validation_data_loader():
            image, label = Variable(image, volatile=True).float(), Variable(label, volatile=True).float()
            if conf.cuda:
                image = image.cuda()
                label = label.cuda()

            output = model(image).contiguous()
            target = label.view(-1).long()

            loss = criterion(output, target)

            epoch_loss += loss.data[0]

            # Compute dice overlap
            pred = output.data.max(1)[1].float()
            true = target.data.float()
            dice_overlap = 2 * torch.sum(pred * true) / (torch.sum(pred) + torch.sum(true)) * 100
            epoch_overlap += dice_overlap

            # Compute accuracy
            accuracy = pred.eq(true).cpu().sum() / true.numel() * 100
            epoch_acc += accuracy

        avg_loss, avg_dice, avg_acc = np.array([epoch_loss, epoch_overlap, epoch_acc]) / conf.val_size
        print(
            '===> ===> Validation Performance', '-' * 60,
            'Loss: %7.5f' % avg_loss, '-' * 2,
            'Dice Overlap %7.5f' % avg_dice, '-' * 2,
            'Accuracy: %7.5f' % avg_acc
        )
        return avg_loss, avg_dice, avg_acc

    # Save statistics, using dictionary contains 6 list.
    names = np.array([x + '_' + y for x in ['train', 'val'] for y in ['loss', 'dice', 'acc']])
    results_dict = {name: np.zeros(total_i) for name in names}

    for i in range(start_i, total_i + 1):
        # train_results == (train_loss, train_dice, train_acc)
        train_results = train()
        for j, name in enumerate(names[:3]):
            results_dict[name][i - 1] = train_results[j]

        # val_results == (val_loss, val_dice, val_acc)
        val_results = validate()
        for j, name in enumerate(names[3:]):
            results_dict[name][i - 1] = val_results[j]

        if i % 20 == 0:
            save_checkpoints(model, i)
            # np.load('path/to/').item()
            np.save(os.path.join(conf.result_dir, 'results_dict.npy'), results_dict)

if __name__ == '__main__':
    main()
