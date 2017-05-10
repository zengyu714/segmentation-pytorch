import time
import torch
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from vnet import VNet
from utils import *
from inputs import DatasetFromFolder

# Set net configuration
conf = config()
conf.prefix = 'vnet'
conf.checkpoint_dir += conf.prefix
conf.learning_rate = 2e-9
conf.from_scratch = False
conf.resume_step = -1

# GPU configuration
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
    model = VNet()
    print('===> Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    if conf.cuda:
        model = model.cuda()

    start_i = 1
    if conf.from_scratch:
        model.apply(weights_init)
    else:
        cp = get_resume_path()
        model.load_state_dict(torch.load(cp))
        cp_name = os.path.basename(cp)
        print('===> Loading checkpoint {}...'.format(cp_name))
        start_i = int(cp_name.split('_')[-1].split('.')[0]) + 1
    print('===> Begin training at epoch {}'.format(start_i))

    # Define loss function (criterion) and optimizer.
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)
    total_i = conf.epochs * conf.augment_size

    for i in range(start_i, total_i + 1):
        train(training_data_loader(), model, criterion, optimizer, i, total_i)
        validate(validation_data_loader(), model, criterion)
        if i % 20 == 0:
            save_checkpoints(model, i)

def train(train_loader, model, criterion, optimizer, i, total_i):
    epoch_loss = 0
    epoch_overlap = 0
    epoch_acc = 0

    # Sets the module in training mode.
    # This has any effect only on modules such as Dropout or BatchNorm.
    model.train()

    for partial_epoch, (image, label) in enumerate(train_loader, 1):
        image, label = Variable(image).float(), Variable(label).float()
        if conf.cuda:
            image = image.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        output_onehot = model(image)

        output = output_onehot.contiguous().view(2, -1)
        target = label.view(-1)

        loss = criterion(target, output)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        # Compute dice overlap by `argmax`
        pred = output.data.max(0)[1].float()
        true = target.data.float()
        dice_overlap = 2 * torch.sum(pred * true) / (torch.sum(pred) + torch.sum(true)) * 100
        epoch_overlap += dice_overlap

        # Compute accuracy
        accuracy = pred.eq(true).cpu().sum() / true.numel() * 100
        epoch_acc += accuracy

    avg_loss, avg_dice, avg_acc = (epoch_loss / conf.training_size), (epoch_overlap / conf.training_size), (epoch_acc / conf.training_size)
    print_format = [i, i // conf.augment_size + 1, conf.epochs, avg_loss, avg_dice, avg_acc]
    print('===> Training step {} ({}/{})\tLoss: {:.5f}\tDice Overlap {:.5f}\tAccuracy: {:.5f}'.format(*print_format))
    return avg_loss, avg_dice, avg_acc

def validate(val_loader, model, criterion):
    epoch_loss = 0
    epoch_overlap = 0
    epoch_acc = 0

    # Sets the module in evaluation mode
    # This has any effect only on modules such as Dropout or BatchNorm.
    model.eval()

    for image, label in val_loader:
        image, label = Variable(image, volatile=True).float(), Variable(label, volatile=True).float()
        if conf.cuda:
            image = image.cuda()
            label = label.cuda()

        output_onehot = model(image)
        output = output_onehot.contiguous().view(2, -1)
        target = label.view(-1)

        loss = criterion(target, output)
        epoch_loss += loss.data[0]

        # Compute dice overlap
        pred = output.data.max(0)[1].float()
        true = target.data.float()
        dice_overlap = 2 * torch.sum(pred * true) / (torch.sum(pred) + torch.sum(true)) * 100
        epoch_overlap += dice_overlap

        # Compute accuracy
        accuracy = pred.eq(true).cpu().sum() / true.numel() * 100
        epoch_acc += accuracy

    avg_loss, avg_dice, avg_acc = (epoch_loss / conf.val_size), (epoch_overlap / conf.val_size), (epoch_acc / conf.val_size)
    print(
        '===> ===> Validation Performance', '-' * 60,
        'Loss: %7.5f' % avg_loss, '-' * 2,
        'Dice Overlap %7.5f' % avg_dice, '-' * 2,
        'Accuracy: %7.5f' % avg_acc
    )
    return avg_loss, avg_dice, avg_acc

if __name__ == '__main__':
    main()
