import torch
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from vnet import VNet
from utils import *
from inputs import load_training_set, load_validation_set

# GPU configuration
torch.cuda.set_device(1)
print('===> Current GPU device is', torch.cuda.current_device())

# Set net configuration
conf = config()
conf.prefix = 'vnet'
conf.checkpoint_dir += conf.prefix
conf.from_scratch = True

def get_checkpoints():
    """Return latest checkpoints by default otherwise return the specified one."""

    names = os.listdir(conf.checkpoint_dir)
    step = conf.resume_step
    require = conf.prefix + '_' + str(step) + '.pth'
    if step == -1:
        res = sorted(names)[-1]
    elif os.path.isfile(conf.checkpoint_dir + require):
        res = require
    return os.path.join(conf.checkpoint_dir, res)

def save_checkpoints(step):
    # Save 20 checkpoints at most.
    names = os.listdir(conf.checkpoint_dir)
    if len(names) >= 20:
        os.remove(os.path.join(conf.checkpoint_dir, names[0]))

    filename = conf.prefix + '_' + str(step) + '.pth'
    torch.save(model, os.path.join(conf.checkpoint_dir, filename))
    print("===> ===> ===> Save checkpoint {} to {}".format(step, filename))

torch.manual_seed(conf.seed)
if conf.cuda:
    torch.cuda.manual_seed(conf.seed)

print('===> Loading datasets...')
train_set = load_training_set()
val_set = load_validation_set()

training_data_loader = DataLoader(dataset=train_set, num_workers=conf.threads, batch_size=conf.batch_size, shuffle=True)
validation_data_loader = DataLoader(dataset=val_set, num_workers=conf.threads, batch_size=conf.batch_size, shuffle=False)

print('===> Building vnet...')
model = VNet()
print('===> Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
if conf.cuda:
    model = model.cuda()

start_i = 1
if conf.from_scratch:
    model.apply(weights_init)
else:
    cp = get_checkpoints()
    print('===> Loading checkpoint {}...'.format(os.path.basename(cp)))
    checkpoint = torch.load(cp)
    start_i = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
print('===> Begin training at step {}'.format(start_i))

optimizer = optim.Adam(model.parameters(),  lr=conf.learning_rate)

def train():
    total_i = conf.epochs * conf.augment_size
    for i in range(start_i, total_i + 1):
        epoch_loss = 0
        epoch_overlap = 0
        epoch_acc = 0
        for step, (image, label) in enumerate(training_data_loader, 1):
            image, label = Variable(image).float(), Variable(label).float()
            if conf.cuda:
                image = image.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            output_onehot = model(image)

            output = output_onehot.contiguous().view(-1, 2)
            target = label.view(-1)

            loss = dice_loss(target, output)
            epoch_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            # Compute dice overlap by `argmax`
            pred = output.data.max(1)[1].float()
            true = target.data.float()
            dice_overlap = 2 * torch.sum(pred * true) / (torch.sum(pred) + torch.sum(true)) * 100
            epoch_overlap += dice_overlap

            # Compute accuracy
            accuracy = pred.eq(target.data).cpu().sum() / target.numel() * 100
            epoch_acc += accuracy

        print_format = [
            i,
            i // total_i + 1,
            total_i,
            epoch_loss / len(training_data_loader),
            epoch_overlap / len(training_data_loader),
            epoch_acc / len(training_data_loader)
        ]
        print('===> Training epoch {} ({}/{})\t\tLoss: {:.5f}\tDice Overlap {:.5f}\tAccyracy: {:.5f}'.format(*print_format))

        if i % 20 == 0:
            validate(i)
            if i % 100 == 0:
                save_checkpoints(i)

def validate(i):
    epoch_loss = 0
    epoch_overlap = 0
    epoch_acc = 0
    for step, (image, label) in enumerate(validation_data_loader, 1):
        image, label = Variable(image, volatile=True).float(), Variable(label, volatile=True).float()
        if conf.cuda:
            image = image.cuda()
            label = label.cuda()

        output_onehot = model(image)
        output = output_onehot.contiguous().view(-1, 2)
        target = label.view(-1)

        loss = dice_loss(target, output)
        epoch_loss += loss.data[0]

        # Compute dice overlap
        pred = output.data.max(1)[1].float()
        true = target.data.float()
        dice_overlap = 2 * torch.sum(pred * true) / (torch.sum(pred) + torch.sum(true)) * 100
        epoch_overlap += dice_overlap

        # Compute accuracy
        accuracy = pred.eq(target.data).cpu().sum() / target.numel() * 100
        epoch_acc += accuracy

    print_format = [
        i,
        epoch_loss / len(validation_data_loader),
        epoch_overlap / len(validation_data_loader),
        epoch_acc / len(validation_data_loader)
    ]
    print('===> ===> Testing at epoch {}\t\tLoss: {:.5f}\tDice Overlap {:.5f}\tAccyracy: {:.5f}'.format(*print_format))

if __name__ == '__main__':
    train()
