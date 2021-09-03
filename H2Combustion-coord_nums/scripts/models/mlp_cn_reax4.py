import os
import numpy as np
import torch
from torch.optim import Adam
import yaml

from combust.data import loader
from combust.layers import ShellProvider, FermiDirac
from combust.layers import swish, shifted_softplus
from combust.schnetpack import GaussianSmearing
from combust.models import MLPReax4FD
from combust.layers import Dense
from combust.train import Trainer
from combust.utils import DataManager

# settings
settings_path = 'mlp_cn_reax4.yml'
settings = yaml.safe_load(open(settings_path, "r"))

device = torch.device(settings['general']['device'])

derivative = True
energy_weight = 1.0
force_weight = 1.0
mag_f_weight = 0.0
dir_f_weight = 0.0
activation = 'ssp' # 'relu' or 'ssp' or 'swish'

# data
dir_path = "/home/moji/hdd/moji/data/h2reaction/AIMD/04/combined"
dm = DataManager()
data = dm.parse(4,dir_path)
data = dm.cutoff(data, 5)
labels = dm.partition(data, reaction_number=4)
data, labels = dm.remove(data, labels, ['o'])
dtrain, dval, dtest = dm.split(data, test_size=1000,val_size=1000,
              random_states=90, stratify=labels)

tr_labels = dm.partition(dtrain, reaction_number=4)
dtrain = dm.oversample(dtrain,tr_labels, ['ts'], [1])

n_tr_data = dtrain['R'].shape[0]
n_val_data = dval['R'].shape[0]

batch_size = 128
epochs = 1000
steps = int(np.ceil(n_tr_data / batch_size))

normalizer = (-210, 12)

dropout = 0.0
lr = 5e-4
weight_decay = 3e-4

### val
val_gen=True
val_batch_size = 128
val_steps = int(np.ceil(n_val_data / val_batch_size))
###
checkpoint_interval=1
validation_interval=1

lr_scheduler = ('plateau', 5, 5, 0.7, 1e-6)   # '', val_loss_length, patience, decay, min
# lr_scheduler = ('decay', 0.001)  # '', decay


# output files
output_dir = 'local/1'
script_name = 'mlp_cn_reax4.py'

# generators
# env = ExtensiveEnvironment()
train_gen = loader(data=dtrain,
                   batch_size=batch_size,
                   device=device,
                   shuffle=True,
                   drop_last=False)

if val_gen is not None:
    val_gen = loader(data=dval,
                     batch_size=val_batch_size,
                     device=device,
                     shuffle=True,
                     drop_last=False)

# irc_path = "/home/moji/Dropbox/IRC/04_H2O+O_2OH.t/"
# irc_gen = loader(dir_path=irc_path,
#                 batch_size=4,
#                 device=device,
#                 shuffle=False,
#                 drop_last=False)

# activation function
if activation == 'ssp':
    activation = shifted_softplus
elif activation == 'relu':
    activation = torch.nn.ReLU()
elif activation == 'swish':
    activation = swish

# model
shell = ShellProvider()
descriptor = FermiDirac()
gaussian = GaussianSmearing(0,3,100)
dense_modules = torch.nn.ModuleList([
    Dense(100,25, activation=activation),
    Dense(25,16, activation=None),
])
dense_output = torch.nn.ModuleList([
    Dense(16*12,128, activation=activation),
    # Dense(256, 128, activation=activation),
    Dense(128,1, activation=None),
])
model = MLPReax4FD(shell,
                   descriptor,
                   gaussian,
                   dense_modules,
                   dense_output,
                    normalizer=normalizer,
                    device=device,
                    derivative=derivative,
                    create_graph=False)

# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)


# loss
# loss
def custom_loss(preds, batch_data):
    # compute the mean squared error on the energies
    diff_energy = preds['E'] - batch_data.E
    assert diff_energy.shape[1] == 1
    err_sq_energy = torch.mean(diff_energy**2)

    # compute the mean squared error on the forces
    diff_forces = preds['F'] - batch_data.F
    err_sq_forces = torch.mean(diff_forces**2)

    # compute the mean square error on the force magnitudes
    diff_forces = torch.norm(preds['F'], p=2, dim=-1) - torch.norm(batch_data.F, p=2, dim=-1)
    err_sq_mag_forces = torch.mean(diff_forces ** 2)

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    direction_diff = 1 - cos(preds['F'], batch_data.F)
    direction_diff *= torch.norm(batch_data.F, p=2, dim=-1)
    direction_loss = torch.mean(direction_diff)

    print('\n',
          '        energy loss: ', err_sq_energy, '\n',
          '        force loss: ', err_sq_forces, '\n',
          '        mag force loss: ', err_sq_mag_forces, '\n',
          '        direction loss: ', direction_loss)

    # build the combined loss function
    err_sq = energy_weight * err_sq_energy + \
             force_weight * err_sq_forces + \
            mag_f_weight * err_sq_mag_forces + \
            dir_f_weight * err_sq_mag_forces

    return err_sq

# def custom_loss(preds, batch_data):
#     # compute the mean squared error on the energies
#     diff_energy = preds[0] - batch_data.E
#     assert diff_energy.shape[1] == 1
#     err_sq_energy = torch.mean(diff_energy**2)
#
#     # compute the mean squared error on the forces
#     diff_forces = preds[1] - batch_data.F
#     err_sq_forces = torch.mean(diff_forces**2)
#
#     cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
#     direction_diff = 1 - cos(preds[1], batch_data.forces)
#     direction_diff *= torch.norm(batch_data.forces, p=2, dim=-1)
#     direction_loss = torch.mean(direction_diff)
#
#     # print('\n',
#     #       '        energy loss: ', err_sq_energy, '\n',
#     #       '        force loss: ', err_sq_forces, '\n',
#     #       '        direction loss: ', direction_loss)
#
#     # build the combined loss function
#     err_sq = rho_tradeoff * err_sq_energy + \
#              (1 - rho_tradeoff) * err_sq_forces #+ \
#              # direction_loss * 10
#
#     return err_sq


# training
trainer = Trainer(model=model,
                  loss_fn=custom_loss,
                  optimizer=optimizer,
                  requires_dr=derivative,
                  device=device,
                  output_path=output_dir,
                  script_name=script_name,
                  lr_scheduler=lr_scheduler,
                  checkpoint_interval=checkpoint_interval,
                  validation_interval=validation_interval)

trainer.print_layers()

trainer.train(train_generator=train_gen,
              epochs=epochs,
              steps=steps,
              val_generator=val_gen,
              val_steps=val_steps)

print('done!')
