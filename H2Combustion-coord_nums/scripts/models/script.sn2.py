import os
import numpy as np
import torch
from torch.optim import Adam

from combust.data import ExtensiveEnvironment
from combust.data import extensive_train_loader, extensive_irc_loader
from combust.layers import ShellProvider, ManyBodyVoxel
from combust.layers import swish, shifted_softplus
from combust.models import MRDenseNet, Voxel3D, Testnet, Simplenet2
from combust.train import Trainer
from combust.utils import Sn2

device = torch.device('cuda:0')

derivative = False
energy_weight = 1.0
force_weight = 0.0
network = Simplenet2
model_mode = 'channelize'  # channelize, embed_elemental, split_elemental, shared_elemental, elemental
activation = 'ssp' # 'relu' or 'ssp' or 'swish'
forcewise = False

# data
data_path = "/home/moji/Documents/repos_dev/3_ff/physnet/data/cl_br.npz"
sn2 = Sn2(data_path)
sn2.split(test_size=0.4,val_size=0.1,
              random_states=90, stratify=None)

n_tr_data = sn2.get_split('train')['R'].shape[0]
n_val_data = sn2.get_split('val')['R'].shape[0]

batch_size = 40
epochs = 500
n_rotations = 3
freeze_rotations = False
steps = int(n_tr_data / batch_size) * (n_rotations + 1)

normalizer = (-8.9, 5.1)
atom_types = [1, 6, 17, 35]
# grid_length = [1., 2., 4., 6., 14.]
grid_length = [1., 2., 4., 12.]
grid_size = 24
sigma = 1 / 3

dropout = 0.0
lr = 1e-3
weight_decay = 3e-4

### val
val_gen=True
val_batch_size = 16
val_n_rotations = 7
val_steps = int(n_val_data / val_batch_size) * (val_n_rotations+1)
###
checkpoint_interval=1
validation_interval=1

# lr_scheduler = ('plateau', 5, 3, 0.7, 1e-6)   # '', val_loss_length, patience, decay, min
lr_scheduler = ('decay', 0.05)  # '', decay


# output files
output_dir = 'local/sn2_clbr/%s_%s_ew%sfw%s_g16-l1246_act-%s_dec-%s_forcewise-%i'%(model_mode,
                                                                                network.__name__,
                                                                                str(energy_weight),
                                                                                str(force_weight),
                                                                                activation,
                                                                                str(lr_scheduler[1]),
                                                                                int(forcewise))
script_name = 'script.sn2.py'

# generators
# env = ExtensiveEnvironment()
data = sn2.get_split('train')
train_gen = extensive_train_loader(data=data,
                                   env_provider=None,
                                   batch_size=batch_size,
                                   n_rotations=n_rotations,
                                   freeze_rotations=freeze_rotations,
                                   device=device,
                                   shuffle=True,
                                   drop_last=False)

if val_gen is not None:
    val_data = sn2.get_split('val')
    val_gen = extensive_train_loader(data=val_data,
                                     env_provider=None,
                                     batch_size=val_batch_size,
                                     n_rotations=val_n_rotations,
                                     freeze_rotations=True,
                                     device=device,
                                     shuffle=True,
                                     drop_last=False)

irc_path = "/home/moji/Dropbox/IRC/04_H2O+O_2OH.t/"
irc_gen = extensive_irc_loader(dir_path=irc_path,
                               env_provider=None,
                               batch_size=4,
                               n_rotations=0,
                               device=device,
                               shuffle=False,
                               drop_last=False)

# activation function
if activation == 'ssp':
    activation = shifted_softplus
elif activation == 'relu':
    activation = torch.nn.ReLU()
elif activation == 'swish':
    activation = swish

# model
shell = ShellProvider()
mb_voxel = ManyBodyVoxel(mode = model_mode,
                         atom_types=atom_types,
                         grid_length=grid_length,
                         grid_size=grid_size,
                         sigma=torch.tensor(sigma,
                                            device=device,
                                            dtype=torch.float))
if model_mode=='channelize':
    in_channels = len(atom_types) * len(grid_length)
    mr_densenet = network(in_channels=in_channels, dropout=dropout,activation=activation,
                          grid_size=grid_size)

elif model_mode=='embed_elemental':
    in_channels = len(grid_length)
    mr_densenet = network(in_channels=in_channels, dropout=dropout,activation=activation)

elif model_mode =='elemental':
    in_channels = len(atom_types) * len(grid_length)
    mr_densenet = torch.nn.ModuleList([
        network(in_channels=in_channels, dropout=dropout,activation=activation)
        for _ in range(len(atom_types))
    ])

elif model_mode in ['split_elemental']:
    in_channels = len(grid_length)
    mr_densenet = torch.nn.ModuleList([
        network(in_channels=in_channels, dropout=dropout,activation=activation)
        for _ in range(len(atom_types))
    ])

elif model_mode=='shared_elemental':
    in_channels = len(grid_length)
    mr_densenet = torch.nn.ModuleList([
        network(in_channels=in_channels, dropout=dropout,activation=activation)
    ]
        * len(atom_types)
    )

model = Voxel3D(shell,
                mb_voxel,
                mr_densenet,
                mode=model_mode,
                forcewise=forcewise,
                normalizer=normalizer,
                device=device,
                derivative=derivative,
                create_graph=False)

# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)


# loss
def custom_loss(preds, batch_data):
    # compute the mean squared error on the energies
    diff_energy = preds[0] - batch_data.energy
    assert diff_energy.shape[1] == 1
    err_sq_energy = torch.mean(diff_energy**2)

    # compute the mean squared error on the forces
    diff_forces = preds[1] - batch_data.forces
    err_sq_forces = torch.mean(diff_forces**2)

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    direction_diff = 1 - cos(preds[1], batch_data.forces)
    direction_diff *= torch.norm(batch_data.forces, p=2, dim=-1)
    direction_loss = torch.mean(direction_diff)

    print('\n',
          '        energy loss: ', err_sq_energy, '\n',
          '        force loss: ', err_sq_forces, '\n',
          '        direction loss: ', direction_loss)

    # build the combined loss function
    err_sq = energy_weight * err_sq_energy + \
             force_weight * err_sq_forces #+ \
             # direction_loss * 10

    return err_sq


# training
trainer = Trainer(model=model,
                  loss_fn=custom_loss,
                  optimizer=optimizer,
                  derivative=derivative,
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
