import os
import numpy as np
import torch
from torch.optim import Adam

from combust.data import ExtensiveEnvironment
from combust.data import extensive_voxel_loader
from combust.layers import ShellProvider, ManyBodyVoxel
from combust.layers import swish, shifted_softplus, EulerRotation
from combust.models import MRDenseNet, Voxel3DRotation, Testnet, Simplenet,Simplenet2
from combust.train import Trainer
from combust.utils import DataManager

device = torch.device('cuda:1')

requires_dr = True
energy_weight = 1.0
force_weight = 1.0
mag_f_weight = 1.0
dir_f_weight = 1.0
network = Simplenet2
model_mode = 'embed_elemental'  # channelize, embed_elemental, split_elemental, shared_elemental, elemental
activation = 'ssp' # 'relu' or 'ssp' or 'swish'
forcewise = False

# data
dir_path = "/home/moji/Dropbox/AIMD/04/combined/"
dm = DataManager()
data = dm.parse(4,dir_path)
data = dm.cutoff(data, 5)
labels = dm.partition(data, reaction_number=4)
data, labels = dm.remove(data, labels, ['o'])
dtrain, dval, dtest = dm.split(data, test_size=0.1,val_size=0.1,
              random_states=90, stratify=labels)

# tr_labels = dm.partition(dtrain, reaction_number=4)
# dtrain = dm.oversample(dtrain,tr_labels, ['ts'], [1])

n_tr_data = dtrain['R'].shape[0]
n_val_data = dval['R'].shape[0]

batch_size = 128
epochs = 2000

steps = int(np.ceil(n_tr_data / batch_size))

normalizer = (-210, 12)
atom_types = [1, 8]
# grid_length = [1., 2., 4., 6., 14.]
grid_length = [2., 3., 4., 6., 10]
grid_size = 20
sigma = 1 / 3

dropout = 0.0
lr = 5e-4
weight_decay = 0#3e-4

### val
val_gen=True
val_batch_size = 32
val_steps = int(np.ceil(n_val_data / val_batch_size))
###
checkpoint_interval=1
validation_interval=1

lr_scheduler = ('plateau', 5, 5, 0.8, 1e-6)   # '', val_loss_length, patience, decay, min
# lr_scheduler = ('decay', 0.05)  # '', decay


# output files
output_dir = 'local/voxelnet_3/%s_%s_ew%sfw%smfw%sdfw%s_g%i-l%i:%i_act-%s_dec-%s_wbnorm'%(model_mode,
                                                                                network.__name__,
                                                                                str(energy_weight),
                                                                                str(force_weight),
                                                                                str(mag_f_weight),
                                                                                str(dir_f_weight),
                                                                                grid_size,
                                                                                len(grid_length),max(grid_length),
                                                                                activation,
                                                                                str(lr_scheduler[1])
                                                                                )
script_name = 'script.voxelnet+rotation.py'

# generators
# env = ExtensiveEnvironment()

train_gen = extensive_voxel_loader(data=dtrain,
                                   env_provider=None,
                                   batch_size=batch_size,
                                   device=device,
                                   shuffle=True,
                                   drop_last=False)

if val_gen is not None:
    val_gen = extensive_voxel_loader(data=dval,
                                     env_provider=None,
                                     batch_size=val_batch_size,
                                     device=device,
                                     shuffle=True,
                                     drop_last=False)

# irc_path = "/home/moji/Dropbox/IRC/04_H2O+O_2OH.t/"
# irc_gen = extensive_irc_loader(dir_path=irc_path,
#                                env_provider=None,
#                                batch_size=4,
#                                n_rotations=0,
#                                device=device,
#                                shuffle=False,
#                                drop_last=False)

# activation function
if activation == 'ssp':
    activation = shifted_softplus
elif activation == 'relu':
    activation = torch.nn.ReLU()
elif activation == 'swish':
    activation = swish

# model
rotation = EulerRotation()
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
    mr_densenet = network(in_channels=in_channels, dropout=dropout,
                          activation=activation,grid_size=grid_size)

elif model_mode=='embed_elemental':
    in_channels = len(grid_length)
    mr_densenet = network(in_channels=in_channels, dropout=dropout,
                          activation=activation,grid_size=grid_size)

elif model_mode =='elemental':
    in_channels = len(atom_types) * len(grid_length)
    mr_densenet = torch.nn.ModuleList([
        network(in_channels=in_channels, dropout=dropout,
                activation=activation,grid_size=grid_size)
        for _ in range(len(atom_types))
    ])

elif model_mode in ['split_elemental']:
    in_channels = len(grid_length)
    mr_densenet = torch.nn.ModuleList([
        network(in_channels=in_channels, dropout=dropout,activation=activation,grid_size=grid_size)
        for _ in range(len(atom_types))
    ])

elif model_mode=='shared_elemental':
    in_channels = len(grid_length)
    mr_densenet = torch.nn.ModuleList([
        network(in_channels=in_channels, dropout=dropout,activation=activation)
    ]
        * len(atom_types)
    )

model = Voxel3DRotation(
                rotation,
                shell,
                mb_voxel,
                mr_densenet,
                mode=model_mode,
                forcewise=forcewise,
                normalizer=normalizer,
                device=device,
                requires_dr=requires_dr,
                create_graph=False)

# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)


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


# training
trainer = Trainer(model=model,
                  loss_fn=custom_loss,
                  optimizer=optimizer,
                  requires_dr=requires_dr,
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
