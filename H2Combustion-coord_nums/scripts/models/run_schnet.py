#!/usr/bin/env python

import os
import shutil
import numpy as np
import pandas as pd
import torch
from combust.utils import read_qchem_files
from scipy.spatial import distance_matrix
from schnetpack import AtomsData
from ase import Atoms
import schnetpack as spk
from torch.optim import Adam
import schnetpack.train as trn
from combust.utils import DataManager

# check if a GPU is available and use a CPU otherwise
if torch.cuda.is_available():
    device = "cuda:1"
else:
    device = "cpu"

print(device)

E_tradeoff = 1.0
F_tradeoff = 1.0
n_features = 64
n_gaussians = 50
n_interactions = 5
cutoff = 5.
cutoff_network = spk.nn.cutoff.CosineCutoff
n_epochs = 500
batch_size = 128
stratify_partitions = False

lr=5e-4

tr_ratio = 0.8
val_ratio = 0.1

negative_dr = True

output_dir = 'local/schnet/E%s_F%s_cf%s_ng%i_nf%i_positivedr'%(str(E_tradeoff), str(F_tradeoff),
                                                   str(cutoff),n_gaussians,n_features)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
shutil.copyfile('run_schnet.py', os.path.join(output_dir, 'script.py'))

# data
n_atoms = 4
dir_path = "/home/moji/Dropbox/AIMD/04/combined/"
dm = DataManager()
data = dm.parse(4,dir_path)
data = dm.cutoff(data, cutoff)
labels = dm.partition(data, reaction_number=4)
data, labels = dm.remove(data, labels, ['o'])
# data = dm.oversample(data, labels, ['ts'], [1])

n_data = data['R'].shape[0]

# ## ASE database
A = np.array(list(range(n_atoms)) * n_atoms).reshape(n_atoms, n_atoms)
neighbours = A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)

atoms = []
property_list = []
for i in range(n_data):
    # atoms
    atoms.append(Atoms('HOHO', positions=data['R'][i]))

    # props
    property_list.append({
        'energy': data['E'][i],
        'forces': data['F'][i]
    })

data = AtomsData(os.path.join(output_dir, 'R4.db'),
                 available_properties=['energy', 'forces'])
data.add_systems(atoms, property_list)
assert len(data) == n_data
print('# database:', len(data))

# ## train/test split

num_train = int(n_data * tr_ratio)
num_val = int(n_data * val_ratio)
num_test = n_data - num_train - num_val

train, val, test = spk.train_test_split(
    data=data,
    num_train=num_train,
    num_val=num_val,
    stratify_partitions=stratify_partitions,
    split_file=os.path.join(output_dir, "split.npz"),
)

train_loader = spk.AtomsLoader(train, batch_size=200, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=100)

print('n_train:', num_train)
print('n_val:', num_val)
print('n_test:', num_test)

means, stddevs = train_loader.get_statistics('energy', divide_by_atoms=True)

print('Mean atomization energy / atom:      {:12.4f} [kcal/mol]'.format(
    means['energy'][0]))
print('Std. dev. atomization energy / atom: {:12.4f} [kcal/mol]'.format(
    stddevs['energy'][0]))

# ## modeling

schnet = spk.representation.SchNet(n_atom_basis=n_features,
                                   n_filters=n_features,
                                   n_gaussians=n_gaussians,
                                   n_interactions=n_interactions,
                                   cutoff=cutoff,
                                   cutoff_network=cutoff_network)

energy_model = spk.atomistic.Atomwise(n_in=n_features,
                                      property='energy',
                                      mean=means['energy'],
                                      stddev=stddevs['energy'],
                                      derivative='forces',
                                      negative_dr=negative_dr)

model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)

# ## training


# tradeoff
# loss function
def loss(batch, result):
    # compute the mean squared error on the energies
    diff_energy = batch['energy'] - result['energy']
    err_sq_energy = torch.mean(diff_energy**2)

    # compute the mean squared error on the forces
    diff_forces = batch['forces'] - result['forces']
    err_sq_forces = torch.mean(diff_forces**2)

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    direction_diff = 1 - cos(batch['forces'], result['forces'])
    direction_loss = torch.mean(direction_diff)

    print('energy loss: ', err_sq_energy, 'force loss: ', err_sq_forces,
          'direction loss: ', direction_loss)

    # build the combined loss function
    err_sq = E_tradeoff * err_sq_energy + \
             F_tradeoff * err_sq_forces #+ direction_loss  # (1 - rho_tradeoff)

    return err_sq


# build optimizer
optimizer = Adam(model.parameters(), lr=lr)

# before setting up the trainer, remove previous training checkpoints and logs
# get_ipython().run_line_magic('rm', '-rf local/output_dir/checkpoints')
# get_ipython().run_line_magic('rm', '-rf local/output_dir/log.csv')

# set up metrics
metrics = [
    spk.metrics.MeanAbsoluteError('energy'),
    spk.metrics.MeanAbsoluteError('forces')
]

# construct hooks
hooks = [
    trn.CSVHook(log_path=output_dir, metrics=metrics),
    trn.ReduceLROnPlateauHook(optimizer,
                              patience=5,
                              factor=0.8,
                              min_lr=1e-6,
                              stop_after_min=True)
]


trainer = trn.Trainer(
    model_path=output_dir,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

# determine number of epochs and train
trainer.train(device=device, n_epochs=n_epochs)

###### test

best_model = torch.load(os.path.join(output_dir, 'best_model'))

test_loader = spk.AtomsLoader(test, batch_size=batch_size)

energy_error = 0.0
forces_error = 0.0

for count, batch in enumerate(test_loader):
    # move batch to GPU, if necessary
    batch = {k: v.to(device) for k, v in batch.items()}

    # apply model
    pred = best_model(batch)

    # calculate absolute error of energies
    tmp_energy = torch.sum(torch.abs(pred['energy'] - batch['energy']))
    tmp_energy = tmp_energy.detach().cpu().numpy(
    )  # detach from graph & convert to numpy
    energy_error += tmp_energy

    # calculate absolute error of forces, where we compute the mean over the n_atoms x 3 dimensions
    tmp_forces = torch.sum(
        torch.mean(torch.abs(pred['forces'] - batch['forces']), dim=(1, 2)))
    tmp_forces = tmp_forces.detach().cpu().numpy(
    )  # detach from graph & convert to numpy
    forces_error += tmp_forces

    # log progress
    percent = '{:3.2f}'.format(count / len(test_loader) * 100)
    print('Progress:', percent + '%' + ' ' * (5 - len(percent)), end="\r")

energy_error /= len(test)
forces_error /= len(test)

print('\nTest MAE:')
print('    energy: {:10.3f} kcal/mol'.format(energy_error))
print('    forces: {:10.3f} kcal/mol/\u212B'.format(forces_error))
