import os
import numpy as np
import torch
from torch.optim import Adam
import yaml

from combust.schnetpack import SchNet, Atomwise, CosineCutoff, HardCutoff
from combust.models import AtomisticModel
from combust.train import Trainer

from build import build_loader

# torch.autograd.set_detect_anomaly(True)

# settings
settings_path = 'schnet/schnet.yml'
settings = yaml.safe_load(open(settings_path, "r"))

# device
device = torch.device(settings['general']['device'])

# data
train_gen, val_gen, irc_gen, test_gen, \
tr_steps, val_steps, irc_steps, test_steps, normalizer = build_loader(settings,device)

# model
if settings['model']['cutoff_network'] == 'cosine':
    cutoff_network = CosineCutoff
else:
    cutoff_network = HardCutoff

normalizer = (torch.tensor(normalizer[0],dtype=torch.float32, device=device),
              torch.tensor(normalizer[1],dtype=torch.float32, device=device))

schnet = SchNet(n_atom_basis=settings['model']['n_features'],
               n_filters=settings['model']['n_features'],
               n_gaussians=settings['model']['n_gaussians'],
               n_interactions=settings['model']['n_interactions'],
               cutoff=settings['data']['cutoff'],
               cutoff_network=cutoff_network)

energy_model = Atomwise(n_in=settings['model']['n_features'],
                          property='E',
                          mean=normalizer[0],
                          stddev=normalizer[1],
                          derivative='F',
                          negative_dr=settings['model']['negative_dr'])

model = AtomisticModel(representation=schnet, output_modules=[energy_model])

# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params,
                 lr=settings['training']['lr'],
                 weight_decay=settings['training']['weight_decay'])

# loss
w_energy = settings['model']['w_energy']
w_force = settings['model']['w_force']
w_f_mag = settings['model']['w_f_mag']
w_f_dir = settings['model']['w_f_dir']


def custom_loss(preds, batch_data, w_e=w_energy, w_f=w_force, w_fm=w_f_mag, w_fd=w_f_dir):
    # compute the mean squared error on the energies
    diff_energy = preds['E'] - batch_data.E
    assert diff_energy.shape[1] == 1
    err_sq_energy = torch.mean(diff_energy**2)

    # compute the mean squared error on the forces
    diff_forces = preds['F'] - batch_data.F
    err_sq_forces = torch.mean(diff_forces**2)

    # compute the mean square error on the force magnitudes
    if w_fm > 0:
        diff_forces = torch.norm(preds['F'], p=2, dim=-1) - torch.norm(batch_data.F, p=2, dim=-1)
        err_sq_mag_forces = torch.mean(diff_forces ** 2)

    if w_fd > 0:
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        direction_diff = 1 - cos(preds['F'], batch_data.F)
        direction_diff *= torch.norm(batch_data.F, p=2, dim=-1)
        direction_loss = torch.mean(direction_diff)

    print('\n',
          '        energy loss: ', err_sq_energy, '\n',
          '        force loss: ', err_sq_forces, '\n')
          # '        mag force loss: ', err_sq_mag_forces, '\n',
          # '        direction loss: ', direction_loss)

    # build the combined loss function
    err_sq = w_e * err_sq_energy + \
             w_f * err_sq_forces
             # w_fm * err_sq_mag_forces + \
             # w_fd * err_sq_mag_forces

    return err_sq

# training
trainer = Trainer(model=model,
                  loss_fn=custom_loss,
                  optimizer=optimizer,
                  requires_dr=settings['model']['requires_dr'],
                  device=device,
                  yml_path=settings['general']['me'],
                  output_path=settings['general']['output'],
                  script_name=settings['general']['driver'],
                  lr_scheduler=settings['training']['lr_scheduler'],
                  checkpoint_log=settings['checkpoint']['log'],
                  checkpoint_val=settings['checkpoint']['val'],
                  checkpoint_model=settings['checkpoint']['model'],
                  verbose=settings['checkpoint']['verbose'])

trainer.print_layers()

# tr_steps=3; val_steps=3; irc_steps=3; test_steps=3

trainer.train(
              train_generator=train_gen,
              epochs=settings['training']['epochs'],
              steps=tr_steps,
              val_generator=val_gen,
              val_steps=val_steps,
              irc_generator=irc_gen,
              irc_steps=irc_steps,
              test_generator=test_gen,
              test_steps=test_steps)

print('done!')
