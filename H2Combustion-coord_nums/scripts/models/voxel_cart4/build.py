import os
import numpy as np
import warnings
from collections import defaultdict
from sklearn.utils.random import sample_without_replacement

from combust.utils import DataManager, parse_irc_data
from combust.data import ExtensiveEnvironment
from combust.data import extensive_train_loader, extensive_loader_rotwise


def reaction_data(reaction_number, settings, all_sets):
    """

    Parameters
    ----------
    reaction_number: int

    settings: dict
        dict of yaml file

    all_sets: dict

    Returns
    -------

    """

    dir_path = settings['data']['root']

    # file name prefix
    if reaction_number < 10:
        pre = '0%i'%reaction_number
    elif reaction_number >= 10:
        pre = '%i'%reaction_number
    # elif reaction_number == 6:
    #     pre = ['0%ia_irc.npz' % reaction_number, '0%ib_irc.npz' % reaction_number]
    # elif reaction_number == 12:
    #     pre = ['%ia_irc.npz' % reaction_number, '%ib_irc.npz' % reaction_number]

    # read npz files
    aimd = nm = irc = None
    aimd_path = os.path.join(dir_path, '%s_aimd.npz'%pre)
    if os.path.exists(aimd_path):
        aimd = dict(np.load(aimd_path))
    nm_path = os.path.join(dir_path, '%s_nm.npz'%pre)
    if os.path.exists(nm_path):
        nm = dict(np.load(nm_path))
    irc_path = os.path.join(dir_path, '%s_irc.npz'%pre)
    if os.path.exists(irc_path):
        irc = dict(np.load(irc_path))

    # merge aimd and normal mode data
    if settings['data']['normal_mode'] and nm is not None:
        data = dict()
        n_nm = min(settings['data']['size_nmode_max'], nm['R'].shape[0])
        nm_select = sample_without_replacement(nm['R'].shape[0],
                                               n_nm,
                                               random_state=settings['data']['random_states'])
        if aimd is not None:
            for k in aimd.keys():
                data[k] = np.concatenate([aimd[k], nm[k][nm_select]], axis=0)

            assert data['R'].shape[0] == (aimd['R'].shape[0]+n_nm)
        else:
            data = None
            warnings.warn('both AIMD and normal mode data for reaction# %i are missing.'%reaction_number)

    elif aimd is not None:
        data = aimd

    else:
        data = None
        warnings.warn('both AIMD and normal mode data for reaction# %i are missing.'%reaction_number)

    if settings['data']['cgem']:
        assert data['E'].shape == data['CE'].shape
        assert data['F'].shape == data['CF'].shape
        data['E'] = data['E'] - data['CE']
        data['F'] = data['F'] - data['CF']

    if data is not None:
        dm = DataManager()
        dtrain, dval, dtest = dm.split(data,
                                       test_size=settings['data']['test_size'],
                                       val_size=settings['data']['val_size'],
                                       random_states=settings['data']['random_states'],
                                       stratify=None)
    else:
        dtrain, dval, dtest = None, None, None

    # compile data sets
    all_sets['train'].append(dtrain)
    all_sets['val'].append(dval)
    all_sets['test'].append(dtest)
    all_sets['irc'].append(irc)

    return all_sets


def concat_listofdicts(listofdicts, axis=0):
    """

    Parameters
    ----------
    listofdicts: list
    axis: int

    Returns
    -------
    dict

    """
    data = dict()
    for k in listofdicts[0].keys():
        data[k] = np.concatenate([d[k] for d in listofdicts], axis=axis)

    return data


def build_loader(settings, device):
    # data
    reaction_number = settings['data']['reaction']

    if isinstance(reaction_number, int):
        reaction_number = [reaction_number]

    all_sets = defaultdict(list)
    for rxn_n in reaction_number:
        all_sets = reaction_data(rxn_n, settings, all_sets)

    dtrain = concat_listofdicts(all_sets['train'], axis=0)
    dval = concat_listofdicts(all_sets['val'], axis=0)
    dtest = concat_listofdicts(all_sets['test'], axis=0)
    irc = concat_listofdicts(all_sets['irc'], axis=0)

    # downsize training data
    if len(reaction_number) == 1:
        factor = dtrain['R'].shape[0]
    else:
        factor = settings['data']['trsize_perrxn_max']
    n_train = min(len(reaction_number)*factor, dtrain['R'].shape[0])
    n_select = sample_without_replacement(dtrain['R'].shape[0],
                                           n_train,
                                           random_state=settings['data']['random_states'])
    for k in dtrain.keys():
        dtrain[k] = dtrain[k][n_select]

    normalizer = (dtrain['E'].mean(), dtrain['E'].std())

    n_tr_data = dtrain['R'].shape[0]
    n_val_data = dval['R'].shape[0]
    n_irc_data = irc['R'].shape[0]
    n_test_data = dtest['R'].shape[0]
    print("# data (train,val,test,irc): %i, %i, %i, %i"%(n_tr_data,n_val_data,n_val_data,n_irc_data))

    tr_batch_size = settings['training']['tr_batch_size']
    val_batch_size = settings['training']['val_batch_size']
    tr_rotations = settings['training']['tr_rotations']
    val_rotations = settings['training']['val_rotations']

    # freeze rotatios
    # Todo: it seems that we don't need separated tr and val anymore
    # Todo: consider keep_original scenario in the code
    # if settings['training']['tr_frz_rot']:
    #     if settings['training']['saved_angle_path']:
    #         tr_fra_rot = list(np.load(settings['training']['saved_angle_path']))[:tr_rotations+1]
    #     tr_frz_rot = (np.random.uniform(-np.pi, np.pi, size=3)
    #                   for _ in range(tr_rotations+1))
    #     val_frz_rot = tr_frz_rot
    # else:
    #     tr_frz_rot = settings['training']['tr_frz_rot']
    #     val_frz_rot = settings['training']['val_frz_rot']

    # generators
    me = settings['general']['driver']
    if me in ['voxel_polar.py', 'voxel_cart.py', 'schnet.py']:
        # steps
        tr_steps = int(np.ceil(n_tr_data / tr_batch_size)) * (tr_rotations + 1)
        val_steps = int(np.ceil(n_val_data / val_batch_size)) * (val_rotations + 1)
        irc_steps = int(np.ceil(n_irc_data / val_batch_size)) * (val_rotations + 1)
        test_steps= int(np.ceil(n_test_data / val_batch_size)) * (val_rotations + 1)

        env = ExtensiveEnvironment()

        train_gen = extensive_train_loader(data=dtrain,
                                           env_provider=env,
                                           batch_size=tr_batch_size,
                                           n_rotations=tr_rotations,
                                           freeze_rotations=settings['training']['tr_frz_rot'],
                                           keep_original=settings['training']['tr_keep_original'],
                                           device=device,
                                           shuffle=settings['training']['shuffle'],
                                           drop_last=settings['training']['drop_last'])

        val_gen = extensive_train_loader(data=dval,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=settings['training']['shuffle'],
                                         drop_last=settings['training']['drop_last'])

        irc_gen = extensive_train_loader(data=irc,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        test_gen = extensive_train_loader(data=dtest,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        return train_gen, val_gen, irc_gen, test_gen, tr_steps, val_steps, irc_steps, test_steps, normalizer

    if me in ['voxel_cart_rotwise.py']:

        tr_steps = int(np.ceil(n_tr_data / tr_batch_size))
        val_steps = int(np.ceil(n_val_data / val_batch_size))
        irc_steps = int(np.ceil(n_irc_data / val_batch_size))
        test_steps = int(np.ceil(n_test_data / val_batch_size))

        env = ExtensiveEnvironment()

        train_gen = extensive_loader_rotwise(data=dtrain,
                                           env_provider=env,
                                           batch_size=tr_batch_size,
                                           n_rotations=tr_rotations,
                                           freeze_rotations=settings['training']['tr_frz_rot'],
                                           keep_original=settings['training']['tr_keep_original'],
                                           device=device,
                                           shuffle=settings['training']['shuffle'],
                                           drop_last=settings['training']['drop_last'])

        val_gen = extensive_loader_rotwise(data=dval,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=settings['training']['shuffle'],
                                         drop_last=settings['training']['drop_last'])

        irc_gen = extensive_loader_rotwise(data=irc,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        test_gen = extensive_loader_rotwise(data=dtest,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        return train_gen, val_gen, irc_gen, test_gen, tr_steps, val_steps, irc_steps, test_steps, normalizer



# dm = DataManager()
# data = dm.parse(reaction_number, dir_path)
# data = dm.cutoff(data, settings['data']['cutoff'])
# labels = dm.partition(data, reaction_number=reaction_number)
# data, labels = dm.remove(data, labels, ['o'])
# dtrain, dval, dtest = dm.split(data,
#                                test_size=settings['data']['test_size'],
#                                val_size=settings['data']['val_size'],
#                                random_states=settings['data']['random_states'],
#                                stratify=labels)
#
# tr_labels = dm.partition(dtrain, reaction_number=reaction_number)
# if settings['data']['oversmpl']:
#     dtrain = dm.oversample(dtrain,
#                            tr_labels,
#                            settings['data']['oversmpl_l'],
#                            settings['data']['oversmpl_n'])
#
# # irc
# irc = parse_irc_data(settings['data']['irc'])
