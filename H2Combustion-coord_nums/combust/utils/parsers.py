"""
Miscellaneous utility functions
"""

import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from ase.io import iread
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix

from .calculator import coord_num


def read_qchem_files(path, skiprows=(0, 0)):
    """
    reads QChem files: tested for Energy, NucCarts, and NucForces

    Parameters
    ----------
    path: str
        path to the file

    skiprows: tuple
        This must be a tuple of two integers.
        The first element indicates number of lines to skip from top,
        and the second element is the number of lines at bottom of file to skip.

    Returns
    -------
    arraylike: 2D array of corresponding data

    """

    # read energy file to dataframe
    df = pd.read_csv(path,
                     skiprows=skiprows[0],
                     skipfooter=skiprows[1],
                     delim_whitespace=True,
                     header=None,
                     engine='python')

    return df.values


class DataManager(object):
    """
    The main class to parse reaction data, partition, oversample, remove outliers and visualize the sensible stats.

    Parameters
    ----------
    max_n_atoms: int
        maximum number of atoms in all sub reactions.

    """
    def __init__(self, max_n_atoms=None):
        self.max_n_atoms = max_n_atoms

    def parse(self, reaction_number, dir_path):
        """
        Main function to parse data by reaction number and main directory path.

        Parameters
        ----------
        reaction_number: int
            The reaction number

        dir_path: str
            relative path to the AIMD data for any of reactions.
            For example: "local_path_to/AIMD/04/combined/"

        Returns
        -------
        dict: A dictionary with following keys:
            'R': positions, array of (n_snapshots, n_atoms, 3)
            'Z': atomic numbers, array of (n_snapshots, n_atoms)
            'E': energy, array of (n_snapshots, 1)
            'F': forces, array of (n_snapshots, n_atoms, 3)

        """
        if reaction_number == 4:
            data = parse_reax4(dir_path)
            return data
        else:
            raise NotImplementedError('The reaction number is not available yet.')

    def partition(self, data, reaction_number):
        """
        partition data based on the labels that can be extracted here based on the coordination numbers.

        Parameters
        ----------
        data: dict
        reaction_number: int

        Returns
        -------
        ndarray: assigned labels, 1D array of the following labels: 'r','p','ts', 'ts_', 'o'

        """
        if reaction_number == 4:
            labels = self._partition_reax4(data)
            return labels
        else:
            raise NotImplementedError('The reaction number is not available yet.')

    def _coordination_number_reax4(self, data):
        """
        #todo: generalize. only good for reaction #4
        Parameters
        ----------
        data

        Returns
        -------

        """
        carts = data['R'].reshape(data['R'].shape[0], 12)

        def get_distances(atoms, n_atoms):
            atoms = atoms.reshape(n_atoms, 3)
            dist = distance_matrix(atoms, atoms, p=2)
            return dist[np.triu_indices(n_atoms, k=1)]

        dist = np.apply_along_axis(get_distances, 1, carts, n_atoms=4)
        dist = pd.DataFrame(dist, columns=['H1_O1', 'H1_H2', 'H1_O2', 'O1_H2', 'O1_O2', 'H2_O2'])
        cn1 = coord_num([dist.H1_O1.values, dist.O1_H2.values], mu=1.0, sigma=3.0)
        cn2 = coord_num([dist.H1_O2.values, dist.H2_O2.values], mu=1.0, sigma=3.0)

        return cn1, cn2, dist

    def _partition_reax4(self, data):
        """
        only for reaction #4
        #todo: move to reax4 sub class
        """

        cn1, cn2, _ = self._coordination_number_reax4(data)

        cn = pd.DataFrame()
        cn['cn1'] = cn1
        cn['cn2'] = cn2
        cn['partition'] = 'ts_'

        # Reactants
        ind_ = cn[
            cn.cn1.map(lambda x: x > 0.4 and x < 0.6) &
            cn.cn2.map(lambda x: x > 0.4 and x < 0.6)
            ].index

        cn.loc[ind_, 'partition'] = 'r'

        # TS
        ind_ = cn[
            cn.cn1.map(lambda x: x >= 0.6) &
            cn.cn2.map(lambda x: x >= 0.2)
            ].index

        cn.loc[ind_, 'partition'] = 'ts'

        # Products
        ind_1 = cn[
            cn.cn1.map(lambda x: x >= 0.6) &
            cn.cn2.map(lambda x: x < 0.2)
            ].index

        ind_2 = cn[
            cn.cn1.map(lambda x: x <= 0.2) &
            cn.cn2.map(lambda x: x > 0.6)
            ].index

        cn.loc[list(ind_1) + list(ind_2), 'partition'] = 'p'

        # Outliers
        ind_1 = cn[
            cn.cn1.map(lambda x: x < 0.6) &
            cn.cn2.map(lambda x: x < 0.2)
            ].index

        ind_2 = cn[
            cn.cn1.map(lambda x: x < 0.4) &
            cn.cn2.map(lambda x: x < 0.4 and x > 0.2)
            ].index

        cn.loc[list(ind_1) + list(ind_2), 'partition'] = 'o'

        return cn.partition.values

    def oversample(self, data, labels, labels2oversample, reps):
        """
        oversampling of particular labels.
        The oversampling should be done on the training data only to avoid data leakage to test or val sets.

        Parameters
        ----------
        data: dict
        labels: ndarray
        labels2oversample: list of str
            list of labels (str): 'r','ts','ts_', 'p'

        reps: list of int
            the number of times to copy each label, accordingly (int)

        """
        assert data['R'].shape[0] == labels.shape[0], 'The input data and labels must have the same length.'
        assert len(labels2oversample) == len(reps), 'The length of labels2oversample and reps must be the same.'

        data = copy.deepcopy(data)
        labels = pd.DataFrame(labels)

        for i in range(len(labels2oversample)):
            over_inds = labels[labels[0].map(lambda x: x in [labels2oversample[i]])].index.tolist()
            for key in data:
                concat_list = [data[key]] + [data[key][over_inds]]*reps[i]
                data[key] = np.concatenate(concat_list, axis=0)

        return data

    def remove(self, data, labels, labels2remove):
        """

        Parameters
        ----------
        data: dict
            data dictionary

        labels: ndarray
            1D array of labels

        labels2remove: list
            list of labels (str: 'r','ts','ts_', 'p', 'o'), which need to be removed.

        """
        assert data['R'].shape[0] == labels.shape[0], 'The input data and labels must have same length.'

        labels = pd.DataFrame(labels)
        inds = labels[labels[0].map(lambda x: x in labels2remove)].index.tolist()
        for key in data:
            data[key] = np.delete(data[key], inds, axis=0)

        labels = np.delete(labels.values.reshape(-1,1), inds, axis=0)

        return data, labels

    def split(self, data, test_size, val_size, random_states, stratify=None):
        """

        Parameters
        ----------
        data: dict
        test_size
        val_size
        random_states
        stratify: None or labels

        Returns
        -------
        dict: train data
        dict: val data
        dict: test data

        """

        tr_ind, te_ind = train_test_split(list(range(data['R'].shape[0])),
                                          test_size=test_size,
                                          random_state=random_states,
                                          stratify=stratify)

        if stratify is not None:
            stratify_new = stratify[tr_ind]
        else:
            stratify_new = None

        tr_ind, val_ind = train_test_split(tr_ind,
                                          test_size=val_size,
                                          random_state=random_states,
                                          stratify=stratify_new)

        train=dict(); val=dict(); test=dict()
        for key in data:
            train[key] = data[key][tr_ind]
            val[key] = data[key][val_ind]
            test[key] = data[key][te_ind]

        if stratify is not None:
            train['L'] = stratify[tr_ind]
            val['L'] = stratify[val_ind]
            test['L'] = stratify[te_ind]

        return train, val, test

    def histogram(self, data, output_path):
        """
        plot histograms of energy and forces.

        Parameters
        ----------
        data: dict
        output_path: str
            The full path to the output directory.
            will be created if it doesn't exist.

        Returns
        -------

        """
        # histogram
        d = {'E':'[kcal/mol]', 'F':'[kcal/mol/\u212B]'}
        for key in d:
            fig, ax = plt.subplots()

            plt.hist(data[key].flatten(), bins=50)
            plt.xlabel('%s  %s'%(key, d[key]))
            plt.ylabel('density')

            # grid line
            ax.set_axisbelow(True)
            ax.grid(color='gray', linestyle='dashed')

            # plt.show()

            plt.tight_layout()

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            fig.savefig(os.path.join(output_path, "%s_hist.eps"%key), close=True, verbose=True)
            fig.savefig(os.path.join(output_path, "%s_hist.png"%key), close=True, verbose=True)
            plt.close(fig)

    def cutoff(self, data, r):
        """
        currently only good for reaction 4

        """
        _,_, dist = self._coordination_number_reax4(data)
        keep_inds = np.invert((dist > r).values.any(axis=-1))

        data = copy.deepcopy(data)
        for key in data:
            data[key] = data[key][keep_inds]

        return data


def parse_reax4(dir_path):
    """
    The main parser function only good for reax4

    Parameters
    ----------
    dir_path
    Returns
    -------
    """
    n_atoms = 4

    energy = []
    forces = []
    carts = []

    for length in ['short', 'long', 'very_short', 'short/more']:
        for temp in [500, 1000, 2000, 3000]:
            if length == 'very_short':
                n_files = 20
            else:
                n_files = 10
            for it in range(1, n_files + 1):
                path = os.path.join(dir_path, str(temp), length, str(it),
                                    'AIMD')

                # read files, skip first row: header line
                tmp_energy = read_qchem_files(os.path.join(path, 'TandV'),
                                              (1, 0))
                tmp_forces = read_qchem_files(os.path.join(path, 'NucForces'),
                                              (1, 0))
                tmp_carts = read_qchem_files(os.path.join(path, 'NucCarts'),
                                             (1, 0))

                # sanity check
                if length in ['short', 'short/more']:
                    n_data = 50
                elif length == 'very_short':
                    n_data = 25
                else:
                    n_data = 100

                assert tmp_energy.shape[0] == tmp_forces.shape[0] == n_data
                assert tmp_energy.shape[1] == 5
                assert tmp_forces.shape[1] == n_atoms * 3 + 1

                assert tmp_carts.shape[0] == tmp_energy.shape[
                    0] + 1  # cartesian coordinates has one extra time step
                assert tmp_carts.shape[1] == n_atoms * 3 + 1
                tmp_carts = tmp_carts[:-1, :]

                # check time steps
                np.testing.assert_almost_equal(tmp_energy[:, 0],
                                               tmp_forces[:, 0],
                                               decimal=4)
                np.testing.assert_almost_equal(tmp_energy[:, 0],
                                               tmp_carts[:, 0],
                                               decimal=4)

                # update data collection
                energy.append(tmp_energy)
                forces.append(tmp_forces)
                carts.append(tmp_carts)

    # concatenate numpy arrays
    energy = np.concatenate(energy, axis=0)
    forces = np.concatenate(forces, axis=0)
    carts = np.concatenate(carts, axis=0)

    # atomic energies (au)
    H = -0.5004966690 * 627.5
    O = -75.0637742413 * 627.5

    # units ( au to kcal/mol)
    energy = energy * 627.5 - 2 * H - 2 * O  # relative energy
    forces = forces * 1182.683

    energy = energy[:, 1].reshape(-1, 1)
    forces = forces[:, 1:].reshape(forces.shape[0], n_atoms, 3)
    # forces are given as the gradients divided by the nuclear mass in amu
    # H: 1.00784
    # O: 15.999
    # amu to au: 1822.88818
    forces[:, [0, 2], :] = forces[:, [0, 2], :] * 1.00784 * 1822.88818
    forces[:, [1, 3], :] = forces[:, [1, 3], :] * 15.999 * 1822.88818

    carts = carts[:, 1:].reshape(forces.shape[0], n_atoms, 3)
    n_data = carts.shape[0]
    atomic_numbers = np.concatenate([[1, 8, 1, 8]*n_data], axis=0).reshape(n_data, 4)

    data = {'R': carts,
            'Z': atomic_numbers,
            'E': energy,
            'F': -1 * forces}

    return data


def parse_irc_data(dir_path):
    """Load IRC data for the specified reaction.

    Parameters
    ----------
    dir_path : str
        Path to directory containing IRC data of a reaction.

    Returns
    -------

    """
    # Hartree to kcal/mol conversion factor
    hartree_to_kcal_mol = 627.509

    # pre-computed atomic energies (Hartree)
    energy_h_atom = -0.5004966690
    energy_o_atom = -75.0637742413

    # load IRC cartesian coordinates (Angstrom) and atomic numbers
    file_rpath = os.path.join(dir_path, os.path.basename(glob(f"{dir_path}/*rpath.TZ.xyz")[0]))
    irc_mols = list(iread(file_rpath))
    irc_carts = np.array([i.positions for i in irc_mols])
    irc_nums = np.array([i.numbers for i in irc_mols])
    assert irc_carts.shape[0] == irc_nums.shape[0]
    assert irc_carts.shape[1] == irc_nums.shape[1]
    assert irc_carts.shape[2] == 3
    # get number of data points and number of atoms
    n_data, n_atoms = irc_nums.shape
    # get number of oxygen and hydrogen atoms
    n_o_atom = np.count_nonzero(irc_nums[0] == 8)
    n_h_atom = np.count_nonzero(irc_nums[0] == 1)

    # load IRC energy (Hartree) and compute relative energy in kcal/mol
    file_energy = os.path.join(dir_path, os.path.basename(glob(f"{dir_path}/*energy.TZ.csv")[0]))
    irc_energy = pd.read_csv(file_energy, header=None)
    irc_energy = irc_energy[0].values - n_h_atom * energy_h_atom - n_o_atom * energy_o_atom
    irc_energy = irc_energy.reshape(-1, 1) * hartree_to_kcal_mol
    assert irc_energy.ndim == 2
    assert irc_energy.shape == (n_data, 1)

    # load IRC gradient (Hartree / Angstrom) and convert to force in kcal/mol/A
    data_dir = os.path.join(dir_path, os.path.basename(glob(f"{dir_path}/*gradient.TZ.csv")[0]))
    irc_forces = -1 * pd.read_csv(data_dir, header=None).values
    assert irc_forces.shape[0] == n_data
    assert irc_forces.shape[1] == n_atoms * 3
    irc_forces = irc_forces.reshape(n_data, n_atoms, 3) * hartree_to_kcal_mol

    output = {
        'Z': irc_nums,
        'R': irc_carts,
        'E': irc_energy,
        'F': irc_forces,
        'N': np.repeat(n_atoms, n_data).reshape(-1, 1),
    }
    return output

def read_gradient(filename):
   gradients = []
   with open(filename,'r') as fp:
      #skip header line
      fp.readline()
      line = fp.readline()
      while line:
         grad = line.split()[1:]
         grad = [float(el) for el in grad]
         gradients.append(grad)
         line = fp.readline()
   return gradients


def parse_aimd_data(dir_path):
    """Load AIMD data for the specified reaction.

    Parameters
    ----------
    dir_path : str
        Path to directory containing AIMD data of a reaction.

    Returns
    -------
    """
    # Hartree to kcal/mol conversion factor
    hartree_to_kcal_mol = 627.509
    hartree_to_kcal_mol_ang = 1185.82062

    # pre-computed atomic energies (Hartree)
    energy_h_atom = -0.5004966690
    energy_o_atom = -75.0637742413

    #list of aimd mols
    aimd_mols = []
    raw_energies = []
    gradients = []
    # load AIMD cartesian coordinates (Angstrom) and atomic numbers
    for subdir1 in os.listdir(dir_path):
       subpath1 = os.path.join(dir_path,subdir1)
       if os.path.isdir(subpath1):
          for subdir2 in os.listdir(subpath1):
             subpath2 = os.path.join(subpath1,subdir2)
             if os.path.isdir(subpath2):
                for subdir3 in os.listdir(subpath2):
                   subpath3 = os.path.join(subpath2,subdir3)
                   if os.path.isdir(subpath3):
                      for aimd_dir in os.listdir(subpath3):
                         final_path = os.path.join(subpath3,aimd_dir)
                         #extract all information from AIMD directory
                         if os.path.isdir(final_path) and aimd_dir == 'AIMD':
                            #get structures from View.xyz
                            new_aimd_mols =list(iread(os.path.join(final_path,'View.xyz')))
                            #view.xyz contains one structure more than TandV and NucForces
                            aimd_mols += new_aimd_mols[:-1]

                            #get absolute energies from TandV
                            energies = pd.read_table(os.path.join(final_path,'TandV'),header=0,delim_whitespace=True)
                            raw_energies += list(energies['Time/fs'].values) #this is confusing but pandas recognizes the '#' in the header as an own symbol

                            #get gradient from NucForces
                            gradients += read_gradient(os.path.join(final_path,'NucForces'))

    aimd_carts = np.array([i.positions for i in aimd_mols])
    aimd_nums = np.array([i.numbers for i in aimd_mols])

    #same checks as in irc case
    assert aimd_carts.shape[0] == aimd_nums.shape[0]
    assert aimd_carts.shape[1] == aimd_nums.shape[1]
    assert aimd_carts.shape[2] == 3
    # get number of data points and number of atoms
    n_data, n_atoms = aimd_nums.shape
    # get number of oxygen and hydrogen atoms
    n_o_atom = np.count_nonzero(aimd_nums[0] == 8)
    n_h_atom = np.count_nonzero(aimd_nums[0] == 1)

    #convert energies to relative energies in kcal/mol
    aimd_energy = [el -  n_h_atom * energy_h_atom - n_o_atom * energy_o_atom for el in raw_energies] 
    aimd_energy = np.asarray(aimd_energy)
    aimd_energy = aimd_energy.reshape(-1, 1) * hartree_to_kcal_mol
    assert aimd_energy.ndim == 2
    assert aimd_energy.shape == (n_data, 1)

    #convert Gradients to Nuclear Forces
    forces = []
    factors = []
    for el in aimd_nums[0]:
       if el == 1:
          factors.append(-1.00784 * 1822.88818 * hartree_to_kcal_mol_ang)
          factors.append(-1.00784 * 1822.88818 * hartree_to_kcal_mol_ang)
          factors.append(-1.00784 * 1822.88818 * hartree_to_kcal_mol_ang)
       else:
          factors.append(-15.999 * 1822.88818 * hartree_to_kcal_mol_ang)
          factors.append(-15.999 * 1822.88818 * hartree_to_kcal_mol_ang)
          factors.append(-15.999 * 1822.88818 * hartree_to_kcal_mol_ang)

    assert len(factors)==3*n_atoms
    
    for grad in gradients:
       force = [grad[i]*factors[i] for i in range(len(factors))]
       forces.append(force)

    aimd_forces = np.asarray(forces)
    assert aimd_forces.shape[0] == n_data
    assert aimd_forces.shape[1] == n_atoms * 3
    aimd_forces = aimd_forces.reshape(n_data, n_atoms, 3)

    output = {
        'Z': aimd_nums,
        'R': aimd_carts,
        'E': aimd_energy,
        'F': aimd_forces,
        'N': np.repeat(n_atoms, n_data).reshape(-1, 1),
    }
    return output

