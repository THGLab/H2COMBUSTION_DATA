"""
A compilation of modules that help to find closest neighbours of each atom in a molecule.
Each Molecule is represented as a dictionary with following keys:
    - atoms: atomic positions with shape (n_atom, 3)
    - z: atomic numbers with shape (n_atoms, 1)
    - cell: unit cell with shape (3,3)
    - atom_prop: atomic property with shape (n_atoms, n_atomic_prop)
    - mol_prop: molecular property with shape (1, n_mol_prop)

"""
import numpy as np
from ase.neighborlist import neighbor_list

from combust.utils import padaxis

class ExtensiveEnvironment(object):
    """
    Provide atomic environment of an array of atoms and their atomic numbers.
    No cutoff, No periodic boundary condition

    Parameters
    ----------
    max_n_neighbors: int, optional (default: None)
        maximum number of neighbors to pad arrays if they have less elements
        if None, it will be ignored (e.g., in case all atoms have same length)

    """
    def __init__(self, max_n_neighbors=None):
        if max_n_neighbors is None:
            max_n_neighbors = 0
        self.max_n_neighbors = max_n_neighbors

    def _check_shapes(self, Rshape, Zshape):
        if Rshape[0] != Zshape[0]:
            msg = "@ExtensiveEnvironment: atoms and atomic_numbers must have same dimension 0 (n_data)."
            raise ValueError(msg)

        if Rshape[2] != 3:
            msg = "@ExtensiveEnvironment: atoms must have 3 coordinates at dimension 2."
            raise ValueError(msg)

        if Rshape[1] != Zshape[1]:
            msg = "@ExtensiveEnvironment: atoms and atomic_numbers must have same dimension 1 (n_atoms)."
            raise ValueError(msg)

    def get_environment(self, positions, atomic_numbers):
        """
        This function finds atomic environments extensively for each atom in the MD snapshot.

        Parameters
        ----------
        positions: ndarray
            A 3D array of atomic positions in XYZ coordinates with shape (D, A, 3), where
            D is number of snapshots of data and A is number of atoms per data point

        atomic_numbers: ndarray
            A 2D array of atomic numbers with shape (D, A)

        Returns
        -------
        ndarray: 3D array of neighbors with shape (D, A, A-1)
        ndarray: 3D array of neighbor mask with shape (D, A, A-1)
        ndarray: 2D array of atomic mask for atomic energies (D, A)

        """
        n_data = positions.shape[0]  # D
        n_atoms = positions.shape[1]  # A

        self._check_shapes(positions.shape, atomic_numbers.shape)

        # 2d array of all indices for all atoms in a single data point
        N = np.tile(np.arange(n_atoms), (n_atoms, 1))  # (A, A)

        # remove the diagonal self indices
        neighbors = N[~np.eye(n_atoms, dtype=bool)].reshape(n_atoms,
                                                        -1)  # (A, A-1)
        neighbors = np.repeat(neighbors[np.newaxis, ...], n_data, axis=0)  # (D, A, A-1)

        # mask based on zero atomic_numbers
        mask = np.ones_like(atomic_numbers)                 #(D, A)
        mask[np.where(atomic_numbers == 0)] = 0
        max_atoms = np.sum(mask, axis=1)

        neighbor_mask = (neighbors < np.tile(max_atoms.reshape(-1,1), n_atoms-1)[:,None,:]).astype('int')
        neighbor_mask *= mask[:,:,None]  # (D,A,A-1)
        neighbors *= neighbor_mask  # (D,A,A-1)

        # atomic numbers
        # atomic numbers are already in correct shape

        if n_atoms < self.max_n_neighbors:
            neighbors = padaxis(neighbors,
                                self.max_n_neighbors,
                                axis=-1,
                                pad_value=-1)  # (D, A, N)
            atomic_numbers = padaxis(atomic_numbers,
                                     self.max_n_neighbors,
                                     axis=-1,
                                     pad_value=0)  # (D, A, N)

        return neighbors, neighbor_mask, mask
