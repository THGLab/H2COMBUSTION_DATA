import os
import numpy as np
import matplotlib.pyplot as plt


def plot_irc_energy(irc, yp, plot_dir, n_rot):
    """
    plot irc energy and all rotations of predicted energy.


    Parameters
    ----------
    irc: dict
        dictionary of irc data

    yp: ndarray
        array of predicted energies with shape (n_rot, n_data, 1)

    plot_dir: str
        path to the directory to save output plots

    n_rot: int
        number of data augmentation by rotation


    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    y = irc['E'].reshape(-1)

    n_irc = y.shape[0]
    x = np.arange(1, n_irc + 1)

    atoms = irc['Z'][0]
    n_atoms = irc['N'][0, 0]
    print('n_irc:', n_irc, ', n_atoms: ', n_atoms)

    plt.figure()
    colors = ['r', 'g', 'gray', 'y', 'cyan', 'orange']
    for r in range(n_rot):
        plt.scatter(x, yp[r].reshape(-1), c=colors[r], marker='.', label='ML, r#%i' % (r + 1))
        plt.plot(x, yp[r].reshape(-1), c=colors[r])

        plt.scatter(x, y, s=50, c='k', marker='.')  # , label='DFT')
        plt.plot(x, y, c='k')

    # plt.scatter([0,], irc_energy[ts], s=500, c='r', marker='.')
    # plt.scatter([0-ts_lo,], irc_energy[ts-ts_lo], s=500, c='y', marker='.')
    # plt.scatter([0+ts_up,], irc_energy[ts+ts_up], s=500, c='g', marker='.')
    plt.title('IRC / Relative Energy')
    plt.ylabel('Energy [kcal/mol]')
    plt.xlabel('Step #')
    # plt.ylim(-230,-209)
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(plot_dir, "irc_energy_pred.png"), dpi=300)

    plt.close()


def plot_irc_force(irc, yp, plot_dir, n_rot):
    """
    plot irc force magnitude per atom and rotations of predictions.


    Parameters
    ----------
    irc: dict
        dictionary of irc data

    yp: ndarray
        array of predicted forces with shape (n_rot, n_data, n_atom, 3)

    plot_dir: str
        path to the directory to save output plots

    n_rot: int
        number of data augmentation by rotation


    """

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    y = irc['F']
    y = np.linalg.norm(y, 2, axis=-1)  # shape: (n_data, n_atom)
    yp = np.linalg.norm(yp, 2, axis=-1)  # shape: (n_rot, n_data, n_atom)

    n_irc = y.shape[0]
    x = np.arange(1, n_irc + 1)

    atoms = irc['Z'][0]
    n_atoms = irc['N'][0, 0]
    print('n_irc:', n_irc, ', n_atoms: ', n_atoms)
    colors = ['r']

    for r in range(n_rot // 2):

        for a in range(n_atoms):
            plt.figure()
            plt.scatter(x, y[:, a].reshape(-1), s=50, c='k', marker='.', label='IRC')
            plt.plot(x, y[:, a].reshape(-1), c='k')

            plt.scatter(x, yp[r, :, a].reshape(-1), c=colors[0], marker='.', label='Z=%i (%i)' % (atoms[a], a + 1))
            plt.plot(x, yp[r, :, a].reshape(-1), c=colors[0])

            # plt.scatter([0,], irc_energy[ts], s=500, c='r', marker='.')
            # plt.scatter([0-ts_lo,], irc_energy[ts-ts_lo], s=500, c='y', marker='.')
            # plt.scatter([0+ts_up,], irc_energy[ts+ts_up], s=500, c='g', marker='.')
            plt.title('IRC / Force')
            plt.ylabel('Force [kcal/mol/A]')
            plt.xlabel('Step #')
            # plt.ylim(-230,-209)
            plt.grid(True)
            plt.legend()

            plt.savefig(os.path.join(plot_dir, "irc_force_rot%i_atom#%s.png" % (r + 1, a + 1)), dpi=300)
            plt.close()


def plot_irc_ei(irc, yi, plot_dir, n_rot, mu=0, sigma=1):
    """
    plot irc energy and all rotations of predicted atomic energies.
    This function is good for a single reaction unless n_atoms and n_rots reamin same between different reactions.


    Parameters
    ----------
    irc: dict
        dictionary of irc data

    yi: ndarray
        array of predicted energies with shape (n_rot, n_data, n_atoms, 1)

    plot_dir: str
        path to the directory to save output plots

    n_rot: int
        number of data augmentation by rotation, required for cross-checking

    mu: float
        from normalizer, if atomic energies are alread normalized it should be 0

    sigma: float
                from normalizer, if atomic energies are alread normalized it should be 1


    """

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    yi = yi * sigma
    y = irc['E'].reshape(-1) - mu

    colors = ['gray', 'r', 'skyblue', 'purple', 'orange', 'y']

    n_irc = y.shape[0]
    atoms = irc['Z'][0]
    x = np.arange(1, n_irc + 1)
    n_atoms = irc['N'][0, 0]
    print('n_irc:', n_irc, ', n_atoms: ', n_atoms)

    for r_idx in range(n_rot//2):  # n_rot reduced by half

        plt.figure()

        for a_idx in range(n_atoms):
            if a_idx == 0:
                plt.scatter(x, y, s=50, c='k', marker='.', label='IRC')
                plt.plot(x, y, c='k')

                yi_sum = np.sum(yi[r_idx, :, :n_atoms], axis=1).reshape(-1)
                plt.scatter(x, yi_sum, c='b', marker='.', label='sum')
                plt.plot(x, yi_sum, c='b')

            if atoms[a_idx] != 0:
                plt.scatter(x, yi[r_idx, :, a_idx].reshape(-1), c=colors[a_idx], marker='.',
                            label='Z=%i (%i)' % (atoms[a_idx], a_idx + 1))
                plt.plot(x, yi[r_idx, :, a_idx].reshape(-1), c=colors[a_idx])

        # plt.scatter([0,], irc_energy[ts], s=500, c='r', marker='.')
        # plt.scatter([0-ts_lo,], irc_energy[ts-ts_lo], s=500, c='y', marker='.')
        # plt.scatter([0+ts_up,], irc_energy[ts+ts_up], s=500, c='g', marker='.')
        plt.title('IRC / Relative Energy')
        if mu == 0:
            plt.ylabel('Energy [kcal/mol]')
        else:
            plt.ylabel('Scaled Energy [kcal/mol]')

        plt.xlabel('Step #')
        # plt.ylim(-140,0)
        # plt.ylim(-235,-210)
        # plt.ylim(-115,-60)
        # plt.ylim(-50,10)
        plt.grid(True)
        plt.legend()  # loc='lower left')#bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.savefig(os.path.join(plot_dir, "irc_energy_atomwise_rot%i.png" % (r_idx + 1)), dpi=300)
        plt.close()
