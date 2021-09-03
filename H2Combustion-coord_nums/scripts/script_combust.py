#!/usr/bin/env python


import os
import sys
import argparse
import numpy as np

from combust.utils.rxn_data import rxn_dict
from combust.utils.parsers import parse_irc_data
from combust.utils.utility import check_data_consistency, combine_rxn_arrays, write_data_npz
from combust.utils.utility import get_data_subset, get_data_remove_appended_zero
from combust.utils.plots import get_cn_arrays, get_rxn_namedtuple, visualize_cn, visualize_irc_energy


def main_plotting_namedtuple(path_npz, rxn_num):
    """Return NamedTuple (used for plotting) corresponding to specified rxn number.

    Parameters
    ----------
    path_npz : str
        Path to *.noz file containing reaction data.
    rxn_num : str
        Reaction number to plot given as a string of length 2.

    """

    # load data
    data = np.load(path_npz)

    # check consistency of data arrays shape
    check_data_consistency(data)
    print(f"{path_npz} number of data points: {data['Z'].shape[0]}")

    # get unique reaction numbers included in data
    rxn_nums = np.unique(data['RXN'])
    print(f"{path_npz} reactions: {rxn_nums}")

    # check or identify which reaction to plot
    if rxn_num is None and len(rxn_nums) == 1:
        rxn_num = f"rxn{rxn_nums[0]}"
    elif rxn_num is not None:
        if rxn_num in rxn_nums:
            rxn_num = f"rxn{parsed.rxn}"
        else:
            raise ValueError(f"{path_npz} does not contain {rxn_num}! Existing rxns: {rxn_nums}")
    else:
        raise ValueError(f"{path_npz} cannot tell which rxn to plot! Existing rxns: {rxn_nums}")

    # get data corresponding to reaction
    data = get_data_subset(data, rxn_num[3:])

    # remove appended zero to data arrays
    data = get_data_remove_appended_zero(data)

    # get reaction data
    cn1s = rxn_dict[rxn_num]['cn1']
    cn2s = rxn_dict[rxn_num]['cn2']
    mu = rxn_dict[rxn_num]['mu']

    # make namedtuple for plotting
    cn1 = get_cn_arrays(data['Z'], data['R'], cn1s, mu=mu[0], sigma=3.0)
    cn2 = get_cn_arrays(data['Z'], data['R'], cn2s, mu=mu[1], sigma=3.0)
    data_namedtuple = get_rxn_namedtuple(cn1, cn2, data['E'].flatten())

    return data_namedtuple, rxn_num


def parse_irc_npz(arguments):
    parser = argparse.ArgumentParser(
        description="Combine IRC data stored in *rpath.TZ.xyz, *energy.TZ.csv, "
                    "& *gradient.TZ.csv into one single npz file."
        )
    parser.add_argument("path_irc", help="Path to IRC directory in which reactions are stored.")
    parser.add_argument("-r", "--rxn", default=None, type=str,
                        help="Reaction number to store in npz file. If None, all reactions are "
                             "parsed and stored. [default=None]")
    parser.add_argument("-n", "--natoms", default=6, type=int,
                        help="Maximum number of atoms [default=6].")
    parser.add_argument("-o", "--output", default="rxns_data_irc.npz", type=str,
                        help="Filename of npz output. [default='rxns_data_irc.npz']")

    return parser.parse_args(arguments)


def parse_plot_cn(arguments):
    parser = argparse.ArgumentParser(
        description="Plot coordination number of reaction for IRC data & other given data files."
        )
    parser.add_argument("path_npz_irc",
                        help="Path to *.npz containing IRC data of reaction(s).")
    parser.add_argument("-a", "--path_npz_aimd", default=None, type=str,
                        help="Path to *.npz containing AIMD data of reaction(s).")
    parser.add_argument("-d", "--path_npz_disp", default=None, type=str,
                        help="Path to *.npz containing normal mode displaced data of reaction(s).")
    parser.add_argument("-r", "--rxn", default=None, type=str,
                        help="Reaction number to plot. If None, and the given npz files "
                             "contain only one reaction, that reaction is plotted, "
                             "otherwise an error is raised. If the reaction number has one digit "
                             "prepend it with 0 (e.g. instead of 1 use 01) [default=None]")
    parser.add_argument("-o", "--output", default=None, type=str,
                        help="Output directory in which the plots are stored. If None, "
                             "the plot_cn_{rxn_number} is created. [default=None].")

    return parser.parse_args(arguments)


if __name__ == "__main__":
    args = sys.argv[1:]
    task = args.pop(0)

    if task == "irc_npz":
        # parse command-line arguments
        parsed = parse_irc_npz(args)

        # check given reaction number
        if parsed.rxn is not None and f"rxn{parsed.rxn}" not in rxn_dict.keys():
            numbers = [item[3:] for item in rxn_dict.keys()]
            raise ValueError(f"Reaction number {parsed.rxn} not among {numbers} rxn numbers!")

        # make a list of reaction dictionaries
        total = []
        for key, rxn in rxn_dict.items():
            # check given reaction number
            if parsed.rxn is not None and f"rxn{parsed.rxn}" != key:
                continue
            # load data
            folder = os.path.join(parsed.path_irc, rxn['folder'])
            if not os.path.isdir(folder):
                raise ValueError(f"Couldn't find {folder}!")
            print(f"Folder: {folder}")
            data = parse_irc_data(os.path.join(parsed.path_irc, rxn['folder']))
            # add reaction number as a string (e.g., '01', '02', ..., '06a', '06b', ... '19')
            n_data = data['Z'].shape[0]
            data['RXN'] = np.repeat(key[3:], n_data).reshape(-1, 1)
            # check consistency of data arrays shape
            check_data_consistency(data)
            # add list of dictionaries
            total.append(data)

        # combine data dictionaries into one dictionary
        data = combine_rxn_arrays(total, n_max=parsed.natoms)

        # write data dictionary
        write_data_npz(data, parsed.output)

    elif task == 'plot_cn':
        # parse command-line arguments
        parsed = parse_plot_cn(args)
        if parsed.rxn is not None and len(parsed.rxn) not in [2, 3]:
            raise ValueError(f"Expect rxn number to have length 2 or 3! Given rxn_num={parsed.rxn}")

        # prepare IRC data
        irc, rxn_num = main_plotting_namedtuple(parsed.path_npz_irc, parsed.rxn)

        # prepare AIMD & Normal Mode Displacement data, if given
        # it is important to check loaded reaction number, because if parsed.rxn is None and each
        # npz file contains only one reaction, it is important to know that it is the same rxn
        aimd, disp = None, None
        if parsed.path_npz_aimd is not None:
            aimd, aimd_rxn_num = main_plotting_namedtuple(parsed.path_npz_aimd, parsed.rxn)
            assert aimd_rxn_num == rxn_num
        if parsed.path_npz_disp is not None:
            disp, disp_rxn_num = main_plotting_namedtuple(parsed.path_npz_disp, parsed.rxn)
            assert disp_rxn_num == rxn_num

        # make directory for storing plots
        directory = f'plot_cn_{rxn_num}'
        if not os.path.isdir(directory):
            os.mkdir(directory)

        # plot data
        title = rxn_dict[rxn_num]['title']
        xtitle = rxn_dict[rxn_num]['xtitle']
        ytitle = rxn_dict[rxn_num]['ytitle']
        visualize_cn(irc, aimd, disp, directory, title, xtitle, ytitle)

    elif task == 'plot_irc_energy':
        # parse command-line arguments
        parsed = parse_plot_cn(args)
        if parsed.rxn is not None and len(parsed.rxn) not in [2, 3]:
            raise ValueError(f"Expect rxn number to have length 2 or 3! Given rxn_num={parsed.rxn}")

        # prepare IRC data
        irc, rxn_num = main_plotting_namedtuple(parsed.path_npz_irc, parsed.rxn)

        # plot data
        title = rxn_dict[rxn_num]['title']
        xtitle = rxn_dict[rxn_num]['xtitle']
        ytitle = rxn_dict[rxn_num]['ytitle']
        visualize_irc_energy(irc,title,xtitle,ytitle)


    else:
        raise ValueError(f"Task={task} is not recognized! Options: irc_npz, plot_cn")
