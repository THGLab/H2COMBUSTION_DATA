Machine Learning of Hydrogen Combustion Reaction.

## Installation and Dependencies
The developer installation is available and for that you need to first clone H2Combustion from this repository:

    git clone https://github.com/THGLab/H2Combustion.git

and then run the following command inside the repository:

    pip install -e .


We recommend using conda environment to install dependencies of this library.
Please install (or load) conda and then proceed with the following commands:

    conda create --name torch-gpu python=3.7
    conda activate torch-gpu
    conda install -c conda-forge numpy scipy pandas ase pyyaml
    conda install -c pytorch pytorch torchvision cudatoolkit=10.1 

Now, you can run combust modules anywhere on your computer as long as you are in the `torch-gpu` environment.
Please note that this is a developer version, and thus you should reinstall the library whenever you pull new changes. 
Otherwise, you always use the previously installed version of this library.


## Guidelines
- You can find several models inside the scripts directory that rely on the implemented modules in the combust library. 
Please modify parameters using the yaml files.

- Please push your changes to a new branch and avoid merging with the master branch unless
your work is reviewed by at least one other contributor.

- The documentation of the modules are available at most cases. Please look up local classes or functions
and consult with the docstrings in the code.


## Data Format
This is an example for loading and combining IRC data and storing them in a single npz file format.
To run this snippet, one needs to download the IRC directory from H2Combustion Box and fix reaction 14 to
remove the duplicate data at the end of energy and gradient files.

```python
#!/usr/bin/env python

import numpy as np

from combust.utils.rxn_data import rxn_dict
from combust.utils.parsers import parse_irc_data
from combust.utils.utility import combine_rxn_arrays, write_data_npz


total = []
for key, rxn in rxn_dict.items():
    # load IRC data for a given reaction
    data = parse_irc_data(f"IRC/{rxn['folder']}")
    # sanity checks
    n_data = data['Z'].shape[0]
    assert data['Z'].shape == (n_data, rxn['natoms'])
    assert data['R'].shape == (n_data, rxn['natoms'], 3)
    assert data['E'].shape == (n_data, 1)
    assert data['F'].shape == (n_data, rxn['natoms'], 3)
    assert data['N'].shape == (n_data, 1)
    # add reaction number as a string (e.g., '01', '02', ..., '06a', '06b', ... '19')
    data['RXN'] = np.repeat(key[3:], n_data).reshape(-1, 1)
    # add data dictionary to list of dictionaries
    total.append(data)

# combine list of data dictionaries into one dictionary (with max natoms being 6)
data = combine_rxn_arrays(total, n_max=6)

# write data dictionary (total number of data points = 1,711)
write_data_npz(data, 'rxns_irc.npz')

# load data dictionary back as a dictionary
data = np.load('rxns_irc.npz')
```

## Scripts
1. To combine IRC data stored in *rpath.TZ.xyz, *energy.TZ.csv, and *gradient.TZ.csv files
   into a single `npz` file (similar to the code snippet above), use the command below and
   provide the required (and optional) arguments:

   ```bash
   ./scripts/script_combust.py irc_npz -h
   ```

   For example:

   ```bash
   ./scripts/script_combust.py irc_npz IRC/ -n 6 -o rxns_data_irc.npz
   ```

2. To plot coordination number for a given reaction using `npz` file(s), use the command below
   and provide the required (and optional) arguments:

   ```bash
   ./scripts/script_combust.py plot_cn -h
   ```

   At least IRC `npz` file is required for plotting, and AIMD and Normal Mode Displacement `npz`
   files can be provided as optional arguments.
   For example using the `npz` file generated above, plot coordination number for reaction 11 by:

   ```bash
   ./scripts/script_combust.py plot_cn rxns_data_irc.npz -r 11
   ```

   **Note:** The `rxn_dict` in `utils/rxn_data.py` containing all reactions information is not
   complete yet, so running this for some of the reactions might fail, at this points.
