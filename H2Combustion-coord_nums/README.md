
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
