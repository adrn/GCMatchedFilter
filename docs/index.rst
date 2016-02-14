Globber
=======

To go from a catalog of PS1 sky positions and photometry, execute the scripts
in this package in the following order:

1. Deredden and clean the photometry. First, we'll use the SFD dust map
(accessed via the [sfd Python package](https://github.com/adrn/sfd)) to
deredden the PS1 photometry. We'll then remove any bad photometry::

        python scripts/deredden-ps1.py -f path/to/data_file

This will output a Numpy save file containing the cleaned catalog information
with the dereddened photometry.

2. Convert the photometry into the colors and covariances taht we will use
to filter all of the photometry.

        python scripts/photometry-to-xcov.py --input=path/to/data_file_dered.npy

3. TODO:

        python scripts/compute-cmd-likelihoods.py -f path/to/data_file_XCov.h5 -n 1000 -i 0 -v

Or compute the status:

        python scripts/compute-cmd-likelihoods.py -f data/ngc5897/PS1_stars_pv3_dered_sm_XCov.h5 --status
