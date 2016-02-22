Globber
=======

To go from a catalog of PS1 sky positions and photometry, execute the scripts
in this package in the following order:

1. Deredden and clean the photometry. First, we'll use the SFD dust map
(accessed via the [sfd Python package](https://github.com/adrn/sfd)) to
deredden the PS1 photometry. We'll then remove any bad photometry:

        python scripts/deredden-ps1.py -f data/ngc5897/PS1_stars_pv3

This will output a Numpy save file containing the cleaned catalog information
with the dereddened photometry.

2. Convert the photometry into the colors and covariances that we will use
to filter the cluster photometry:

        python scripts/photometry-to-xcov.py --input=data/ngc5897/PS1_stars_pv3_dered.npy --output=data/ngc5897/XCov_med.h5 -v

3. Optionally, add an isochrone to the XCov file.

        python scripts/isochrone-to-xcov.py --iso-file=data/ngc5897/dartmouth_iso_ps1.dat --xcov-file=data/ngc5897/XCov_med.h5 -v

4. TODO:

        python scripts/compute-cmd-likelihoods.py -f path/to/data_file_XCov.h5 -n 1000 -i 0 -v

Or check the status of the run:

        python scripts/compute-cmd-likelihoods.py -f path/to/data_file_XCov.h5 --status

5. Plot filtered images of number counts of stars

        python make-filtered-images.py -f path/to/data_file_XCov.h5
