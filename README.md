[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


![](docs/website_illustration.png)

# pyHiM

*pyHiM* implements the analysis of multiplexed DNA-FISH data, as described in our [Hi-M paper](https://www.nature.com/articles/s41596-019-0269-9).

## Documentation

Find the full documentation in [ReadtheDocs](https://pyhim.readthedocs.io/en/latest/) and [tutorials](https://pyhim.readthedocs.io/en/latest/getting_started/tutorials.html) illustrating the main steps of pyHiM analysis pipeline.

To install, please follow the instructions [here](https://pyhim.readthedocs.io/en/latest/getting_started/quick_install.html).

After you installed *pyHiM*, you may want to consult a [guide](https://pyhim.readthedocs.io/en/latest/user_guide/pyhim_presentation.html) on how to use it.

If you are a developer, follow instructions [here](https://pyhim.readthedocs.io/en/latest/contributor/dev_env.html).

## Publications

For more information on Hi-M, please see the following resources:
- [Hi-M protocol](https://github.com/NollmannLab/HiM_protocol)
- [Hi-M method](https://www.cell.com/molecular-cell/fulltext/S1097-2765(19)30011-5)
- [Hi-M application using *pyHiM*](https://www.nature.com/articles/s41588-021-00816-z)

## Dependencies

*pyHiM* was entirely written in python and makes extensive use of the following packages:

- [Astropy](https://www.astropy.org/)
- [scikit-image](https://scikit-image.org/)
- [starDist](https://github.com/stardist/stardist)

*pyHiM* also uses functions from [Big-FISH](https://github.com/fish-quant/big-fish) to perform Gaussian 3D fits.


## Support

If you have any question relative to the repository, please open an issue. You can also contact Marcelo Nollmann.

## License

*pyHiM* is licensed under GPLv3 (see LICENSE.txt).
Packages used by *pyHiM* are licensed under the revised 3-clause BSD style license.

Check COPYRIGHT.txt for a list of authors and the git history for their individual contributions.

*pyHiM* is a software package developed by the [Nollmann Lab](http://www.nollmannlab.org) at the [Center of Structural Biology](http://www.cbs.cnrs.fr), a department of the [CNRS](http://www.cnrs.fr) and the [INSERM](http://www.inserm.fr).
