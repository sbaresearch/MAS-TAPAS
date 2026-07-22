![tests](https://github.com/sbaresearch/MAS-TAPAS/actions/workflows/ci.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/tapas-privacy/badge/?version=latest)](https://tapas-privacy.readthedocs.io/en/latest/index.html)

# MAS-TAPAS: Extension of a Toolbox for Adversarial Privacy Auditing of Synthetic Data

Evaluating the privacy of synthetic data with an adversarial toolbox. This code extends the TAPAS toolbox presented in [the associated paper](https://arxiv.org/abs/2211.06550) with additional attacks, threat models and metrics, including attacks provided in [Synth-MIA](https://github.com/joshward96/Synth-MIA) which are specific for the No-Box scenario. To ensure interoperability and align the evaluation frameworks, we implemented a custom integration wrapper and expanded the available threat models.



## Reference

This repository builds on top of **TAPAS**, a toolbox for adversarial privacy auditing of synthetic data.

Official [documentation](https://tapas-privacy.readthedocs.io/en/latest/index.html) of TAPAS.

> Houssiau, F., Jordon, J., Cohen, S.N., Daniel, O., Elliott, A., Geddes, J., Mole, C., Rangel-Smith, C. and Szpruch, L., 2022. *TAPAS: a toolbox for adversarial privacy auditing of synthetic data.*

In `BibTeX`:
```bibtex
@article{houssiau2022tapas,
  title={TAPAS: a toolbox for adversarial privacy auditing of synthetic data},
  author={Houssiau, F and Jordon, J and Cohen, SN and Daniel, O and Elliott, A and Geddes, J and Mole, C and Rangel-Smith, C and Szpruch, L},
  year={2022},
  publisher={Neural Information Processing Systems Foundation}
}
```

This project also incorporates components from the following software:

- Ward, J. et al., 2025. *Synth-MIA: A Privacy Leakage Auditing Tool for Synthetic Data.* Source code available at [github.com/joshward96/Synth-MIA](https://github.com/joshward96/Synth-MIA).



## Direct Installation

### Requirements
The framework and its building blocks have been developed and tested under Python 3.11.


#### Poetry installation
To mimic our environment exactly, we recommend using `poetry`. To install poetry (system-wide), follow the instructions [here](https://python-poetry.org/docs/).

Then run
```
poetry install
```
from inside the project directory. This will create a virtual environment (default `.venv`), that can be accessed by running `poetry shell`, or in the usual way (with `source .venv/bin/activate`).

#### Pip installation

It is also possible to install from pip:
```
pip install git+https://github.com/sbaresearch/MAS-TAPAS
```

