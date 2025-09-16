# GHost in the SHELL
This project is the open-source component of our paper "GHost in the SHELL: 
A GPU-to-Host Memory Attack and Its Mitigation", submitted in IEEE Security
and Privacy (S&P) 2026. This repository contains our attack poc, mitigation
implementations, and evaluation platforms.

# Environment
It is recommended to be worked in following environemnts.

**Hardware**: Intel CPU + NVIDIA GPU RTX 2000+ \
**Software**: Ubuntu 22.04

* This attack can be applied to CUDA version 12.2+ and NVIDIA GPU driver r535+,
  but, all our patches and build system are focused on CUDA 12.8 with driver
  r570.


# Native installation
To try GHost-Attack and its mitigation, you need to setup its dependencies and
requirements by following command (it needs sudo privileges).

    cd GHost-Attack
    ./init.sh

It takes about 3 to 4 hours to install whole needed dependencies and requirements.

# Directory Overview
Our project is structured as follows.

    GHost-Attack/
        ├── artifacts/
        │    ├── 3.GHost_Attack
        │    ├── 4.SHELL_Mitigation
        │    └── 6.Evaluation
        ├── third_party/
        ├── init.sh
        ├── apply-no-mitigation.sh
        └── apply-mitigation.sh

`artifacts/`: All testable POCs and evaluation codes are located in here.
- `3.GHost_Attack`: Includes attack POCs introduced in paper section 3.
- `4.SHELL_Mitigation`: Includes patch files for the mitigation (SHELL).
- `6.Evaluation`: Includes end-to-end performance test on HMM_sample_code, and micro
evaluation results.

`third_party/`: third party platforms required for testing are located in here. (e.g.,
llvm, cuda, pytorch ...)
