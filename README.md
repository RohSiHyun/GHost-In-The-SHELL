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

# Testing Attack

## 1. Breaking ASLR on Host code from GPU kernel

Compile the Code

    $ ./apply-no-mitigation.sh
    $ export PATH="$HOME/.local/bin:$PATH"
    $ export PATH="/usr/local/cuda-12.8/bin:$PATH"
    $ cd artifacts/3.GHost_Attack/3.1.Breaking_ASLR_with_libcuda
    $ make
    
Run the Code

    $ ./hmm-aslr-break

## 2. Overwrite Return address of libcuda-rt function from GPU kernel

Compile the code:

    $ ./apply-no-mitigation.sh
    $ export PATH="$HOME/.local/bin:$PATH"
    $ export PATH="/usr/local/cuda-12.8/bin:$PATH"
    $ cd artifacts/3.GHost_Attack/3.2.Hijacking_Host_Control_Flow/RetAddr_Overwrite
    $ make

Run the code:

    $ ./retAddr-overwrite

## 3. Overwrite GOT of libcuda-rt from GPU kernel

Compile the code:
   
    $ ./apply-no-mitigation.sh
    $ export PATH="$HOME/.local/bin:$PATH"
    $ export PATH="/usr/local/cuda-12.8/bin:$PATH"
    $ cd artifacts/3.GHost_Attack/3.2.Hijacking_Host_Control_Flow/GOT_Overwrite
    $ make

Run the code:

    $ ./got-overwrite

## 4. Chain of GPU kernel vulnerability to GOT overwrite

Virtual environment: 

    $ source third_party/pytorch/.pytorch/bin/activate

make attacker's payload:

    $ cd artifacts/6.Evaluation/6.1.Attack/Vulnerable_GPU_Kernel/
    $ ./prepare.sh

Run the code:

    $ python3 ex.py

## 5. Attacker supplied GPU kernel leaks ASLR and overwrite GOT on Host

prepare attacker's GPU kernel to be loaded into chrome:

    $ cd artifacts/6.Evaluation/6.1.Attack/Attacker_Controlled_GPU_Kernel
    $ ./prepare.sh

Run chrome:

    # Need to run in a machine with GPU and display.

    $ cd ../../../../third_party/chromium/src
    $ ./out/Release/chrome --enable-features=Vulkan --enable-unsafe-webgpu # It cannot be done in SSH session, need display.
    $ With chrome opened, visit "https://webgpu.github.io/webgpu-samples/?sample=helloTriangle"

    # Check if the chrome suddenly restarted.


# Applying Mitigation

Virtual environment:

    $ source third_party/pytorch/.pytorch_patch/bin/activate

Apply mitigation: 

    $ ./apply-mitigation.sh

* Identical / instrumented codes of `artifacts/6.Evaluation/6.1.Attack/` are in `artifacts/6.Evaluation/6.2.Defense/`. You can test them with the same way done on #4, #5.
