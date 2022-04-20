Installation Guide
==================

This document describes the steps needed to install Mera in your system.

System Requirements
-------------------

For a *x86* architecture you will need to have `Ubuntu 18.04` as your OS whereas for *aarch64* you will need `Ubuntu 20.04`.
The following software packages will also need to be installed:

* LLVM-10
* python3.6
* pip >= 21.3.1

Mera Installation
-----------------

The Mera environment provides 3 different modes depending on the target usage:

* host-only: Meant for performing deployments only targetting simulation running on the host.
* runtime: Meant for running inference in HW accelerators using the DNA IP, requires extra system dependencies depending on the HW device.
* full: Meant for user who want the functionality of both `host-only` and `runtime` modesl

After choosing the desired mode you can install Mera with the following command:

.. code-block:: bash
   :linenos:

   pip install mera[<MERA_MODE>]

So for example, installing Mera in host-only mode:

.. code-block:: bash
   :linenos:

   pip install mera[host-only]

`mera` provides packages for installing in both *x86* and *aarch64* architectures.
The pip command will also install all the necesary dependencies in order to perform deployments for Mera. Note that some of the tutorials require some extra 
dependencies to be installed. Please check the tutorial's `README.md` file to check which other packages might be needed.
