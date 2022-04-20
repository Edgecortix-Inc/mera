Quickstart Guide
================

After following instructions in the :doc:`Installation Guide <installation_guide>`, you can see an example of Mera
running one of the available tutorials. The tutorials contain documented examples of the Mera API and serve as templates
for running your own deployments.

Tutorial List
-------------

- PyTorch resnet50 on Simulator (`pytorch/resnet50_simulator.py`):

Contains an example on how to deploy and run a traced `resnet50` model in x86 host simulation.
Can be executed with the following command:

.. code-block:: bash
    :linenos:

    cd tutorials/pytorch
    python3 resnet50_simulator.py


- PyTorch resnet50 on IP (`pytorch/resnet50_ip.py`):

Contains an example on how to deploy and run a traced `resnet50` model in FPGA environment.
Needs to have FPGA runtime setup before running.
Can be executed with the following command:

.. code-block:: bash
    :linenos:

    cd tutorials/pytorch
    # Needs to enable RUN_IP env in order to actually run the tutorial in HW
    RUN_IP=1 python3 resnet50_ip.py


- TFLite EfficientNet on Simulator (`tflite/efficientnet_simulator.py`):

Contains an example on how to deploy and run a quantized `efficientnet-lite1` and `efficientnet-lite4` model in x86 host simulation and run an example object classification.
Can be executed with the following command:

.. code-block:: bash
    :linenos:

    cd tutorials/tflite
    python3 efficientnet_simulator.py


- TFLite EfficientNet on IP (`tflite/efficientnet_ip.py`):

Contains an example on how to deploy and run a quantized `efficientnet-lite1` and `efficientnet-lite4` model in
FPGA environment and run an example object classification. Needs to have FPGA runtime setup before running.
Can be executed with the following command:

.. code-block:: bash
    :linenos:

    cd tutorials/tflite
    # Needs to enable RUN_IP env in order to actually run the tutorial in HW
    RUN_IP=1 python3 efficientnet_ip.py

