import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import mera
from mera import Target, Platform

import argparse
import torch
import tensorflow as tf
import numpy as np


def get_total_latency_ms(run_result, latency_key_name = 'elapsed_latency'):
    metrics = run_result.get_runtime_metrics()
    total_us = sum([x[latency_key_name] for x in metrics if x])
    return total_us / 1000


def mera_run_sim_tflite(tflite_filename, platform, target, build_config, host_arch, output_dir):
    with mera.TVMDeployer(output_dir, overwrite=True) as deployer:
        model = mera.ModelLoader(deployer).from_tflite(tflite_filename)
        input_data = {}
        for name, idesc in model.input_desc.all_inputs.items():
            input_data[name] = np.random.rand(*idesc.input_shape)
        deploy_ip = deployer.deploy(model, mera_platform=platform, target=target, build_config=build_config, host_arch=host_arch)
        ip_runner = deploy_ip.get_runner().set_input(input_data).run()
        return ip_runner


def mera_run_sim_onnx(onnx_filename, platform, target, build_config, host_arch, output_dir):
    with mera.TVMDeployer(output_dir, overwrite=True) as deployer:
        model = mera.ModelLoader(deployer).from_onnx(onnx_filename)
        input_data = {}
        for name, idesc in model.input_desc.all_inputs.items():
            input_data[name] = np.random.rand(*idesc.input_shape)
        deploy_ip = deployer.deploy(model, mera_platform=platform, target=target, build_config=build_config, host_arch=host_arch)
        ip_runner = deploy_ip.get_runner().set_input(input_data).run()
        return ip_runner


def mera_run_sim_pytorch(pt_filename, input_data, platform, target, build_config, host_arch, output_dir):
    with mera.TVMDeployer(output_dir, overwrite=True) as deployer:
        input_desc = {'input0' : (input_data.shape, 'float32')}
        model = mera.ModelLoader(deployer).from_pytorch(pt_filename, input_desc)

        deploy_ip = deployer.deploy(model, mera_platform=platform, target=target, build_config=build_config, host_arch=host_arch)
        ip_runner = deploy_ip.get_runner().set_input(input_data).run()
        return ip_runner


def mera_run_sim_mera(mera_filename, platform, target, build_config, host_arch, output_dir):
    with mera.TVMDeployer(output_dir, overwrite=True) as deployer:
        model = mera.ModelLoader(deployer).from_quantized_mera(mera_filename)
        input_data = {}
        for name, idesc in model.input_desc.all_inputs.items():
            input_data[name] = np.random.rand(*idesc.input_shape)
        deploy_ip = deployer.deploy(model, mera_platform=platform, target=target, build_config=build_config, host_arch=host_arch)
        ip_runner = deploy_ip.get_runner().set_input(input_data).run()
        return ip_runner


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=dir_path, required=True, help='Models directory')
    opt = parser.parse_args()

    import sys
    sys.stderr = open('error.log', 'w')

    target = Target.Simulator
    platform = Platform.SAKURA_1
    build_config = {
      "compiler_workers": 4,
      "scheduler_config": {
        "mode": "Fast",
        "pre_scheduling_iterations": 8000,
        "main_scheduling_iterations": 32000,
      }
    }
    host_arch = "x86"

    latencies = {}
    with os.scandir(opt.models) as it:
        for entry in it:
            if entry.is_file():
                basename, extension = os.path.splitext(entry.name)
                if extension == ".tflite":
                    runner = mera_run_sim_tflite(entry.path, platform, target, build_config, host_arch, basename)
                    latencies[basename] = get_total_latency_ms(runner, 'sim_time_us')
                elif extension == ".onnx":
                    target = Target.SimulatorBf16
                    platform = Platform.SAKURA_II
                    runner = mera_run_sim_onnx(entry.path, platform, target, build_config, host_arch, basename)
                    latencies[basename] = get_total_latency_ms(runner, 'sim_time_us')
                elif extension == ".mera":
                    runner = mera_run_sim_mera(entry.path, platform, target, build_config, host_arch, basename)
                    latencies[basename] = get_total_latency_ms(runner, 'sim_time_us')
                elif extension == ".pt":
                    components = basename.split("_")
                    if len(components) == 3 and "x" in components[1]:
                        batch, depth = 1, 3
                        w, h = components[1].split("x")
                        input_data = np.random.rand(batch, int(h), int(w), depth)
                        runner = mera_run_sim_pytorch(entry.path, input_data, platform, target, build_config, host_arch, basename)
                        latencies[basename] = get_total_latency_ms(runner, 'sim_time_us')
    


    results_file = "latencies.txt"
    with open(results_file, 'w') as latfile:
        for model, latency in latencies.items():
            latfile.write(model + ": " + str(latency) + " milliseconds\n")

    print("Models compiled and executed. Results have been saved to the file: ", results_file)

