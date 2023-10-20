import argparse

import mera
from mera import Target, Platform


def compile_mera(onnx_filename, out_dir, platform, target):
    host_arch = "x86"
    with mera.TVMDeployer(out_dir, overwrite=True) as deployer:
        model = mera.ModelLoader(deployer).from_onnx(onnx_filename)
        deployer.deploy(model, mera_platform=platform, target=target, host_arch=host_arch)
        return out_dir


def main(arg):
    mera_platform = Platform.SAKURA_2C
    mera_target = Target.SimulatorBf16
    mera_path = compile_mera(arg.model_path, arg.out_dir, mera_platform, mera_target)
    print(f"SUCCESS, saved at {mera_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="deploy_detr",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="./source_model_files/DETR_600x400_ONNX.onnx",
        type=str,
    )
    main(parser.parse_args())
