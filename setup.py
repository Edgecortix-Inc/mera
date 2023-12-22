# Copyright 2022 EdgeCortix Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
import pathlib
import re
import sys
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        # Here we force the package to not be pure python, so we can get different pip packages depending on OS/python version.
        self.root_is_pure = False

cmdclass = {
    "bdist_wheel": bdist_wheel,
}

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

def get_version():
    ver_file = pathlib.Path(__file__).parent.resolve() / 'src/mera/version.py'
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, ver_file.read_text(), re.M)
    if mo:
        return mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (ver_file,))

_BASE_REQUIREMENTS = ['PyYAML', 'pytest', 'tqdm', 'seaborn', 'matplotlib']

# If we are packaging for an ARM platform, we need to specify different dependency packages
_IS_ARM = any([bool(re.match(r"--plat-name=.*_aarch64", a)) for a in sys.argv])
if _IS_ARM:
    #_ML_REQUIREMENTS = ['tensorflow-aarch64<=2.7.1', 'tflite==2.4.0']
    _EXTRAS = {
        'runtime' : ['mera-tvm-runtime']
    }
else: # x86
    _ML_REQUIREMENTS = [
        'onednn-cpu-gomp==2022.0.2',
        'tensorflow<=2.9.0',
        'tflite==2.4.0',
        'torch<=1.12.1',
        'torchvision<=0.13.1'
    ]
    _EXTRAS = {
        'full' : (['mera-tvm-full'] + _ML_REQUIREMENTS),
        'runtime' : ['mera-tvm-runtime']
    }

_BINARIES = [
    'bin_utils/intel_get_board_id',
    'bin_utils/ifc_uio_5.15.0-56-generic.ko',
    'bin_utils/sakura_ddr_init',
    'bin_utils/ec_dma_daemon_proc',
    'pcie_driver/usd_pcie.c',
    'pcie_driver/usd_pcie.h',
    'pcie_driver/pcieioctl.h',
    'pcie_driver/Makefile'
]

_CONSOLE_ENTRY_POINTS = [
    'mera = mera.app.mera:main'
]

setup(
    name='mera',
    version=get_version(),
    description='An heterogeneous deep learning compiler framework.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Edgecortix-Inc/mera',
    author='EdgeCortix Inc.',
    author_email='mera-compiler@edgecortix.com',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers"
    ],
    keywords='', # FIXME
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6',
    install_requires=_BASE_REQUIREMENTS,
    extras_require=_EXTRAS,
    scripts=[],
    package_data={"mera" : _BINARIES},
    data_files=[],
    entry_points={
        'console_scripts' : _CONSOLE_ENTRY_POINTS
    },
    project_urls={
        'Bug Tracker': 'https://github.com/Edgecortix-Inc/mera/issues',
        'Web Page': 'https://www.edgecortix.com',
    },
    cmdclass=cmdclass,
)
