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

_BASE_REQUIREMENTS = ['onednn-cpu-gomp==2022.0.2', 'PyYAML', 'pytest']
_ML_REQUIREMENTS = ['tensorflow<=2.6.2', 'tflite==2.4.0', 'torch==1.7.1']

# If we are packaging for an ARM platform, we need to specify different dependency packages
_IS_ARM = any([bool(re.match(r"--plat-name=.*_aarch64", a)) for a in sys.argv])
if _IS_ARM:
    _ML_REQUIREMENTS = ['tensorflow-aarch64<=2.7.1', 'tflite==2.4.0']

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
    extras_require={
        'host-only' : (['mera-tvm-host-only==1.0'] + _ML_REQUIREMENTS),
        'full' : (['mera-tvm-full==1.0'] + _ML_REQUIREMENTS),
        'runtime' : ['mera-tvm-runtime==1.0']
    },
    package_data={},
    data_files=[],
    entry_points={},
    project_urls={
        'Bug Tracker': 'https://github.com/Edgecortix-Inc/mera/issues',
        'Web Page': 'https://www.edgecortix.com',
    },
)
