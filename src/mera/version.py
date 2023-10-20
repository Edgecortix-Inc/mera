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


# Current version, modify this file for release using "bump_version.py"
__version__ = "1.5.0"
__release_date__ = "20/10/2023"

def get_mera_tvm_version() -> str:
    """Gets the version string for mera-tvm module

    :return: mera-tvm version
    """
    from tvm import __version__ as __tvm_version__
    return __tvm_version__

def get_mera_dna_version() -> str:
    """Gets the version string for libmeradna

    :return: Summary of libmeradna version
    """
    from tvm.relay.mera import get_version as __get_mera_dna_version
    return __get_mera_dna_version()

def get_mera_version() -> str:
    """Gets the version string for Mera

    :return: Version string for Mera
    """
    return f'mera version {__version__} released on {__release_date__}'

def _get_version_metadata():
    meradna_ver = get_mera_dna_version()

    return {
        "mera": {
            "version": __version__,
            "release_date": __release_date__
        },
        "mera-tvm": {
            "version": get_mera_tvm_version(),
            "release_date": None
        },
        "mera-dna": {
            "version": meradna_ver.split(' ')[1],
            "release_date": None
        }
    }

def get_versions() -> str:
    """Return a summary of all installed modules on the Mera environment

    :return: List of all module's versions.
    """
    s = 'Mera Environment Versions:\n'
    for m, d in _get_version_metadata().items():
        s += f' * {m} version {d["version"]} '
        if d["release_date"]:
            s += f'released on {d["release_date"]}'
        s += '\n'
    return s
