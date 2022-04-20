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
"""Generate PEP440 compliant version string"""


import argparse
import pathlib
import re
import sys
from packaging.version import Version

ver_file = pathlib.Path(__file__).parent.resolve() / 'src/mera/version.py'

def get_version():
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, ver_file.read_text(), re.M)
    if mo:
        return mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (ver_file,))

def update_version(old_v, new_v):
    ver_contents = ver_file.read_text()
    ver_contents = ver_contents.replace(f'__version__ = "{old_v}"', f'__version__ = "{new_v}"', 1)
    ver_file.write_text(ver_contents)

def main():
    arg_p = argparse.ArgumentParser(description='Bump version for mera')
    arg_p.add_argument('-M', '--major', required=False, help='Update major version')
    arg_p.add_argument('-m', '--minor', required=False, help='Update to minor version')
    arg_p.add_argument('-p', '--pre', required=False, help='Pre-release tag to apply')
    arg_p.add_argument('-l', '--local', required=False, help='Update local version')
    arg_p.add_argument('-o', '--override', required=False, action='store_true',
        help='Whether to take the values from command line to override the version completely')
    arg_p.add_argument('-v', '--version', required=False, action='store_true',
        help='Displays the current Mera version')

    args = arg_p.parse_args()

    v = get_version()
    if args.version:
        print(v)
        return 0
    print(f'Current version {v}')
    _v = Version(v)

    if not args.override:
        major = args.major if args.major else _v.major
        minor = args.minor if args.minor else _v.minor
        if args.pre:
            pre = args.pre
        else:
            pre = f"{_v.pre[0]}{_v.pre[1]}" if _v.pre else None
        local = args.local if args.local else _v.local
    else:
        major, minor, pre, local = (args.major, args.minor, args.pre, args.local)

    # Reassemble version
    new_v = f'{major}.{minor}'
    if pre:
        new_v += f'.{pre}'
    if local:
        new_v += f'+{local}'
    print(f'New version = {new_v}')

    # Save version
    update_version(v, new_v)
    print('SUCCESS')
    return 0

if __name__ == '__main__':
    sys.exit(main())
