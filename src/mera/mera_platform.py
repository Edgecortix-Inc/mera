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
"""MERA platform selection"""

from enum import Enum


class Platform(Enum):
    """List of all valid MERA platforms"""
    DNAF200L0001 = 'DNAF200L0001'
    DNAF200L0002 = 'DNAF200L0002'
    DNAF200L0003 = 'DNAF200L0003'

    DNAF100L0001 = 'DNAF100L0001'
    DNAF100L0002 = 'DNAF100L0002'
    DNAF100L0003 = 'DNAF100L0003'

    DNAF132S0001 = 'DNAF132S0001'

    DNAF232S0001 = 'DNAF232S0001'
    DNAF232S0002 = 'DNAF232S0002'

    DNAF632L0001 = 'DNAF632L0001'
    DNAF632L0002 = 'DNAF632L0002'
    DNAF632L0003 = 'DNAF632L0003'

    DNAF300L0001 = 'DNAF300L0001'

    DNAA400L0001 = 'DNAA400L0001'

    DNAA600L0001 = 'DNAA600L0001'
    DNAA600L0002 = 'DNAA600L0002'
    SAKURA_1     = 'DNAA600L0002'
    SAKURA_2C    = 'DNAA600L0003'


    def __str__(self):
        return self.value
