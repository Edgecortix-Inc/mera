# Copyright 2023 EdgeCortix Inc.
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
"""Container for Power metric analysis"""

import re
import json
import numpy as np

class PowerMetrics:
    """Container class for analysis of power measurements obtained from MERA."""

    def __init__(self, raw_str):
        self.data = {}
        pat = re.compile(r"([a-zA-Z0-9]+)=([\[\],0-9]+),")
        for g_power in filter(None, raw_str.split('|')):
            # Make regex parsing easier
            if g_power[-1] != ',':
                g_power += ','
            for m in pat.finditer(g_power):
                m_name = m.group(1)
                m_data = m.group(2)
                if m_name not in self.data:
                    self.data[m_name] = []
                self.data[m_name].append(np.array(json.loads(m_data)))

        self.__report_raw = {}
        for metric_name,metric_data_list in self.data.items():
            self.__report_raw[metric_name] = {}
            __reduce = lambda f: [np.round(f(x) / 1000, 3) for x in metric_data_list]
            self.__report_raw[metric_name]["avg"] = __reduce(np.mean)
            self.__report_raw[metric_name]["min"] = __reduce(np.min)
            self.__report_raw[metric_name]["max"] = __reduce(np.max)
            self.__report_raw[metric_name]["median"] = __reduce(np.median)
        self.__report_str = ''
        for metric_name,report in self.__report_raw.items():
            self.__report_str += f'{metric_name} Power Report:\n'
            __msg = lambda n,fn: f'  * {n}: {np.round(fn(report[n]), 3)}W ({" ".join([str(x) + "W" for x in report[n]])})\n'
            self.__report_str += __msg('avg', np.mean) + __msg('min', np.min) + __msg('max', np.max) + __msg('median', np.median)
        if not self.__report_raw:
            self.__report_str = "<No power metrics available>"

    @property
    def report(self) -> str:
        """Gets a report string displaying the Watt consumption of the latest run model

        :return: A string containing a power report.
        """
        return self.__report_str

    @property
    def report_raw(self) -> dict:
        """Gets the report string in dictionary form, useful for extra data analysis.

        :return: The values on the :func:`report` as a dictionary
        """
        return self.__report_raw

    def __str__(self):
        return self.report
