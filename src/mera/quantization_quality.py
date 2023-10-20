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
"""MERA Quantization quality metrics."""
import numpy as np
import math
import json
from typing import List, Union, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class QuantizationQualityMetrics:
    def __init__(self, ref_data, qtz_data):
        assert isinstance(ref_data, list) and isinstance(qtz_data, list)
        assert len(ref_data) == len(qtz_data)
        assert len(ref_data) > 0
        assert np.alltrue([len(r) == len(q) for r,q in zip(ref_data, qtz_data)])

        self.ref_data = ref_data
        self.qtz_data = qtz_data
        self.error = [tuple([a - b for a,b in zip(r,q)]) for r,q in zip(self.ref_data, self.qtz_data)]

    def __reduce_img(self, data, f):
        assert isinstance(data, list)
        return [tuple([f(x) for x in d]) for d in data]

    @property
    def images(self) -> int:
        return len(self.qtz_data)

    @property
    def num_elements(self) -> Tuple[int]:
        r = self.__reduce_img(self.qtz_data, lambda x: int(x.size))
        return r if self.images == 1 else r[0]

    @property
    def mean_error(self) -> List[Tuple[float]]:
        return self.__reduce_img(self.error, lambda x: float(np.mean(x)))

    @property
    def std_dev(self) -> List[Tuple[float]]:
        return self.__reduce_img(self.error, lambda x: float(np.std(x)))

    @property
    def max_abs_error(self) -> List[Tuple[float]]:
        return self.__reduce_img(self.error, lambda x: float(np.max(np.abs(x))))

    @property
    def mean_squared_error(self) -> List[Tuple[float]]:
        return self.__reduce_img(self.error, lambda x: float(np.mean(x ** 2)))

    @property
    def range(self) -> List[Tuple[float]]:
        return self.__reduce_img(self.ref_data, lambda x: float(np.max(x) - np.min(x)))

    @property
    def max(self) -> List[Tuple[float]]:
        return self.__reduce_img(self.ref_data, lambda x: float(np.max(x)))

    @property
    def psnr_all(self) -> List[Tuple[float]]:
        max = self.max
        mse = self.mean_squared_error
        psnr = []
        f = lambda max,mse : 10 * math.log10((max ** 2) / mse) if mse > 0 else 100.0
        for max_t, mse_t in zip(max, mse):
            psnr.append(tuple([f(max_e, mse_e) for max_e,mse_e in zip(max_t, mse_t)]))
        return psnr

    @property
    def psnr(self) -> Tuple[float]:
        return tuple(np.round(np.min(np.array(self.psnr_all), axis=0), decimals=2).tolist())

    @property
    def all_scores(self) -> List[Tuple[float]]:
        """Custom function to give a score % on the quantization quality of a model. Computed as
        (1 - (max_abs_error / range)) * 100
        Should return a percentage on the worst case accuracy drop in the output data set.
        """
        # FIXME - Better score when range == 0
        score_f = lambda max_e, range: ((1 - max_e / range) * 100) if range != 0 else (100 if max_e == 0 else 0)
        return [tuple([score_f(max_err_e, range_e) for max_err_e,range_e in zip(max_err,range)]) \
            for max_err,range in zip(self.max_abs_error, self.range)]

    @property
    def score(self) -> Tuple[float]:
        """Simplified score function which will be the median across all the evaluation images
            for each of the outputs."""
        return tuple(np.round(np.min(np.array(self.all_scores), axis=0), decimals=2))

    def to_dict(self):
        _METRICS = ["images", "num_elements", "mean_error", "std_dev", "max_abs_error",
            "mean_squared_error", "range", "all_scores", "score", "psnr", "psnr_all"]
        return {m: getattr(self, m) for m in _METRICS}


def calculate_quantization_quality(ref_data : Union[Tuple[np.ndarray], List[Tuple[np.ndarray]]],
        qtz_data : Union[Tuple[np.ndarray], List[Tuple[np.ndarray]]]) -> QuantizationQualityMetrics:
    _ref_data = ref_data if isinstance(ref_data, list) else [ref_data]
    _qtz_data = qtz_data if isinstance(qtz_data, list) else [qtz_data]
    return QuantizationQualityMetrics(_ref_data, _qtz_data)


class MeraQualityContainer:
    def __init__(self, out_metrics):
        self.out_metrics = out_metrics

    def __getattr__(self, __attr: str):
        if __attr in self.__dict__:
            return getattr(self, __attr)
        return getattr(self.out_metrics, __attr)

    def plot_histogram(self, dest_file : Path, channels : Union[int, Tuple[int]] = None,
        output_id : int = None, plot_title : str = None, axis : int = None) -> None:
        """Computes the histogram of the distribution comparing the floating point reference with the MERA quantized results.

        :param dest_file: Path to destination file where the plotted histogram will be saved.
        :param channels: Selection of channels where the distribution should be computed. If 'None' provided,
            it will compute the whole tensor.
            Accepts an integer for selecting a single channel or a tuple with ('start_channel', 'end_channel') values.
        :param output_id: If this quality container wraps several output tensors, selects which one to apply the operation.
            If there is a single output this argument is ignored.
        :param plot_title: If provided, overrides the autogenerated title of the plot.
        :param axis: Axis of tensor's dimension for the channels. It will be 1 for NCHW and 3 for NHWC.
            Only used when providing a value for 'channels' argument. 
        """
        # TODO - Currently only generating for the first evaluation image.
        ref_data = self.out_metrics.ref_data[0]
        mera_data = self.out_metrics.qtz_data[0]

        if len(ref_data) > 1 and output_id is None:
            raise ValueError(f"Quality container has multiple outputs but 'output_id' parameter has not been specified.\n"
                + f"Please set which of the {len(ref_data)} outputs you want to compute the histogram.")
        if not output_id:
            output_id = 0
        ref_data = ref_data[output_id]
        mera_data = mera_data[output_id]
        if len(ref_data.shape) < len(mera_data.shape):
            ref_data = np.expand_dims(ref_data, 0)
        if channels is None:
            ref_data = ref_data.flatten()
            mera_data = mera_data.flatten()
        else:
            if axis is None and len(ref_data.shape) > 1:
                raise ValueError(f'Need to provide an axis when selecting channels from multidimensional data.\n'
                    + f'Current shape: {ref_data.shape}')
            slc = channels if isinstance(channels, int) else slice(channels[0], channels[1])
            ref_data = np.take(ref_data, slc, axis=axis).flatten()
            mera_data = np.take(ref_data, slc, axis=axis).flatten()
        if plot_title is None:
            plot_title = f"Data Histogram on "\
                + (f'Channel(s) {channels}' if channels is not None else "Tensor") + f' of output #{output_id}'
        plt.clf()
        sns.histplot(data={"fp32" : ref_data, "mera_qtz" : mera_data}, stat="percent").set(title=plot_title)
        plt.savefig(dest_file)

class MeraDebugQualityContainer(MeraQualityContainer):
    def __dequantize(qtzed_images, q_params):
        deq_data = []
        data_dims = len(qtzed_images[0].shape)
        if data_dims == 1:
            scl = np.array([q[0] for q in q_params])
            zp = np.array([q[1] for q in q_params])
        elif data_dims == 4:
            n, c, h, w = qtzed_images[0].shape
            if len(q_params) == 1:
                scl = np.ones_like(qtzed_images[0]) * q_params[0][0]
                zp = np.ones_like(qtzed_images[0]) * q_params[0][1]
            elif n == 1:
                # Data dequantizing
                # Converts a list of channel scalars into a NCHW tensor with data filled out
                __expand_per_channel = lambda data, h, w: np.expand_dims(np.array([np.tile(x, (h, w)) for x in data]), axis=0)
                scl = __expand_per_channel([q[0] for q in q_params], h, w)
                zp = __expand_per_channel([q[1] for q in q_params], h, w)
            else:
                # Weight dequantizing
                __expand_per_channel = lambda data, c, h, w: np.array([np.tile(x, (c, h, w)) for x in data])
                scl = __expand_per_channel([q[0] for q in q_params], c, h, w)
                zp = __expand_per_channel([q[1] for q in q_params], c, h, w)
        else:
            raise ValueError(f"Cannot dequantize data with {data_dims}D shape")
        for img in qtzed_images:
            deq_data.append(scl * (img.astype(np.float32) - zp))
        return deq_data

    def __init__(self, out_metrics, qtzer_node_data, q_params):
        MeraQualityContainer.__init__(self, out_metrics)
        self.node_metrics = {}
        for node_name, node_data in qtzer_node_data.items():
            ref_data = node_data["ref"]
            got_data_qtz = node_data["got"]

            try:
                got_data = MeraDebugQualityContainer.__dequantize(got_data_qtz, q_params[node_name])
                __wrap = lambda data : [(x,) for x in data]
                self.node_metrics[node_name] = calculate_quantization_quality(__wrap(ref_data), __wrap(got_data))
            except ValueError as ex:
                print(f'ERROR While dequantizing {node_name} {got_data_qtz[0].shape} {ex}')

    def summary(self):
        data = {}
        for op_name, metric in self.node_metrics.items():
            data[op_name] = metric.to_dict()
        with open('qtz_debug_metrics.json', 'w') as w:
            w.write(json.dumps(data))
