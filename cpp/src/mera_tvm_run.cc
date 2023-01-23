/*
 * Copyright 2022 EdgeCortix Inc
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "mera_tvm_run.h"

#include <fstream>
#include <sstream>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <iostream>

template<typename T>
std::vector<T> LoadBinary(const std::string& bin_file) {
  std::ifstream file(bin_file.c_str(), std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open file: " + bin_file);
  }

  file.seekg(0, file.end);
  const uint32_t file_size = static_cast<uint32_t>(file.tellg());
  file.seekg(0, file.beg);

  const auto file_buffer = std::unique_ptr<char>(new char[file_size]);
  file.read(file_buffer.get(), file_size);

  if (file.bad() || file.fail()) {
    throw std::runtime_error("An error occured while reading the file");
  }

  file.close();

  auto ptr = reinterpret_cast<T*>(file_buffer.get());
  const auto num_elements = file_size / sizeof(T);
  return std::vector<T>(ptr, ptr + num_elements);
}

size_t GetShapeSize(const tvm::runtime::ShapeTuple &shape) {
  size_t s = 1;
  for (int i = 0; i < shape.size(); ++i) {
    s *= shape[i];
  }
  return s;
}

namespace mera {

std::string TargetToStr(const mera::Target &t) {
  switch (t) {
    case Target::IP:                 return "IP";
    case Target::Interpreter:        return "Interpreter";
    case Target::InterpreterHw:      return "InterpreterHw";
    case Target::Simulator:          return "Simulator";
    case Target::VerilatorSimulator: return "VerilatorSimulator";
    case Target::None:               return "None";
    default:                         return "???";
  }
}

mera::Target StrToTarget(const std::string &t_str) {
#define STR2T(__name) if (t_str == #__name) { return mera::Target::__name; }
  STR2T(IP)
  STR2T(Interpreter)
  STR2T(InterpreterHw)
  STR2T(Simulator)
  STR2T(VerilatorSimulator)
  STR2T(None)
#undef STR2T
  throw std::runtime_error("Unknown Target value " + t_str);
}

std::ostream &operator<<(std::ostream &os, const Target &t) {
  os << TargetToStr(t);
  return os;
}

class MeraTvmModelRunner : public MeraModelRunner {
  tvm::runtime::Module rt_mod_;
  tvm::runtime::DataType in_type_;
  tvm::runtime::ShapeTuple in_shape_;
  size_t in_size_;
  uint8_t in_type_bits_;
  tvm::runtime::PackedFunc run_func_;

  const int device_type_ = kDLCPU;
  const int device_id_ = 0;

  friend class MeraTvmDeployment;

  MeraTvmModelRunner(tvm::runtime::Module rt_mod): MeraModelRunner(), rt_mod_(rt_mod) {
    auto get_input_func = rt_mod_.GetFunction("get_input");
    auto xx = get_input_func(0).operator tvm::runtime::NDArray();
    in_shape_ = xx.Shape();
    in_size_ = GetShapeSize(in_shape_);
    in_type_ = xx.DataType();
    in_type_bits_ = in_type_.bits();
    run_func_ = rt_mod_.GetFunction("run");
  }

  virtual ~MeraTvmModelRunner() {}

  void SetInputInternal(const tvm::runtime::NDArray &in_data) {
    auto data_size = GetShapeSize(in_data.Shape());
    if (in_size_ != data_size) {
      throw std::runtime_error("Expected input tensor to be of shape size " + std::to_string(in_size_)
        + " but provided input data is of shape size " + std::to_string(data_size));
    }
    if (in_type_ != in_data.DataType()) {
      std::stringstream ss;
      ss << "Expected input type to be " << in_type_ << " but provided data with input type "
        << in_data.DataType();
      throw std::runtime_error(ss.str());
    }
    auto set_input_func = rt_mod_.GetFunction("set_input");
    if (set_input_func == nullptr) {
      throw std::runtime_error("ERROR accessing 'set_input' TVM function.");
    }
    set_input_func(0, in_data);
  }

  virtual void SetInput(const path_t &input_file) override {
    if (!std::experimental::filesystem::is_regular_file(input_file)) {
      throw std::invalid_argument("Input file " + input_file.string() + " could not be found.");
    }
    auto data_bin = LoadBinary<uint8_t>(input_file);
    return SetInput(data_bin.data(), data_bin.size());
  }

  virtual void SetInput(const void *in_ptr, size_t size) override {
    DLDevice dev;
    dev.device_id = device_id_;
    dev.device_type = DLDeviceType(device_type_);
    auto data_arr = tvm::runtime::NDArray::Empty(in_shape_, in_type_, dev);
    void *data_ptr = data_arr->data;
    const auto total_size = in_type_.bytes() * in_size_;
    if (total_size != size) {
      throw std::runtime_error("SetInput: Unexpected input data byte size. Expected "
        + std::to_string(total_size) + " but got " + std::to_string(size));
    }
    std::memcpy(data_ptr, in_ptr, size);
    SetInputInternal(data_arr);
  }

  virtual void Run() override { run_func_(); }

  virtual unsigned GetNumOutputs() override {
    auto get_num_outputs_func = rt_mod_.GetFunction("get_num_outputs");
    return static_cast<unsigned>(get_num_outputs_func().operator int());
  }

  tvm::runtime::NDArray GetOutputRaw(unsigned output_idx) {
    auto get_output_func = rt_mod_.GetFunction("get_output");
    return get_output_func(output_idx);
  }

  template<typename T>
  std::vector<T> UnpackNDArray(const tvm::runtime::NDArray &data) {
    const auto num_elements = GetShapeSize(data.Shape());
    auto *out_ptr = reinterpret_cast<T*>(data->data);
    return std::vector<T>(out_ptr, out_ptr + num_elements);
  }

  virtual std::vector<uint8_t> GetOutput(unsigned output_idx) override {
    return UnpackNDArray<uint8_t>(GetOutputRaw(output_idx));
  }

  virtual std::vector<float> GetOutputFloat(unsigned output_idx) override {
    auto out = GetOutputRaw(output_idx);
    if (!out.DataType().is_float()) {
      throw std::runtime_error("GetOutputFloat: Tried to get float output data on idx "
        + std::to_string(output_idx) + ", but output is not of float type.");
    }
    return UnpackNDArray<float>(out);
  }
};


MeraTvmDeployment::MeraTvmDeployment(const path_t &root_path, const mera::Target &target):
  lib_path_(root_path / "deploy.so"), params_path_(root_path / "deploy.params"), lib_json_path_(root_path / "deploy.json") {}


std::unique_ptr<MeraModelRunner> MeraTvmDeployment::GetRunner() {
  const int device_type_ = kDLCPU;
  const int device_id_ = 0;
  std::ifstream json_in(lib_json_path_.c_str(), std::ios::in);
  const std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

  auto mod_syslib = tvm::runtime::Module::LoadFromFile(lib_path_);
  tvm::runtime::Module rt_mod = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))
    (json_data, mod_syslib, device_type_, device_id_);

  auto load_params_func = rt_mod.GetFunction("load_params");
  auto params_data = LoadBinary<char>(params_path_);
  TVMByteArray params_arr;
  params_arr.data = params_data.data();
  params_arr.size = params_data.size();
  load_params_func(params_arr);

  return std::unique_ptr<MeraModelRunner>(new MeraTvmModelRunner(std::move(rt_mod)));
}


MeraTvmDeployment LoadMeraDeployment(const path_t &deploy_dir, const std::optional<mera::Target> &target) {
  if (!std::experimental::filesystem::is_directory(deploy_dir)) {
    throw std::invalid_argument("Deployment directory '" + deploy_dir.string()
      + "' is not a directory or could not be accessed");
  }
  const bool is_mera_prj = std::experimental::filesystem::is_regular_file(deploy_dir / "project.mdp");
  path_t artifact_folder = deploy_dir;
  auto has_artifact = [&](const auto &p) -> bool {
    return std::experimental::filesystem::is_regular_file(p / "deploy.so")
      && std::experimental::filesystem::is_regular_file(p / "deploy.params")
      && std::experimental::filesystem::is_regular_file(p / "deploy.json");
  };
  auto get_res_dir = [&](const auto &t) { return deploy_dir / "build" / TargetToStr(t) / "result"; };
  static std::vector<mera::Target> k_all_targets = {
    mera::Target::IP,
    mera::Target::Interpreter,
    mera::Target::InterpreterHw,
    mera::Target::Simulator,
    mera::Target::VerilatorSimulator
  };
  auto tt = target;
  if (is_mera_prj) {
    std::vector<mera::Target> built_targets;
    for (const auto &t : k_all_targets) {
      if (has_artifact(get_res_dir(t))) {
        built_targets.emplace_back(t);
      }
    }
    if (built_targets.empty()) {
      throw std::invalid_argument("Could not find any built deployments in MERA project '" + deploy_dir.string() + "'");
    }
    if (built_targets.size() > 1) {
      if (target.has_value()) {
        artifact_folder = get_res_dir(*target);
      } else {
        std::stringstream ss;
        ss << "Multiple MERA deployments found at project '" << deploy_dir << "'.\n"
        << "Please specify which one to load using target argument. Found:\n";
        for (const auto &t : built_targets) {
          ss << t << ", ";
        }
        throw std::invalid_argument(ss.str());
      }
    } else {
      // Only one deployment
      artifact_folder = get_res_dir(built_targets[0]);
      tt = built_targets[0];
    }
  } else if (!has_artifact(deploy_dir)) {
    throw std::invalid_argument("Could not find any built deployment at directory '" + deploy_dir.string()
      + "', and folder is not a MERA project");
  }
  return MeraTvmDeployment(artifact_folder, tt.value_or(mera::Target::None));
}

} // namespace mera
