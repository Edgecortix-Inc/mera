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
/**
 * @brief Library with API for running inference on a MERA deployment from C++.
 * API calls are similar to Python library. See inference app for an example on how to use this library.
 */
#ifndef MERA_TVM_RUN_H
#define MERA_TVM_RUN_H

#include <vector>
#include <optional>
#include <memory>
#include <experimental/filesystem>

namespace mera {

using path_t = std::experimental::filesystem::path;

/**
 * @brief List of possible MERA targets.
 */
enum class Target {
  IP,
  Interpreter,
  InterpreterHw,
  Simulator,
  VerilatorSimulator,
  None
};

std::string TargetToStr(const mera::Target &t);
mera::Target StrToTarget(const std::string &t_str);
std::ostream &operator<<(std::ostream &os, const Target &t);

/**
 * @brief List of possible MERA devices.
 */
enum class Device : int {
  Sakura1 = 1,
  Xilinx = 2,
  Intel = 3
};

mera::Device StrToDev(const std::string &d_str);


/**
 * @brief Base class that can communicate and run with a MERA deployment.
 */
struct MeraModelRunner {
  virtual ~MeraModelRunner() {}

  /**
   * @brief Sets the input data for the model via a raw pointer. Size will be checked against
   * expected tensor dimensions to prevent buffer overflows.
   *
   * @param in_ptr The data pointer with the contents of the input tensor.
   * @param size Size, in bytes, of the input tensor.
   */
  virtual void SetInput(const void *in_ptr, size_t size) = 0;

  /**
   * @brief Set the Input object via a std vector.
   *
   * @param data The data vector
   */
  template<typename T>
  void SetInput(const std::vector<T> &data) { SetInput(data.data(), data.size() * sizeof(T)); }

  /**
   * @brief Sets the input data for the model via reading a binary file.
   *
   * @param input_file Path to the binary file from where the data will be loaded.
   */
  virtual void SetInput(const path_t &input_file) = 0;
  void SetInput(const std::string &input_file) { SetInput(path_t(input_file)); }

  /**
   * @brief Runs the model. Requires providing input data first. @see SetInput()
   */
  virtual void Run() = 0;

  /**
   * @brief Returns the number of outputs on this model.
   */
  virtual unsigned GetNumOutputs() = 0;

  /**
   * @brief Get the output data tensor at index 'output_idx' as a float vector.
   */
  virtual std::vector<float> GetOutputFloat(unsigned output_idx = 0) = 0;

  /**
   * @brief Get the output data tensor at index 'output_idx' as a byte array.
   */
  virtual std::vector<uint8_t> GetOutput(unsigned output_idx = 0) = 0;
};


/**
 * @brief Base class representing a pre-built MERA deployment.
 */
class MeraTvmDeployment {
  const path_t lib_path_;
  const path_t params_path_;
  const path_t lib_json_path_;
public:
  MeraTvmDeployment(const path_t &root_path, const mera::Target &target);

  /**
   * @brief Prepares the model for running with a given target.
   * @return const MeraTvmModelRunner Model runner object
   */
  std::unique_ptr<MeraModelRunner> GetRunner(const mera::Device &dev);
};


/**
 * @brief Loads an already build deployment from a directory
 *
 * @param deploy_dir Root directory of a MERA deployment project or path to MERA deployment built artifacts.
 * @param target If there  are multiple targets build in the MERA project, selects which one when provided.
 *   Optional if not loading a project of if there is only a single target built.
 * @return MeraTvmDeployment Reference to a deployment object.
 */
MeraTvmDeployment LoadMeraDeployment(const path_t &deploy_dir, const std::optional<mera::Target> &target = {});

} // namespace mera

#endif // MERA_TVM_RUN_H
