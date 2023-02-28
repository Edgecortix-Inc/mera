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
 * @brief App that makes use of mera_tvm_run library that allows to
 * run inference on deployment projects using the MERA C++ API.
 */
#include <iostream>
#include <fstream>
#include <getopt.h>
#include <mera_tvm_run.h>

template<typename T>
void SaveToFile(const std::vector<T> &v, const std::string &fname) {
  std::ofstream fout(fname, std::ios::out | std::ios::binary);
  fout.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
  fout.close();
  std::cout << "   > Data saved to file " << fname << std::endl;
}

void Usage() {
  std::cout << "Usage: ./inference -p <d_dir> -i <in_data> -n <inf_runs> [-h] [-o out_file]" << std::endl;
  std::cout << "  -p/--project_dir: Path to directory where a MERA deployment has been built" << std::endl;
  std::cout << "  -i/--input_data:  Path to a binary file containing input data to run inference" << std::endl;
  std::cout << "  -t/--target:      mera::Target to load from the project, if needed" << std::endl;
  std::cout << "  -n/--n_runs:      Number of inference runs to repeat" << std::endl;
  std::cout << "  -o/--output_name: Name for generated output from inference. By default will be output_0.bin. Provide only basename." << std::endl;
  std::cout << "  -h/--help:        Displays this message." << std::endl;
}

int main(int argc, char **argv) {
  std::string proj_dir, in_data, out_path = "output";
  std::optional<mera::Target> target;
  int n_runs = 1;
  const struct option longopts[] = {
    {"help", no_argument, 0, 'h'},
    {"project_dir", required_argument, 0, 'p'},
    {"input_data", required_argument, 0, 'i'},
    {"output_name", required_argument, 0, 'o'},
    {"target", required_argument, 0, 't'},
    {"n_runs", required_argument, 0, 'n'},
    {0, 0, 0, 0}
  };

  int c;
  int opt_idx;
  while (true) {
    c = getopt_long(argc, argv, "p:i:n:o:t:h", longopts, &opt_idx);
    if (-1 == c) {
      break;
    }
    switch (c) {
      case 'p':
        proj_dir = std::string(optarg);
        break;
      case 'i':
        in_data = std::string(optarg);
        break;
      case 'n':
        n_runs = atoi(optarg);
        break;
      case 'o':
        out_path = std::string(optarg);
        break;
      case 't':
        target = mera::StrToTarget(std::string(optarg));
        break;
      case '?':
      case 'h':
      default:
        Usage();
        return -1;
    }
  }

  if (proj_dir.empty() || in_data.empty()) {
    Usage();
    throw std::invalid_argument("Missing some mandatory arguments");
  }

  if (n_runs < 1) {
    throw std::invalid_argument("Number of runs must be bigger than 0, not " + std::to_string(n_runs));
  }

  std::cout << "Running inference on MERA project " << proj_dir << "..." << std::endl;

  // Load deployment project
  std::cout << "Loading MERA deployment..." << std::endl;
  auto deploy = mera::LoadMeraDeployment(mera::path_t(proj_dir), target);

  // Get and configure the runner
  std::cout << "Configuring MERA runner..." << std::endl;
  auto runner = deploy.GetRunner();
  runner->SetInput(in_data);

  // Run inference N times
  std::cout << "Running inference " << n_runs << " time(s)" << std::endl;
  for (int i = 0; i < n_runs; ++i) {
    runner->Run();
  }

  // Grab and dump output
  for (int i = 0; i < runner->GetNumOutputs(); ++i) {
    auto out_data = runner->GetOutputFloat();
    std::cout << " > Output #" << i << " got " << out_data.size() << " elements." << std::endl;
    SaveToFile(out_data, out_path + "_" + std::to_string(i) + ".bin");
  }

  std::cout << "SUCCESS" << std::endl;
  return 0;
}
