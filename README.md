# FTTN: Feature-Targeted Testing for Numerical Properties of NVIDIA & AMD Matrix Accelerators
FTTN is a test suite to evaluate the numerical behaviors of GPU matrix accelerators (NVIDIA Tensor Cores and AMD Matrix Cores) in a quick and simple setting. Matrix accelerators are heavily used in today's computationally intense applications to speed up matrix multiplications. This test suite provides a comprehensive study on the numerical behaviors of these accelerators, including support for subnormals, rounding modes, extra precision bits and FMA features. See the following table for the features FTTN tests:

| **Test name**            | **Test description**                            |
|--------------------------|-------------------------------------------------|
| T_si_no                  | Subnormal ins, normal outs?                     |
| T_no_so                  | Normal ins, subnormal outs?                     |
| T_sa                     | Subnormal accumulation?                         |
| T_1_bit                  | One extra bit?                                  |
| T_rnd_dir                | Rounding direction?                             |
| T_3_bits_fin_rnd         | Extra 3 bits? Final rounding mode?              |
| T_prod                   | Rounding mode of product                        |
| T_blk_fma_width          | Blocked FMA width                               |
| T_pres_extra_acc         | Extra bits are preserved during accumulation?   |
| T_acc_order              | Accumulation order control?                     |

## Getting start
FTTN is easy to use in any GPUs which configures matrix accelerators 
### Change the Configuration
Set up CUDA compiler `NVCC`, GPU architecture `CUDA_ARCH` and HIP compiler `HIPCC` in the `Makefile`.  
```
NVCC = nvcc
CUDA_ARCH = 80
HIPCC = hipcc
```

#### Executing all tests
For nvidia GPUs
```bash
make nvidia
```
For AMD GPUs,
```bash
make amd
```

#### Executing a single test
To run tests for a specific data type, use the command
```bash
make <cppfilename>-<GPU_Type>
```
For example, to run all tests for the FP16 input data type on an NVIDIA GPU, the command should be 
```bash
make fp16_NVIDIA
```

#### Run with sbatch script on SLRUM system
The tests can also be run using `.sbatch` script. Ensure you load the relevant modules (C++ compiler, CUDA for NVIDIA GPUs, or ROCm for AMD GPUs) before executing the `make` command. 

#### Results
The outcomes of the tests will be recorded in a text file named according to the pattern `<filename>_result<GPU_type>.txt`. For instance, the results for FP16 test on NVIDIA GPU would be in `fp16_resultNVIDIA.txt`. At the end of the file, there’s a summary that compiles the results into a format corresponding to a row in [Table IV](https://github.com/user-attachments/assets/4313f31e-f621-4c62-aa42-7c4aeb20f896) of the paper. Before this summary, the results of each individual test are printed, providing an in-depth review of the specific test outcomes.

## Experiments Results
We used our test suites in V100, A100 (NVIDIA 3060), H100, MI100, MI250X and the results is shown as the table here (Table IV in the paper):

<img width="661" alt="image" src="https://github.com/user-attachments/assets/4313f31e-f621-4c62-aa42-7c4aeb20f896">

## Contact
For questions, contact Ganesh Gopalakrishnan [ganesh@cs.utah.edu](mailto:ganesh@cs.utah.edu) and
 Xinyi Li [xin_yi.li@utah.edu](mailto:xin_yi.li@utah.edu).

To cite GPU-FPX, please use
```
@inproceedings{li2024fttnfeaturetargetedtestingnumerical,
      url={https://arxiv.org/abs/2403.00232},
      title={FTTN: Feature-Targeted Testing for Numerical Properties of NVIDIA & AMD Matrix Accelerators}, 
      author={Xinyi Li and Ang Li and Bo Fang and Katarzyna Swirydowicz and Ignacio Laguna and Ganesh Gopalakrishnan},
      year={2024},
      booktitle={2024 IEEE/ACM 24th International Symposium on Cluster, Cloud and Internet Computing (CCGrid)},
      year={2024},
      organization={IEEE}
}
```

## License
GPU-FPX is distributed under the terms of the MIT license.

See [LICENSE-MIT](LICENSE), and [NOTICE](NOTICE) for details.

LLNL-CODE-2000523

