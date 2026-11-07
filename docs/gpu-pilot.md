# GPU environment and timing pilot

The measured pilot below predates the switch from MNLI to RTE. Its throughput is used only as a rough RTE runtime proxy; no RTE training timing pilot has been run.

## Environment

- Slurm partition: `MGPU-TC2`; account QoS allows one GPU and a 6-hour maximum per job.
- Pilot node: `TC2N02`, NVIDIA A40 with 46,068 MiB, driver 570.148.08.
- Python: 3.12.10.
- PyTorch: 2.11.0+cu128 (CUDA runtime 12.8).
- Transformers: 5.22.1; Datasets: 5.1.0.
- Environment: Python 3.12 virtual environment with the CUDA-enabled packages above.
  For another user, activate their environment or set `REGBENCH_PYTHON` to its Python executable.

Run `sbatch scripts/gpu_pilot.slurm` from the project root. The pilot runs a bounded smoke/timing check, not the experiment sweep. Logs are written to `outputs/slurm/`.

## Measured MNLI pilot

Successful Slurm job `51658` trained `microsoft/deberta-v3-base` for one epoch on 512 examples (batch 16), then evaluated 256-example validation and test smoke samples.

| Measurement | Result |
|---|---:|
| Training duration | 4.257 s |
| Training throughput | 120.27 examples/s |
| End-to-end duration (model load, training, validation and test smoke) | 13.92 s |

These are MNLI pilot measurements. The smoke metrics are not benchmark results, and throughput is only an approximate basis for RTE estimates.

The successful metrics and predictions are in `outputs/pilot/gpu-baseline-fp32master-bf16/`. The captured Slurm output is `outputs/slurm/regbench-pilot-51658.out`.

## Numerical stability fix

The pretrained checkpoint loads its parameters as FP16. The first run exposed non-finite parameters after an AdamW step when optimizing those half-precision parameters directly. The trainer now promotes model parameters to FP32 before creating AdamW optimizer state; BF16 autocast remains enabled on supported GPUs. The successful pilot used this path and has finite training, validation and smoke-test losses. The trainer also fails explicitly on non-finite loss, gradients or evaluation logits.
