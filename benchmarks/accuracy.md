# Accuracy benchmark

## Centralized Comparison Table

All methods evaluated on GSM8K (1 epoch, r=16, a=32, b=2, ga=4):

| Method                 | GSM8k  |
| ---------------------- | ------ |
| lora-bf16              | 0.2335 |
| lora-fp4-cutlass       | 0.2282 |
| qlora-fp4-fp8-cudnn    | 0.2282 |
| lora-fp4-cudnn         | 0.2183 |
| lora-fp8               | 0.2176 |
| qlora-fp8-fp8          | 0.2176 |
| qlora-fp4-fp8-cutlass  | 0.2077 |

**Key Findings:**
- Best overall: **lora-fp8** (0.1873 strict match)
- Best LoRA variant: **lora-fp8** (0.1873 strict match)
- Best QLoRA variant: **qlora-fp8-fp8** and **qlora-fp8-fp16** (0.1683 strict match, tied)
- FP8 methods generally outperform FP4 methods
- CUTLASS implementation slightly outperforms cuDNN for FP4 LoRA

---

## Detailed Results by Method
## lora-bf16
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |
|-----|------:|----------------|-----:|-----------|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.1835|
|     |       |strict-match    |     0|exact_match|↑  |0.2335|

## lora-fp8
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |
|-----|------:|----------------|-----:|-----------|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.1888|
|     |       |strict-match    |     0|exact_match|↑  |0.2176|

## lora-fp4-cudnn
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |
|-----|------:|----------------|-----:|-----------|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.1797|
|     |       |strict-match    |     0|exact_match|↑  |0.2183|

## lora-fp4-cutlass
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |
|-----|------:|----------------|-----:|-----------|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.1941|
|     |       |strict-match    |     0|exact_match|↑  |0.2282|

## qlora-fp8-fp8
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |
|-----|------:|----------------|-----:|-----------|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.1789|
|     |       |strict-match    |     0|exact_match|↑  |0.2176|

## qlora-fp4-fp8-cutlass
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |
|-----|------:|----------------|-----:|-----------|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.1691|
|     |       |strict-match    |     0|exact_match|↑  |0.2077|

## qlora-fp4-fp8-cudnn
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |
|-----|------:|----------------|-----:|-----------|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.1865|
|     |       |strict-match    |     0|exact_match|↑  |0.2282|

# Training config:
1 epoch
r=16,a=32,b=2,ga=4