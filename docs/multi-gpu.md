# Multi-GPU Training

Surogate provides robust support for multi-GPU training, enabling efficient utilization of multiple GPUs to accelerate model training and handle larger models and datasets. The framework leverages data parallelism and model parallelism techniques to distribute the workload across available GPUs.

## Data Parallelism

In data parallelism, Surogate replicates the model on each GPU and splits the input data across the GPUs. Each GPU processes its portion of the data independently, computes gradients, and then synchronizes the gradients across all GPUs to update the model parameters. This approach is particularly effective for large datasets where the same model can be applied to different data samples in parallel.

## Model Parallelism

For very large models that cannot fit into a single GPU's memory, Surogate supports model parallelism. In this approach, different parts of the model are distributed across multiple GPUs. During training, data flows through the model segments on different GPUs, allowing the training of models that exceed the memory capacity of a single GPU.

## Configuration Parameters

### `gpus`

Specifies the number of GPUs to use for training.

- **Default**: `1` (uses the first available GPU)
- **Special value**: `0` uses all available GPUs

Example:

```yaml
gpus: 4 # Use 4 GPUs
```

The effective batch size scales with the number of GPUs:

```
Effective batch size = per_device_batch_size × gradient_accumulation_steps × gpus
```

### `zero_level`

Controls the ZeRO (Zero Redundancy Optimizer) optimization level, which determines how optimizer states, gradients, and weights are partitioned across GPUs to reduce memory consumption.

- **Default**: `1`
- **CLI flag**: `--zero-level`

| Level | Description              | What's Sharded                               |
| ----- | ------------------------ | -------------------------------------------- |
| 1     | Sharded optimizer states | Optimizer states (momentum, variance)        |
| 2     | Sharded gradients        | Optimizer states + gradients                 |
| 3     | Sharded weights          | Optimizer states + gradients + model weights |

Example:

```yaml
zero_level: 2 # Shard optimizer states and gradients
```

Higher ZeRO levels reduce per-GPU memory consumption but increase communication overhead. ZeRO-3 enables training models that don't fit on a single GPU.

### `shard_weights`

Enables sharding of model weights across data-parallel processes. This is automatically enabled when `zero_level: 3` is set, but can also be configured independently.

- **Default**: `false`

When enabled:

- Model weights are partitioned across GPUs
- Each GPU only stores a fraction of the weights
- All-gather operations reconstruct full weights when needed
- Enables more effective use of CPU offloading
- Reduces per-GPU memory consumption

Example:

```yaml
shard_weights: true
```

Note: When training with FP8, it may be beneficial to enable weight sharding before gradient sharding, as weights require only half the bandwidth compared to gradients.

### `shard_gradients`

Enables sharding of gradients across data-parallel processes. This is automatically enabled when `zero_level >= 2` is set, but can also be configured independently.

- **Default**: `false`

When enabled:

- Gradients are partitioned across GPUs via reduce-scatter operations
- Each GPU only stores and updates its shard of gradients
- Enables more effective use of CPU offloading
- Reduces per-GPU memory consumption

Example:

```yaml
shard_gradients: true
```

## Advanced Options

For fine-grained control over multi-GPU communication and memory:

- `memcpy_all_gather`: Use memcpy for all-gather operations (threads backend only). Generally achieves better PCIe bandwidth utilization.
- `memcpy_send_recv`: Use memcpy for send/receive operations (threads backend only).
- `use_all_to_all_reduce`: Use all-to-all-based reduce algorithm for potentially better performance.
