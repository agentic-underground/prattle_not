//! Where the host tools run: `PRATTLE_DEVICES` names the source's device and the target's, as
//! `cpu` or `cuda:<n>`, comma-separated — `cuda:0,cuda:1` puts the two models on two GPUs. One name
//! puts both on it; unset is the CPU. A GPU needs the crate built with the `cuda` feature, and asking
//! for one without it is an error, never a silent fall back to the CPU.

use candle_core::Device;

/// One device from its name.
pub fn named(name: &str) -> Result<Device, String> {
    match name.trim() {
        "cpu" => Ok(Device::Cpu),
        n => match n.strip_prefix("cuda:").and_then(|i| i.parse::<usize>().ok()) {
            // A stream of the device's own: `Device::new_cuda` hands every GPU the per-thread default
            // stream, a handle resolved against whichever context is current on the thread, so two
            // GPUs driven from one thread can queue work on each other's stream.
            Some(i) => Device::new_cuda_with_stream(i).map_err(|e| format!("{n}: {e} (build with --features cuda)")),
            None => Err(format!("{n}: a device is cpu or cuda:<n>")),
        },
    }
}

/// The source's device and the target's, from `PRATTLE_DEVICES`.
pub fn pair() -> Result<(Device, Device), String> {
    let spec = std::env::var("PRATTLE_DEVICES").unwrap_or_else(|_| "cpu".into());
    let names: Vec<&str> = spec.split(',').collect();
    let source = named(names[0])?;
    let target = match names.get(1) {
        Some(n) => named(n.split('+').next().unwrap_or(n))?,
        None => source.clone(),
    };
    Ok((source, target))
}

/// The target's devices: its name in `PRATTLE_DEVICES` split on `+`. `cuda:0,cuda:0+cuda:1` puts the
/// source on cuda:0 and the target's layers across both cards (`spread`); one name is one card. The
/// first is `pair`'s target, where the target's input and the bridge live.
pub fn target_devices() -> Result<Vec<Device>, String> {
    let spec = std::env::var("PRATTLE_DEVICES").unwrap_or_else(|_| "cpu".into());
    let names: Vec<&str> = spec.split(',').collect();
    let first = pair()?.1;
    match names.get(1) {
        Some(n) if n.contains('+') => {
            let rest = n.split('+').skip(1).map(named).collect::<Result<Vec<_>, _>>()?;
            Ok(std::iter::once(first).chain(rest).collect())
        }
        _ => Ok(vec![first]),
    }
}

/// How many host threads a tool's own worker pool may run: `RAYON_NUM_THREADS` when set — the same cap
/// candle's kernels obey, so a shared machine can hold every CPU job to it — else every core, at most 16.
pub fn cpu_workers() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
    std::env::var("RAYON_NUM_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0).map_or(cores, |n| n.min(cores))
}

/// Which of `devices` devices each of `layers` layers sits on, by index: contiguous runs, as even as the
/// count allows, the earlier devices taking one more where it does not divide.
pub fn spread(layers: usize, devices: usize) -> Vec<usize> {
    let devices = devices.max(1);
    let (each, extra) = (layers / devices, layers % devices);
    (0..devices).flat_map(|d| std::iter::repeat_n(d, each + usize::from(d < extra))).collect()
}

#[cfg(test)]
mod tests {
    use super::spread;

    #[test]
    fn layers_spread_across_devices_in_even_contiguous_runs() {
        assert_eq!(spread(64, 2), [vec![0; 32], vec![1; 32]].concat(), "a 64-layer model halves across two cards");
        assert_eq!(spread(5, 2), vec![0, 0, 0, 1, 1], "the first card takes the remainder");
        assert_eq!(spread(3, 1), vec![0, 0, 0], "one card holds every layer");
        assert_eq!(spread(80, 3).iter().filter(|&&d| d == 2).count(), 26, "three cards: 27, 27, 26");
    }

    /// A tool's own pool obeys RAYON_NUM_THREADS, the cap a shared machine sets every CPU job.
    #[test]
    fn a_tools_worker_pool_obeys_rayon_num_threads() {
        // SAFETY: nextest runs each test in its own process, so no other thread reads the environment
        unsafe { std::env::set_var("RAYON_NUM_THREADS", "3") };
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
        assert_eq!(super::cpu_workers(), cores.min(3), "the pool ignored RAYON_NUM_THREADS");
    }
}

/// The precision the TARGET's weights are held in, from `PRATTLE_TARGET_DTYPE` — `bf16` or `f16` where
/// a target does not fit its card in f32 (`crate::half`); unset is f32.
/// `q8` and `q4k` hold the linear layers quantized (`target_quant`) and everything else in bf16.
pub fn target_dtype() -> Result<candle_core::DType, String> {
    match std::env::var("PRATTLE_TARGET_DTYPE").as_deref() {
        Ok("q8") | Ok("q4k") => Ok(candle_core::DType::BF16),
        _ => dtype("PRATTLE_TARGET_DTYPE"),
    }
}

/// The quantization of the TARGET's linear layers, from `PRATTLE_TARGET_DTYPE` — `q8` (Q8_0) or `q4k`
/// (Q4K), where even bf16 does not fit its card (a 32B is 64 GB in bf16); otherwise none.
pub fn target_quant() -> Result<Option<candle_core::quantized::GgmlDType>, String> {
    match std::env::var("PRATTLE_TARGET_DTYPE").as_deref() {
        Ok("q8") => Ok(Some(candle_core::quantized::GgmlDType::Q8_0)),
        Ok("q4k") => Ok(Some(candle_core::quantized::GgmlDType::Q4K)),
        _ => Ok(None),
    }
}

/// The SOURCE's, from `PRATTLE_SOURCE_DTYPE`: a bf16 target beside an f32 source reads faster than its
/// source does on a GPU's tensor cores, so a pair is timed fairly only with both at one precision.
pub fn source_dtype() -> Result<candle_core::DType, String> {
    dtype("PRATTLE_SOURCE_DTYPE")
}

fn dtype(var: &str) -> Result<candle_core::DType, String> {
    match std::env::var(var).as_deref() {
        Err(_) | Ok("f32") => Ok(candle_core::DType::F32),
        Ok("bf16") => Ok(candle_core::DType::BF16),
        Ok("f16") => Ok(candle_core::DType::F16),
        Ok(other) => Err(format!("{var}={other}: one of f32, bf16, f16")),
    }
}

/// `t` on `device`. Every move between devices goes through here, and between two GPUs it goes
/// THROUGH THE HOST: candle's direct GPU-to-GPU copy filled a two-GPU fit with NaN while the same
/// fit on one GPU was clean, and neither a contiguous copy nor synchronizing both streams cured it.
/// The rows a fit moves are a few hundred KB a question. Pinned by the bridge tests'
/// `gram_on_a_gpu_sums_rows_from_the_other_gpu`.
pub fn moved(t: &candle_core::Tensor, device: &Device) -> candle_core::Result<candle_core::Tensor> {
    if t.device().same_device(device) {
        return Ok(t.clone());
    }
    match (t.device(), device) {
        (Device::Cuda(_), Device::Cuda(_)) => t.to_device(&Device::Cpu)?.to_device(device),
        _ => t.to_device(device),
    }
}
