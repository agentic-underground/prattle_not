//! Loading a model must hold its weights in memory ONCE. A plain read leaves every byte it read in the
//! page cache as well as in the merged buffer, and a container's memory limit counts both: a 70B's
//! 141 GB came to ~282 GB against a 256 GB machine, and the rented evaluation stalled at its limit.
//! INTEGRATION: needs a sharded model folder (default models/Qwen3-4B; PRATTLE_CACHE_PROBE overrides).
#![cfg(unix)]

use std::os::unix::io::AsRawFd;

const M: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");

/// The shard files the loader reads.
fn shards(dir: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.ends_with(".safetensors"))
        .map(|n| format!("{dir}/{n}"))
        .collect();
    v.sort();
    v
}

/// Bytes of `path` resident in the page cache, by mincore over a read-only mapping.
fn resident(path: &str) -> u64 {
    let f = std::fs::File::open(path).unwrap();
    let len = f.metadata().unwrap().len() as usize;
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    unsafe {
        let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0);
        assert!(p != libc::MAP_FAILED, "mmap {path}");
        let mut vec = vec![0u8; len.div_ceil(page)];
        assert_eq!(libc::mincore(p, len, vec.as_mut_ptr()), 0, "mincore {path}");
        libc::munmap(p, len);
        vec.iter().filter(|b| **b & 1 == 1).count() as u64 * page as u64
    }
}

/// Ask the kernel to drop `path` from the page cache, so the measurement starts from nothing.
fn evict(path: &str) {
    let f = std::fs::File::open(path).unwrap();
    unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
}

#[test]
#[ignore = "integration"]
fn loading_a_model_leaves_its_shards_out_of_the_page_cache() {
    let dir = std::env::var("PRATTLE_CACHE_PROBE").unwrap_or_else(|_| format!("{M}Qwen3-4B"));
    let files = shards(&dir);
    assert!(!files.is_empty(), "no shards in {dir}");
    files.iter().for_each(|f| evict(f));
    let before: u64 = files.iter().map(|f| resident(f)).sum();
    let weights = prattle_core::weights::read_dir(&dir).unwrap();
    let after: u64 = files.iter().map(|f| resident(f)).sum();
    let total: u64 = files.iter().map(|f| std::fs::metadata(f).unwrap().len()).sum();
    println!("{dir}: {} shards, {:.2} GB; page cache {:.2} GB before the load, {:.2} GB after; buffer {:.2} GB",
        files.len(), total as f64 / 1e9, before as f64 / 1e9, after as f64 / 1e9, weights.len() as f64 / 1e9);
    assert!(after * 20 < total, "the load left {:.2} of {:.2} GB in the page cache: the weights are held twice", after as f64 / 1e9, total as f64 / 1e9);
}
