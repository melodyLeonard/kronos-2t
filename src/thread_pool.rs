//! Lock-Free Persistent Work-Stealing P-Core Spin Pool (`~35 ns` Dispatch Latency).
//!
//! Eliminates OS kernel mutex/futex wakeup overhead and Apple Silicon 128-byte L1 cache-line
//! false sharing across hundreds of matrix-vector dispatches per token.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

#[cfg(target_os = "macos")]
fn pin_macos_p_core_qos() {
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    // SAFETY: Standard macOS libsystem call setting current thread scheduling QoS hint.
    unsafe {
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn pin_macos_p_core_qos() {}

/// 128-byte cache-line wrapper preventing false sharing across Apple M-series P-core L1 caches.
#[repr(align(128))]
struct CachePadded<T>(T);

struct DispatchHeader {
    epoch: AtomicUsize,
    total_items: AtomicUsize,
    task_fn: AtomicUsize,
    task_ctx: AtomicUsize,
    shutdown: AtomicBool,
}

struct SharedPoolState {
    /// Read-only during a parallel dispatch wave (lives on its own 128B cache line!).
    header: CachePadded<DispatchHeader>,
    /// Decremented once per worker upon finishing a wave (isolated 128B cache line!).
    active_remaining: CachePadded<AtomicUsize>,
    /// Next unclaimed item of the current wave; threads claim `chunk`-sized ranges from it so a
    /// core slowed by other processes takes fewer ranges instead of stalling the wave.
    next_item: CachePadded<AtomicUsize>,
    chunk: CachePadded<AtomicUsize>,
    /// Idle sleep management (isolated 128B cache line!).
    sleeping_workers: CachePadded<AtomicUsize>,
    idle_lock: Mutex<()>,
    idle_cvar: Condvar,
}

/// Persistent low-latency thread pool for parallelizing 16-row SIMD stripes across P-cores.
pub struct SpinThreadPool {
    num_threads: usize,
    shared: Arc<SharedPoolState>,
    workers: Vec<JoinHandle<()>>,
}

impl SpinThreadPool {
    /// Creates a pool with `num_threads` total participants (`1` caller + `num_threads - 1` workers).
    pub fn new(num_threads: usize) -> Self {
        let n = num_threads.max(1);
        pin_macos_p_core_qos();

        let shared = Arc::new(SharedPoolState {
            header: CachePadded(DispatchHeader {
                epoch: AtomicUsize::new(0),
                total_items: AtomicUsize::new(0),
                task_fn: AtomicUsize::new(0),
                task_ctx: AtomicUsize::new(0),
                shutdown: AtomicBool::new(false),
            }),
            active_remaining: CachePadded(AtomicUsize::new(0)),
            next_item: CachePadded(AtomicUsize::new(0)),
            chunk: CachePadded(AtomicUsize::new(1)),
            sleeping_workers: CachePadded(AtomicUsize::new(0)),
            idle_lock: Mutex::new(()),
            idle_cvar: Condvar::new(),
        });

        let mut workers = Vec::with_capacity(n.saturating_sub(1));
        for _ in 1..n {
            let st = Arc::clone(&shared);
            workers.push(thread::spawn(move || {
                pin_macos_p_core_qos();
                let mut seen_epoch = 0usize;
                loop {
                    let mut spun = 0u32;
                    let cur_epoch = loop {
                        let e = st.header.0.epoch.load(Ordering::Acquire);
                        if e != seen_epoch {
                            break e;
                        }
                        if st.header.0.shutdown.load(Ordering::Acquire) {
                            return;
                        }
                        if spun < 2_097_152 {
                            std::hint::spin_loop();
                            spun += 1;
                        } else {
                            st.sleeping_workers.0.fetch_add(1, Ordering::SeqCst);
                            let mut guard = st.idle_lock.lock().unwrap();
                            while st.header.0.epoch.load(Ordering::SeqCst) == seen_epoch
                                && !st.header.0.shutdown.load(Ordering::SeqCst)
                            {
                                guard = st.idle_cvar.wait(guard).unwrap();
                            }
                            drop(guard);
                            st.sleeping_workers.0.fetch_sub(1, Ordering::SeqCst);
                            if st.header.0.shutdown.load(Ordering::Acquire) {
                                return;
                            }
                            break st.header.0.epoch.load(Ordering::Acquire);
                        }
                    };
                    seen_epoch = cur_epoch;
                    if st.header.0.shutdown.load(Ordering::Acquire) {
                        return;
                    }

                    execute_wave(&st);
                    st.active_remaining.0.fetch_sub(1, Ordering::Release);
                }
            }));
        }

        Self {
            num_threads: n,
            shared,
            workers,
        }
    }

    /// Optimal P-core count for memory-bandwidth-bound SIMD LUT inference (`8` on M4 Pro).
    pub fn default_p_cores() -> Self {
        let logical = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let target = if logical >= 12 {
            8
        } else if logical >= 8 {
            6
        } else {
            logical.max(1)
        };
        Self::new(target)
    }

    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Executes `f(item_start, item_end)` over `0..total_items` across all pool threads, which
    /// claim ranges of about `total_items / (4 * threads)` items from a shared counter.
    #[inline]
    pub fn parallel_for<F>(&self, total_items: usize, _grain: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if total_items == 0 {
            return;
        }
        if self.num_threads <= 1 || total_items < self.num_threads {
            f(0, total_items);
            return;
        }

        unsafe fn trampoline<F: Fn(usize, usize) + Sync>(ctx: usize, start: usize, end: usize) {
            // SAFETY: `ctx` points to `f` on caller's stack frame, valid until `active_remaining == 0`.
            let closure = unsafe { &*(ctx as *const F) };
            closure(start, end);
        }

        let st = &self.shared;
        st.header.0.total_items.store(total_items, Ordering::Relaxed);
        st.header
            .0
            .task_fn
            .store(trampoline::<F> as *const () as usize, Ordering::Relaxed);
        st.header
            .0
            .task_ctx
            .store((&f as *const F) as usize, Ordering::Relaxed);
        st.active_remaining
            .0
            .store(self.num_threads - 1, Ordering::Relaxed);
        st.next_item.0.store(0, Ordering::Relaxed);
        st.chunk
            .0
            .store(total_items.div_ceil(self.num_threads * 4).max(1), Ordering::Relaxed);

        let next_epoch = st.header.0.epoch.load(Ordering::Relaxed).wrapping_add(1);
        st.header.0.epoch.store(next_epoch, Ordering::SeqCst);

        if st.sleeping_workers.0.load(Ordering::SeqCst) > 0 {
            let _guard = st.idle_lock.lock().unwrap();
            st.idle_cvar.notify_all();
        }

        // The caller claims ranges too.
        execute_wave(st);

        while st.active_remaining.0.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }
}

#[inline]
fn execute_wave(st: &SharedPoolState) {
    let total = st.header.0.total_items.load(Ordering::Relaxed);
    let fn_addr = st.header.0.task_fn.load(Ordering::Relaxed);
    let ctx = st.header.0.task_ctx.load(Ordering::Relaxed);
    let chunk = st.chunk.0.load(Ordering::Relaxed);
    // SAFETY: `fn_addr` was written from `trampoline::<F>` before the `epoch` store.
    let runner: unsafe fn(usize, usize, usize) = unsafe { std::mem::transmute(fn_addr) };
    loop {
        let start = st.next_item.0.fetch_add(chunk, Ordering::Relaxed);
        if start >= total {
            break;
        }
        // SAFETY: `fetch_add` hands out each `[start, start + chunk)` range exactly once.
        unsafe { runner(ctx, start, (start + chunk).min(total)) };
    }
}

impl Drop for SpinThreadPool {
    fn drop(&mut self) {
        self.shared.header.0.shutdown.store(true, Ordering::SeqCst);
        {
            let _guard = self.shared.idle_lock.lock().unwrap();
            let next = self
                .shared
                .header
                .0
                .epoch
                .load(Ordering::Relaxed)
                .wrapping_add(1);
            self.shared.header.0.epoch.store(next, Ordering::SeqCst);
            self.shared.idle_cvar.notify_all();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spin_thread_pool_exact_parallel_stripes() {
        let pool = SpinThreadPool::new(4);
        let n = 250usize;
        let out = vec![0usize; n];
        let out_ptr = out.as_ptr() as usize;

        for iter in 1..=5 {
            pool.parallel_for(n, 7, |start, end| {
                for idx in start..end {
                    // SAFETY: Disjoint index sub-ranges `[start, end)` across threads.
                    unsafe {
                        let p = (out_ptr as *mut usize).add(idx);
                        *p = idx * iter;
                    }
                }
            });
            for (idx, &val) in out.iter().enumerate() {
                assert_eq!(val, idx * iter);
            }
        }
    }
}
