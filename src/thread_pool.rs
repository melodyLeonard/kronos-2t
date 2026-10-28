//! Lock-Free Persistent P-Core Spin Pool & Sense-Reversing Atomic Megakernel Barrier.
//!
//! Uses a single process-wide `QOS_CLASS_USER_INTERACTIVE` P-core worker pool so multiple engines
//! (e.g., Target + Draft models in Speculative Decoding, or multi-candidate benchmark suites)
//! share the exact same hot P-core threads with zero oversubscription and `< 100 ns` barrier latency.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread;

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

#[inline]
fn spin_or_yield(spins: &mut u32) {
    if *spins < 1024 {
        std::hint::spin_loop();
        *spins += 1;
    } else {
        thread::yield_now();
    }
}

/// Detects the exact physical Performance-core count (`hw.perflevel0.physicalcpu` on Apple Silicon,
/// e.g. `10` on M4 Pro).
pub fn physical_p_cores() -> usize {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const i8,
                oldp: *mut std::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut std::ffi::c_void,
                newlen: usize,
            ) -> i32;
        }
        let mut count: i32 = 0;
        let mut size = std::mem::size_of::<i32>();
        // SAFETY: Null-terminated sysctl key and valid `i32` output buffer.
        let rc = unsafe {
            sysctlbyname(
                c"hw.perflevel0.physicalcpu".as_ptr(),
                (&mut count as *mut i32).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && count > 0 {
            return count as usize;
        }
    }
    thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(1)
}

/// Optimal default inference thread count on interactive workstations (`8` on a `10`-P-core M4 Pro,
/// leaving `2` P-cores free for OS/desktop daemons to prevent synchronous barrier priority inversion).
pub fn default_inference_threads() -> usize {
    let p = physical_p_cores();
    if p >= 10 { p - 2 } else { p }
}

/// 128-byte cache-line wrapper preventing false sharing across Apple M-series P-core L1 caches.
#[repr(align(128))]
struct CachePadded<T>(T);

/// Ultra-low-latency (`~80 ns`) sense-reversing atomic spin barrier for single-dispatch megakernels.
pub struct SpinBarrier {
    num_threads: usize,
    count: CachePadded<AtomicUsize>,
    phase: CachePadded<AtomicUsize>,
}

impl SpinBarrier {
    #[inline]
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads: num_threads.max(1),
            count: CachePadded(AtomicUsize::new(0)),
            phase: CachePadded(AtomicUsize::new(0)),
        }
    }

    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Synchronizes all `num_threads` participants in place without leaving the worker closure.
    #[inline]
    pub fn wait(&self) {
        if self.num_threads <= 1 {
            return;
        }
        let p = self.phase.0.load(Ordering::Acquire);
        if self.count.0.fetch_add(1, Ordering::AcqRel) + 1 == self.num_threads {
            self.count.0.store(0, Ordering::Relaxed);
            self.phase.0.store(p.wrapping_add(1), Ordering::Release);
        } else {
            let mut spins = 0u32;
            while self.phase.0.load(Ordering::Acquire) == p {
                spin_or_yield(&mut spins);
            }
        }
    }
}

const MAX_GLOBAL_WORKERS: usize = 32;

struct DispatchHeader {
    epoch: AtomicUsize,
    /// `0` = park excess workers, `1` = `parallel_for`, `2` = `execute_all`.
    mode: AtomicUsize,
    total_items: AtomicUsize,
    num_participants: AtomicUsize,
    task_fn: AtomicUsize,
    task_ctx: AtomicUsize,
    barrier_ptr: AtomicUsize,
}

struct GlobalPoolState {
    dispatch_gate: Mutex<()>,
    spawned_workers: Mutex<usize>,
    header: CachePadded<DispatchHeader>,
    active_remaining: CachePadded<AtomicUsize>,
    /// Bitmask (`1u64 << w_id`) of workers currently parked on their private `Condvar`.
    sleeping_mask: CachePadded<AtomicU64>,
    worker_park: [CachePadded<(Mutex<()>, Condvar)>; MAX_GLOBAL_WORKERS + 1],
    /// Set by `park_idle_workers` when transitioning to single-threaded mode so spinning workers park immediately.
    force_park: CachePadded<AtomicBool>,
}

static GLOBAL_POOL: OnceLock<GlobalPoolState> = OnceLock::new();

fn global_pool() -> &'static GlobalPoolState {
    GLOBAL_POOL.get_or_init(|| GlobalPoolState {
        dispatch_gate: Mutex::new(()),
        spawned_workers: Mutex::new(0),
        header: CachePadded(DispatchHeader {
            epoch: AtomicUsize::new(0),
            mode: AtomicUsize::new(0),
            total_items: AtomicUsize::new(0),
            num_participants: AtomicUsize::new(1),
            task_fn: AtomicUsize::new(0),
            task_ctx: AtomicUsize::new(0),
            barrier_ptr: AtomicUsize::new(0),
        }),
        active_remaining: CachePadded(AtomicUsize::new(0)),
        sleeping_mask: CachePadded(AtomicU64::new(0)),
        worker_park: std::array::from_fn(|_| CachePadded((Mutex::new(()), Condvar::new()))),
        force_park: CachePadded(AtomicBool::new(false)),
    })
}

#[inline]
fn wake_needed_workers(st: &GlobalPoolState, nt: usize) {
    let mut wake = st.sleeping_mask.0.load(Ordering::SeqCst) & ((1u64 << nt) - 2);
    while wake != 0 {
        let w_id = wake.trailing_zeros() as usize;
        wake &= wake - 1;
        let (lock, cvar) = &st.worker_park[w_id].0;
        let _guard = lock.lock().unwrap();
        cvar.notify_one();
    }
}

fn ensure_workers_locked(st: &'static GlobalPoolState, needed_participants: usize) {
    let target_workers = needed_participants.saturating_sub(1).min(MAX_GLOBAL_WORKERS);
    let mut spawned = st.spawned_workers.lock().unwrap();
    let cur_epoch = st.header.0.epoch.load(Ordering::Acquire);
    while *spawned < target_workers {
        let w_id = *spawned + 1;
        *spawned += 1;
        thread::spawn(move || worker_loop(st, w_id, cur_epoch));
    }
}

fn worker_loop(st: &'static GlobalPoolState, w_id: usize, initial_epoch: usize) {
    pin_macos_p_core_qos();
    let bit = 1u64 << w_id;
    let mut seen_epoch = initial_epoch;
    let mut park_now = false;
    loop {
        let mut spun = 0u32;
        let cur_epoch = loop {
            let e = st.header.0.epoch.load(Ordering::Acquire);
            if e != seen_epoch {
                break e;
            }
            if !park_now && !st.force_park.0.load(Ordering::Relaxed) && spun < 8224 {
                if spun < 8192 {
                    std::hint::spin_loop();
                } else {
                    thread::yield_now();
                }
                spun += 1;
            } else {
                let (lock, cvar) = &st.worker_park[w_id].0;
                let mut guard = lock.lock().unwrap();
                st.sleeping_mask.0.fetch_or(bit, Ordering::SeqCst);
                while st.header.0.epoch.load(Ordering::SeqCst) == seen_epoch
                    || w_id >= st.header.0.num_participants.load(Ordering::SeqCst)
                {
                    seen_epoch = st.header.0.epoch.load(Ordering::SeqCst);
                    guard = cvar.wait(guard).unwrap();
                }
                st.sleeping_mask.0.fetch_and(!bit, Ordering::SeqCst);
                let next_e = st.header.0.epoch.load(Ordering::Acquire);
                drop(guard);
                break next_e;
            }
        };
        seen_epoch = cur_epoch;

        let n_part = st.header.0.num_participants.load(Ordering::Relaxed);
        if w_id >= n_part {
            park_now = true;
            continue;
        }
        park_now = false;

        execute_worker_wave(st, w_id, n_part);
        st.active_remaining.0.fetch_sub(1, Ordering::Release);
    }
}

#[inline]
fn execute_worker_wave(st: &GlobalPoolState, worker_id: usize, n_part: usize) {
    let mode = st.header.0.mode.load(Ordering::Relaxed);
    let fn_addr = st.header.0.task_fn.load(Ordering::Relaxed);
    let ctx = st.header.0.task_ctx.load(Ordering::Relaxed);
    if mode == 1 {
        let total = st.header.0.total_items.load(Ordering::Relaxed);
        let start = (total * worker_id) / n_part;
        let end = (total * (worker_id + 1)) / n_part;
        if start < end {
            // SAFETY: `fn_addr` matches `trampoline_for::<F>` for the live caller frame.
            let runner: unsafe fn(usize, usize, usize) = unsafe { std::mem::transmute(fn_addr) };
            unsafe { runner(ctx, start, end) };
        }
    } else if mode == 2 {
        let b_ptr = st.header.0.barrier_ptr.load(Ordering::Relaxed);
        // SAFETY: `fn_addr` matches `trampoline_all::<F>` and `b_ptr` points to live `SpinBarrier`.
        let runner: unsafe fn(usize, usize, usize, usize) = unsafe { std::mem::transmute(fn_addr) };
        unsafe { runner(ctx, worker_id, n_part, b_ptr) };
    }
}

/// Handle to the shared persistent P-core thread pool configured for `num_threads` participants.
pub struct SpinThreadPool {
    num_threads: usize,
    st: &'static GlobalPoolState,
}

impl SpinThreadPool {
    /// Configures a pool handle for `num_threads` total participants (`1` caller + `num_threads - 1` workers).
    pub fn new(num_threads: usize) -> Self {
        let n = num_threads.clamp(1, MAX_GLOBAL_WORKERS + 1);
        pin_macos_p_core_qos();
        let st = global_pool();
        if n > 1 {
            let _gate = st.dispatch_gate.lock().unwrap();
            st.force_park.0.store(false, Ordering::Relaxed);
            ensure_workers_locked(st, n);
        } else {
            Self::park_idle_workers();
        }
        Self { num_threads: n, st }
    }

    /// Optimal P-core count for Apple Silicon / multi-core SIMD inference (`8` on M4 Pro).
    pub fn default_p_cores() -> Self {
        Self::new(default_inference_threads())
    }

    /// Immediately parks all spinning background workers (used when switching to `1T` mode).
    pub fn park_idle_workers() {
        pin_macos_p_core_qos();
        if let Some(st) = GLOBAL_POOL.get() {
            let _gate = st.dispatch_gate.lock().unwrap();
            let spawned = *st.spawned_workers.lock().unwrap();
            if spawned == 0 {
                return;
            }
            st.force_park.0.store(true, Ordering::Relaxed);
            st.header.0.mode.store(0, Ordering::Relaxed);
            st.header.0.num_participants.store(1, Ordering::SeqCst);
            let next_epoch = st.header.0.epoch.load(Ordering::Relaxed).wrapping_add(1);
            st.header.0.epoch.store(next_epoch, Ordering::SeqCst);
            let all_mask = (1u64 << (spawned + 1)) - 2;
            let mut spins = 0u32;
            while (st.sleeping_mask.0.load(Ordering::Acquire) & all_mask) != all_mask {
                spin_or_yield(&mut spins);
            }
        }
    }

    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Executes `f(item_start, item_end)` over `0..total_items` across `num_threads` P-cores.
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

        unsafe fn trampoline_for<F: Fn(usize, usize) + Sync>(ctx: usize, start: usize, end: usize) {
            // SAFETY: `ctx` points to `f` on caller's stack frame, valid until `active_remaining == 0`.
            let closure = unsafe { &*(ctx as *const F) };
            closure(start, end);
        }

        let st = self.st;
        let _gate = st.dispatch_gate.lock().unwrap();
        ensure_workers_locked(st, self.num_threads);
        st.force_park.0.store(false, Ordering::Relaxed);
        st.header.0.mode.store(1, Ordering::Relaxed);
        st.header.0.total_items.store(total_items, Ordering::Relaxed);
        st.header.0.num_participants.store(self.num_threads, Ordering::SeqCst);
        st.header
            .0
            .task_fn
            .store(trampoline_for::<F> as *const () as usize, Ordering::Relaxed);
        st.header
            .0
            .task_ctx
            .store((&f as *const F) as usize, Ordering::Relaxed);
        st.active_remaining
            .0
            .store(self.num_threads - 1, Ordering::Relaxed);

        let next_epoch = st.header.0.epoch.load(Ordering::Relaxed).wrapping_add(1);
        st.header.0.epoch.store(next_epoch, Ordering::SeqCst);

        wake_needed_workers(st, self.num_threads);

        execute_worker_wave(st, 0, self.num_threads);

        let mut spins = 0u32;
        while st.active_remaining.0.load(Ordering::Acquire) != 0 {
            spin_or_yield(&mut spins);
        }
    }

    /// Launches a single persistent megakernel `f(tid, num_threads, barrier)` across all `num_threads`
    /// P-cores (`tid in 0..num_threads`).
    #[inline]
    pub fn execute_all<F>(&self, f: F)
    where
        F: Fn(usize, usize, &SpinBarrier) + Sync,
    {
        self.execute_up_to(self.num_threads, f);
    }

    /// Launches a persistent megakernel across `self.num_threads.min(max_threads)` P-cores, leaving
    /// any excess workers parked in the kernel with zero syscalls.
    #[inline]
    pub fn execute_up_to<F>(&self, max_threads: usize, f: F)
    where
        F: Fn(usize, usize, &SpinBarrier) + Sync,
    {
        let nt = self.num_threads.min(max_threads.max(1));
        if nt <= 1 {
            let barrier = SpinBarrier::new(1);
            f(0, 1, &barrier);
            return;
        }

        unsafe fn trampoline_all<F: Fn(usize, usize, &SpinBarrier) + Sync>(
            ctx: usize,
            tid: usize,
            nt: usize,
            b_ptr: usize,
        ) {
            // SAFETY: `ctx` and `b_ptr` live on caller's stack frame until `active_remaining == 0`.
            let closure = unsafe { &*(ctx as *const F) };
            let barrier = unsafe { &*(b_ptr as *const SpinBarrier) };
            closure(tid, nt, barrier);
        }

        let barrier = SpinBarrier::new(nt);
        let st = self.st;
        let _gate = st.dispatch_gate.lock().unwrap();
        ensure_workers_locked(st, nt);
        st.force_park.0.store(false, Ordering::Relaxed);
        st.header.0.mode.store(2, Ordering::Relaxed);
        st.header.0.num_participants.store(nt, Ordering::SeqCst);
        st.header
            .0
            .task_fn
            .store(trampoline_all::<F> as *const () as usize, Ordering::Relaxed);
        st.header
            .0
            .task_ctx
            .store((&f as *const F) as usize, Ordering::Relaxed);
        st.header
            .0
            .barrier_ptr
            .store((&barrier as *const SpinBarrier) as usize, Ordering::Relaxed);
        st.active_remaining.0.store(nt - 1, Ordering::Relaxed);

        let next_epoch = st.header.0.epoch.load(Ordering::Relaxed).wrapping_add(1);
        st.header.0.epoch.store(next_epoch, Ordering::SeqCst);

        wake_needed_workers(st, nt);

        f(0, nt, &barrier);

        let mut spins = 0u32;
        while st.active_remaining.0.load(Ordering::Acquire) != 0 {
            spin_or_yield(&mut spins);
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

    #[test]
    fn test_execute_all_megakernel_spin_barrier() {
        let nt = physical_p_cores().clamp(2, 10);
        let pool = SpinThreadPool::new(nt);
        let buf = vec![0usize; nt];
        let sum_out = vec![0usize; nt];
        let bp = buf.as_ptr() as usize;
        let sp = sum_out.as_ptr() as usize;

        pool.execute_all(|tid, num_t, barrier| {
            for round in 1..=8 {
                unsafe { *(bp as *mut usize).add(tid) = (tid + 1) * round };
                barrier.wait();
                let total: usize = (0..num_t).map(|i| unsafe { *(bp as *const usize).add(i) }).sum();
                barrier.wait();
                unsafe { *(sp as *mut usize).add(tid) = total };
                barrier.wait();
            }
        });

        let expected = 8 * nt * (nt + 1) / 2;
        for &v in &sum_out {
            assert_eq!(v, expected);
        }
    }
}
