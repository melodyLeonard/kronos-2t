//! Hardware memory envelope detection and deterministic RAM budgeting for 8 GB MacBooks and mobile devices.

use std::sync::atomic::{AtomicUsize, Ordering};

const GIB: usize = 1024 * 1024 * 1024;
const MIB: usize = 1024 * 1024;

static GLOBAL_BUDGET_OVERRIDE_BYTES: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
thread_local! {
    static TEST_THREAD_BUDGET_OVERRIDE_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryTier {
    /// Smartphone / iOS extended-address-space budget (`<= 5.0 GiB` usable working set).
    Mobile5G,
    /// 8 GB unified-memory MacBook (`<= 5.6 GiB` safe working set, zero swap).
    Edge8G,
    /// 12-16 GB unified-memory laptop or tablet (`<= 12.5 GiB` working set).
    Standard16G,
    /// 24 GB+ desktop or workstation unified memory.
    Workstation24GPlus,
}

impl MemoryTier {
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryTier::Mobile5G => "mobile-5g",
            MemoryTier::Edge8G => "edge-8g",
            MemoryTier::Standard16G => "standard-16g",
            MemoryTier::Workstation24GPlus => "workstation-24g+",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryBudgetProfile {
    pub physical_ram_bytes: usize,
    pub recommended_working_set_bytes: usize,
    pub budget_bytes: usize,
    pub tier: MemoryTier,
    pub prompt_chunk_gpu: usize,
    pub initial_kv_cap_tokens: usize,
    pub quantize_untied_embeddings: bool,
    pub prefer_q4_kv: bool,
}

impl MemoryBudgetProfile {
    /// Installs a process-wide memory budget cap in bytes (`0` clears the override).
    pub fn set_global_budget_override_bytes(bytes: usize) {
        #[cfg(test)]
        {
            TEST_THREAD_BUDGET_OVERRIDE_BYTES.with(|c| c.set(bytes));
        }
        #[cfg(not(test))]
        {
            GLOBAL_BUDGET_OVERRIDE_BYTES.store(bytes, Ordering::SeqCst);
        }
    }

    /// Installs a process-wide memory budget cap in MiB (`0` clears the override).
    pub fn set_global_budget_override_mb(mb: usize) {
        Self::set_global_budget_override_bytes(mb.saturating_mul(MIB));
    }

    /// Builds a profile from an explicit usable byte cap (used by `--memory-budget-mb` and tests).
    pub fn for_budget_bytes(budget_bytes: usize) -> Self {
        let hw_ram = detect_physical_ram_bytes();
        let rec_ws = default_working_set_from_ram(hw_ram);
        Self::from_resolved(hw_ram, rec_ws, budget_bytes.max(512 * MIB))
    }

    /// Resolves the active hardware profile, honoring `set_global_budget_override_*` and
    /// `KRONOS_MEMORY_BUDGET_MB`, plus optional Metal `recommendedMaxWorkingSetSize`.
    pub fn active() -> Self {
        Self::active_with_metal_working_set(None)
    }

    pub fn active_with_metal_working_set(metal_ws_bytes: Option<usize>) -> Self {
        let hw_ram = detect_physical_ram_bytes();
        let rec_ws = metal_ws_bytes
            .filter(|&b| b >= 512 * MIB)
            .unwrap_or_else(|| default_working_set_from_ram(hw_ram));
        let override_b = {
            #[cfg(test)]
            {
                let tl = TEST_THREAD_BUDGET_OVERRIDE_BYTES.with(|c| c.get());
                if tl > 0 {
                    tl
                } else {
                    GLOBAL_BUDGET_OVERRIDE_BYTES.load(Ordering::Relaxed)
                }
            }
            #[cfg(not(test))]
            {
                GLOBAL_BUDGET_OVERRIDE_BYTES.load(Ordering::Relaxed)
            }
        };
        let env_b = std::env::var("KRONOS_MEMORY_BUDGET_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .map(|mb| mb.saturating_mul(MIB))
            .unwrap_or(0);
        let budget = if override_b > 0 {
            override_b
        } else if env_b > 0 {
            env_b
        } else {
            rec_ws
        };
        Self::from_resolved(hw_ram, rec_ws, budget)
    }

    fn from_resolved(
        physical_ram_bytes: usize,
        recommended_working_set_bytes: usize,
        budget_bytes: usize,
    ) -> Self {
        let tier = if budget_bytes < 5 * GIB {
            MemoryTier::Mobile5G
        } else if budget_bytes <= (56 * GIB) / 10 {
            MemoryTier::Edge8G
        } else if budget_bytes <= (125 * GIB) / 10 {
            MemoryTier::Standard16G
        } else {
            MemoryTier::Workstation24GPlus
        };

        let (prompt_chunk_gpu, initial_kv_cap_tokens, quantize_embed_default, prefer_q4_kv) =
            match tier {
                MemoryTier::Mobile5G => (256, 512, true, true),
                MemoryTier::Edge8G => (256, 512, true, true),
                MemoryTier::Standard16G => (512, 1024, false, false),
                MemoryTier::Workstation24GPlus => (1536, 2048, false, false),
            };
        let force_q_embed = std::env::var("KRONOS_QUANTIZE_EMBED")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

        Self {
            physical_ram_bytes,
            recommended_working_set_bytes,
            budget_bytes,
            tier,
            prompt_chunk_gpu,
            initial_kv_cap_tokens,
            quantize_untied_embeddings: quantize_embed_default || force_q_embed,
            prefer_q4_kv,
        }
    }

    /// Computes the SLRU DRAM expert-cache arena budget after reserving pinned attention/embedding
    /// weights, KV + activation scratch, optional draft engine, and OS safety headroom.
    pub fn slru_expert_arena_bytes(
        &self,
        pinned_core_bytes: usize,
        kv_and_scratch_bytes: usize,
        draft_bytes: usize,
    ) -> usize {
        let safety_headroom = match self.tier {
            MemoryTier::Mobile5G => 320 * MIB,
            MemoryTier::Edge8G => 384 * MIB,
            MemoryTier::Standard16G => 640 * MIB,
            MemoryTier::Workstation24GPlus => 1024 * MIB,
        };
        let reserved = pinned_core_bytes
            .saturating_add(kv_and_scratch_bytes)
            .saturating_add(draft_bytes)
            .saturating_add(safety_headroom);
        self.budget_bytes.saturating_sub(reserved).max(64 * MIB)
    }
}

fn default_working_set_from_ram(hw_ram: usize) -> usize {
    #[cfg(target_os = "ios")]
    {
        if hw_ram <= 6 * GIB {
            (hw_ram * 60) / 100
        } else if hw_ram <= 8 * GIB {
            4900 * MIB
        } else {
            (hw_ram * 65) / 100
        }
    }
    #[cfg(not(target_os = "ios"))]
    {
        if hw_ram <= 8 * GIB {
            // 8 GB MacBook Air `recommendedMaxWorkingSetSize` is ~5.33 GiB; cap at 5.0 GiB for zero swap.
            5120 * MIB
        } else if hw_ram <= 16 * GIB {
            (hw_ram * 70) / 100
        } else {
            (hw_ram * 78) / 100
        }
    }
}

/// Queries physical unified RAM in bytes (`hw.memsize` on Apple platforms).
pub fn detect_physical_ram_bytes() -> usize {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const std::ffi::c_char,
                oldp: *mut std::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut std::ffi::c_void,
                newlen: usize,
            ) -> i32;
        }
        let mut mem: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        // SAFETY: Valid NUL-terminated sysctl key and properly sized output `u64` buffer.
        let rc = unsafe {
            sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&mut mem as *mut u64).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && mem > 0 {
            return mem as usize;
        }
    }
    8 * GIB
}

/// Returns current live physical resident set size (RSS) of this process in bytes.
pub fn current_resident_bytes() -> usize {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],
            system_time: [u32; 2],
            policy: i32,
            suspend_count: i32,
        }
        const MACH_TASK_BASIC_INFO: u32 = 20;
        unsafe extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut MachTaskBasicInfo,
                task_info_out_cnt: *mut u32,
            ) -> i32;
        }
        let mut info = MachTaskBasicInfo {
            virtual_size: 0,
            resident_size: 0,
            resident_size_max: 0,
            user_time: [0; 2],
            system_time: [0; 2],
            policy: 0,
            suspend_count: 0,
        };
        let mut count =
            (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;
        // SAFETY: `info` and `count` match `MACH_TASK_BASIC_INFO` layout in Darwin Mach headers.
        let kr = unsafe {
            task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &mut info,
                &mut count,
            )
        };
        if kr == 0 {
            return info.resident_size as usize;
        }
    }
    peak_resident_bytes()
}

/// Returns peak physical resident set size (`ru_maxrss`) of this process in bytes.
pub fn peak_resident_bytes() -> usize {
    #[cfg(unix)]
    {
        #[repr(C)]
        struct Timeval {
            tv_sec: i64,
            tv_usec: i64,
        }
        #[repr(C)]
        struct Rusage {
            ru_utime: Timeval,
            ru_stime: Timeval,
            ru_maxrss: isize,
            _rest: [isize; 14],
        }
        unsafe extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        let mut ru = Rusage {
            ru_utime: Timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            ru_stime: Timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            ru_maxrss: 0,
            _rest: [0; 14],
        };
        // SAFETY: `RUSAGE_SELF == 0` and `Rusage` matches POSIX/Darwin `struct rusage`.
        if unsafe { getrusage(0, &mut ru) } == 0 && ru.ru_maxrss > 0 {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                return ru.ru_maxrss as usize;
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                return (ru.ru_maxrss as usize) * 1024;
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budget_tiers_and_slru_arena_sizing() {
        let hw = detect_physical_ram_bytes();
        assert!(hw >= 4 * GIB);
        let rss = current_resident_bytes();
        let peak = peak_resident_bytes();
        assert!(rss > 0 && peak >= rss / 2);

        let edge = MemoryBudgetProfile::for_budget_bytes(5120 * MIB);
        assert_eq!(edge.tier, MemoryTier::Edge8G);
        assert_eq!(edge.prompt_chunk_gpu, 256);
        assert!(edge.quantize_untied_embeddings);
        assert!(edge.prefer_q4_kv);
        // 1.0 GiB pinned core + 160 MiB KV/scratch + 243 MiB draft + 384 MiB OS headroom -> ~3.23 GiB SLRU arena
        let slru = edge.slru_expert_arena_bytes(1024 * MIB, 160 * MIB, 243 * MIB);
        assert!(
            (3100 * MIB..=3500 * MIB).contains(&slru),
            "unexpected 8GB SLRU arena: {} MiB",
            slru / MIB
        );

        let mobile = MemoryBudgetProfile::for_budget_bytes(4900 * MIB);
        assert_eq!(mobile.tier, MemoryTier::Mobile5G);
        assert_eq!(mobile.prompt_chunk_gpu, 256);
        assert!(mobile.quantize_untied_embeddings);

        let ws = MemoryBudgetProfile::for_budget_bytes(18 * GIB);
        assert_eq!(ws.tier, MemoryTier::Workstation24GPlus);
        assert_eq!(ws.prompt_chunk_gpu, 1536);
        assert!(!ws.quantize_untied_embeddings);
    }
}
