//! IEEE half <-> single conversion.

/// IEEE half -> single conversion (NEON `fcvtl` for full blocks of 4).
pub fn f16_to_f32(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    while i + 4 <= src.len() {
        // SAFETY: 4 halves in, 4 floats out.
        unsafe {
            let h = std::arch::aarch64::vld1_u16(src.as_ptr().add(i));
            let f: std::arch::aarch64::float32x4_t;
            core::arch::asm!("fcvtl {o:v}.4s, {h:v}.4h", o = out(vreg) f, h = in(vreg) h, options(pure, nomem, nostack));
            std::arch::aarch64::vst1q_f32(dst.as_mut_ptr().add(i), f);
        }
        i += 4;
    }
    for (d, &h) in dst[i..].iter_mut().zip(&src[i..]) {
        let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
        let bits = match exp {
            0 if man == 0 => sign << 31,
            0 => {
                let lz = man.leading_zeros() - 21;
                (sign << 31) | ((113 - lz) << 23) | (((man << lz) & 0x3ff) << 13)
            }
            31 => (sign << 31) | (0xff << 23) | (man << 13),
            e => (sign << 31) | ((e + 112) << 23) | (man << 13),
        };
        *d = f32::from_bits(bits);
    }
}

/// IEEE single -> half conversion (round to nearest even, NEON `fcvtn`).
pub fn f32_to_f16(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    while i + 4 <= src.len() {
        // SAFETY: 4 floats in, 4 halves out.
        unsafe {
            let f = std::arch::aarch64::vld1q_f32(src.as_ptr().add(i));
            let h: std::arch::aarch64::uint16x4_t;
            core::arch::asm!("fcvtn {h:v}.4h, {f:v}.4s", h = out(vreg) h, f = in(vreg) f, options(pure, nomem, nostack));
            std::arch::aarch64::vst1_u16(dst.as_mut_ptr().add(i), h);
        }
        i += 4;
    }
    for (d, &x) in dst[i..].iter_mut().zip(&src[i..]) {
        let b = x.to_bits();
        let (sign, exp, man) = ((b >> 16) & 0x8000, ((b >> 23) & 0xff) as i32 - 127 + 15, b & 0x7f_ffff);
        *d = if exp >= 31 {
            (sign | 0x7c00) as u16
        } else if exp <= 0 {
            sign as u16
        } else {
            let mut h = sign | ((exp as u32) << 10) | (man >> 13);
            if (man & 0x1fff) > 0x1000 || ((man & 0x1fff) == 0x1000 && (h & 1) == 1) {
                h += 1;
            }
            h as u16
        };
    }
}
