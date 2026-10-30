//! Parallel row-major SYRK / Cholesky / triangular inverse for the GPTQ compiler.

fn workers() -> usize {
    std::thread::available_parallelism().map_or(1, |p| p.get()).clamp(1, 8)
}

/// Upper triangle of `xs^T xs` (`xs` is `n x c`).
pub fn syrk_upper(xs: &[f32], n: usize, c: usize, out: &mut [f32]) {
    assert_eq!(xs.len(), n * c);
    assert_eq!(out.len(), c * c);
    // Column-major copy so each entry is a contiguous dot product over samples.
    let mut xt = vec![0.0f32; c * n];
    for (t, row) in xs.chunks_exact(c).enumerate() {
        for (i, &v) in row.iter().enumerate() {
            xt[i * n + t] = v;
        }
    }
    let tiles = c.div_ceil(4);
    let w = workers();
    let out_addr = out.as_mut_ptr() as usize;
    let xt = &xt;
    std::thread::scope(|s| {
        for wid in 0..w {
            s.spawn(move || {
                // Round-robin row tiles balance the triangle.
                for ti in (wid..tiles).step_by(w) {
                    let i0 = ti * 4;
                    for j0 in (i0..c).step_by(4) {
                        let mut acc = [[0.0f32; 4]; 4];
                        dot_tile(xt, n, c, i0, j0, &mut acc);
                        for (a, acc_row) in acc.iter().enumerate() {
                            for (b, &v) in acc_row.iter().enumerate() {
                                let (i, j) = (i0 + a, j0 + b);
                                if j < c && j >= i {
                                    // SAFETY: each `(i, j)` is written by exactly one worker.
                                    unsafe { *(out_addr as *mut f32).add(i * c + j) = v };
                                }
                            }
                        }
                    }
                }
            });
        }
    });
}

/// `acc[a][b] = dot(xt[i0 + a], xt[j0 + b])`; rows past `c` count as zero.
fn dot_tile(xt: &[f32], n: usize, c: usize, i0: usize, j0: usize, acc: &mut [[f32; 4]; 4]) {
    let row = |r: usize| if r < c { &xt[r * n..(r + 1) * n] } else { &xt[0..0] };
    let ri = [row(i0), row(i0 + 1), row(i0 + 2), row(i0 + 3)];
    let rj = [row(j0), row(j0 + 1), row(j0 + 2), row(j0 + 3)];
    #[cfg(target_arch = "aarch64")]
    if ri.iter().chain(&rj).all(|r| r.len() == n) {
        // SAFETY: all eight rows hold `n` floats.
        unsafe { dot_tile_neon(&ri, &rj, n, acc) };
        return;
    }
    for a in 0..4 {
        for b in 0..4 {
            acc[a][b] = ri[a].iter().zip(rj[b]).map(|(x, y)| x * y).sum();
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_tile_neon(ri: &[&[f32]; 4], rj: &[&[f32]; 4], n: usize, acc: &mut [[f32; 4]; 4]) {
    use std::arch::aarch64::*;
    let mut v = [[vdupq_n_f32(0.0); 4]; 4];
    let mut t = 0;
    while t + 4 <= n {
        // SAFETY: `t + 4 <= n` for every row.
        unsafe {
            let a = [
                vld1q_f32(ri[0].as_ptr().add(t)),
                vld1q_f32(ri[1].as_ptr().add(t)),
                vld1q_f32(ri[2].as_ptr().add(t)),
                vld1q_f32(ri[3].as_ptr().add(t)),
            ];
            let b = [
                vld1q_f32(rj[0].as_ptr().add(t)),
                vld1q_f32(rj[1].as_ptr().add(t)),
                vld1q_f32(rj[2].as_ptr().add(t)),
                vld1q_f32(rj[3].as_ptr().add(t)),
            ];
            for x in 0..4 {
                for y in 0..4 {
                    v[x][y] = vfmaq_f32(v[x][y], a[x], b[y]);
                }
            }
        }
        t += 4;
    }
    for x in 0..4 {
        for y in 0..4 {
            let tail: f32 = (t..n).map(|k| ri[x][k] * rj[y][k]).sum();
            acc[x][y] = vaddvq_f32(v[x][y]) + tail;
        }
    }
}

fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let mut s = [0.0f64; 4];
    let (ac, bc) = (a.chunks_exact(4), b.chunks_exact(4));
    let tail: f64 = ac.remainder().iter().zip(bc.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ac.zip(bc) {
        s[0] += x[0] * y[0];
        s[1] += x[1] * y[1];
        s[2] += x[2] * y[2];
        s[3] += x[3] * y[3];
    }
    (s[0] + s[1]) + (s[2] + s[3]) + tail
}

/// Runs `f(i, row)` for rows `first..` of `rows`, round-robin across workers.
fn par_rows(rows: &mut [f64], c: usize, first: usize, f: impl Fn(usize, &mut [f64]) + Sync) {
    let w = workers();
    let mut buckets: Vec<Vec<(usize, &mut [f64])>> = (0..w).map(|_| Vec::new()).collect();
    for (r, row) in rows.chunks_exact_mut(c).enumerate() {
        buckets[r % w].push((first + r, row));
    }
    let f = &f;
    std::thread::scope(|s| {
        for bucket in buckets {
            s.spawn(move || {
                for (i, row) in bucket {
                    f(i, row);
                }
            });
        }
    });
}

const CHOL_BLOCK: usize = 64;

/// In-place lower Cholesky `A = L L^T` (reads the lower triangle); `false` if not positive definite.
pub fn cholesky_lower(a: &mut [f64], c: usize) -> bool {
    assert_eq!(a.len(), c * c);
    for k0 in (0..c).step_by(CHOL_BLOCK) {
        let k1 = (k0 + CHOL_BLOCK).min(c);
        // Diagonal block; earlier blocks were already subtracted by trailing updates.
        for i in k0..k1 {
            for j in k0..=i {
                let s = dot_f64(&a[i * c + k0..i * c + j], &a[j * c + k0..j * c + j]);
                let rem = a[i * c + j] - s;
                if i == j {
                    if rem <= 0.0 || !rem.is_finite() {
                        return false;
                    }
                    a[i * c + i] = rem.sqrt();
                } else {
                    a[i * c + j] = rem / a[j * c + j];
                }
            }
        }
        if k1 == c {
            break;
        }
        let (head, tail) = a.split_at_mut(k1 * c);
        let head = &*head;
        par_rows(tail, c, k1, |_, row| {
            for j in k0..k1 {
                let s = dot_f64(&row[k0..j], &head[j * c + k0..j * c + j]);
                row[j] = (row[j] - s) / head[j * c + j];
            }
        });
        let width = k1 - k0;
        let panel: Vec<f64> = tail.chunks_exact(c).flat_map(|r| r[k0..k1].to_vec()).collect();
        let panel = &panel;
        par_rows(tail, c, 0, |r, row| {
            let li = &panel[r * width..(r + 1) * width];
            for r2 in 0..=r {
                row[k1 + r2] -= dot_f64(li, &panel[r2 * width..(r2 + 1) * width]);
            }
        });
    }
    true
}

/// In-place inverse of a lower-triangular matrix (column-parallel forward substitution).
pub fn invert_lower(a: &mut [f64], c: usize) -> bool {
    assert_eq!(a.len(), c * c);
    if (0..c).any(|i| a[i * c + i] == 0.0 || !a[i * c + i].is_finite()) {
        return false;
    }
    let w = workers();
    let src = &*a;
    let cols: Vec<(usize, Vec<f64>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..w)
            .map(|wid| {
                s.spawn(move || {
                    (wid..c)
                        .step_by(w)
                        .map(|j| {
                            let mut x = vec![0.0f64; c - j];
                            x[0] = 1.0 / src[j * c + j];
                            for i in j + 1..c {
                                let s = dot_f64(&src[i * c + j..i * c + i], &x[..i - j]);
                                x[i - j] = -s / src[i * c + i];
                            }
                            (j, x)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    for (j, x) in cols {
        for (k, v) in x.into_iter().enumerate() {
            a[(j + k) * c + j] = v;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syrk_cholesky_inverse() {
        let (n, c) = (301usize, 133usize);
        let xs: Vec<f32> = (0..n * c)
            .map(|i| (((i * 2654435761usize) >> 7) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let mut h = vec![f32::NAN; c * c];
        syrk_upper(&xs, n, c, &mut h);
        for i in 0..c {
            for j in i..c {
                let r: f32 = (0..n).map(|t| xs[t * c + i] * xs[t * c + j]).sum();
                assert!((h[i * c + j] - r).abs() < 1e-3, "syrk ({i},{j})");
            }
        }

        let mut a = vec![0.0f64; c * c];
        for i in 0..c {
            for j in 0..=i {
                a[i * c + j] = h[j * c + i] as f64 + if i == j { 1.0 } else { 0.0 };
            }
        }
        let orig = a.clone();
        assert!(cholesky_lower(&mut a, c));
        for i in 0..c {
            for j in 0..=i {
                let r: f64 = (0..=j).map(|k| a[i * c + k] * a[j * c + k]).sum();
                assert!((r - orig[i * c + j]).abs() < 1e-8, "L L^T ({i},{j})");
            }
        }
        let l = a.clone();
        assert!(invert_lower(&mut a, c));
        for i in 0..c {
            for j in 0..=i {
                let r: f64 = (j..=i).map(|k| l[i * c + k] * a[k * c + j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((r - want).abs() < 1e-9, "L L^-1 ({i},{j})");
            }
        }
    }
}
