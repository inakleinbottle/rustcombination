use std::error::Error;

/// kernel is a m by r_p matrix
pub fn reweight(
    mass: &mut [f64],
    kernel: &mut [f64],
    r_p: usize,
    c_k: usize,
    ldk: usize,
    _tol: f64,
) -> Result<(Vec<usize>, Vec<usize>), Box<dyn Error>> {
    debug_assert_eq!(kernel.len(), c_k * ldk);
    debug_assert!(!kernel.iter().any(|k| k.is_nan()));
    debug_assert!(!mass.iter().any(|k| k.is_nan()));
    debug_assert!(kernel.iter().all(|k| k.is_finite()));
    debug_assert!(mass.iter().all(|k| k.is_finite()));

    let r_k = r_p;

    let mut r_idx: Vec<usize> = (0..r_p).collect();
    let mut c_idx: Vec<usize> = (0..c_k).collect();

    #[allow(non_snake_case)]
    let mut zeros_in_P: Vec<usize> = mass
        .iter()
        .enumerate()
        .filter_map(|(i, p)| if p.abs() == 0.0f64 { Some(i) } else { None })
        .rev()
        .collect();

    for offset in 0..c_k {
        let k_begin = offset * (ldk + 1);
        let k_end = k_begin + r_k;
        // let okernel = &mut kernel[k_begin..];
        // let omass = &mass[offset..];
        let ocl = r_k - offset;
        let ocn = c_k - offset;
        debug_assert_eq!(kernel[offset * ldk..].len(), ocn * ldk);

        let mut update_k_increment_0 = |r: usize, ker: &mut [f64], mass: &mut [f64]| {
            debug_assert!(r < r_p);
            for c in 1..ocn {
                let denom = ker[offset + r];

                const CHUNK_SIZE: usize = 512;
                if denom != 0.0f64 {
                    let factor = -ker[offset + r + c * ldk] / denom;
                    let mut j = 0usize;
                    loop {
                        let this_chunk_size = if j + CHUNK_SIZE < ocl {
                            CHUNK_SIZE
                        } else if j < ocl {
                            ocl - j
                        } else {
                            break;
                        };
                        for jj in 0..this_chunk_size {
                            ker[offset + c * ldk + j + jj] += factor * ker[offset + j + jj];
                        }
                        j += this_chunk_size;
                    }
                }

                if r != 0usize {
                    ker[offset + c * ldk + r] = ker[offset + c * ldk]
                }
                ker[offset + c * ldk] = 0.0f64;
            }

            if r != 0 {
                mass[offset + r] = mass[offset];
                mass[offset] = 0.0f64;
                ker.swap(offset + r, offset);
                r_idx.swap(offset + r, offset);
            }
            debug_assert!(!ker.iter().any(|k| k.is_nan()));
            debug_assert!(!mass.iter().any(|k| k.is_nan()));
            debug_assert!(ker.iter().all(|k| k.is_finite()));
            debug_assert!(mass.iter().all(|k| k.is_finite()));
        };

        if let Some(idx) = zeros_in_P.pop() {
            let r = idx - offset;
            let mut max_col = 0;
            let mut curr_max = kernel[offset + r + max_col].abs();
            for i in 0..ocn {
                let test = offset + r + i * ldk;
                let test_val = kernel[test].abs();
                if test_val > curr_max {
                    max_col = i;
                    curr_max = test_val;
                }
            }
            if max_col != 0 {
                for i in 0..ocl {
                    kernel.swap(offset + i, offset + i + r + (max_col) * ldk)
                }
                c_idx.swap(max_col, 0usize);
            }
            update_k_increment_0(r, kernel, mass);
        } else {
            let mut found = offset;
            let mut test_val = mass[found] / kernel[found];

            debug_assert!(!test_val.is_nan());

            for i in offset..r_p {
                let fract = mass[i] / kernel[i];
                if f64::abs(fract) < f64::abs(test_val) {
                    found = i;
                    test_val = fract;
                }
            }
            let r = found - offset;
            let factor = -test_val;
            for i in offset..r_p {
                if i == found {
                    mass[i] = 0.0f64;
                } else {
                    mass[i] += factor * kernel[i];
                }
                debug_assert!(mass[i] >= 0.0f64);
            }

            update_k_increment_0(r, kernel, mass);

            zeros_in_P = mass[(offset + 1)..]
                .iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    if p.abs() == 0.0f64 {
                        Some(i + offset + 2)
                    } else {
                        None
                    }
                })
                .rev()
                .collect()
        }
    }

    Ok((r_idx, c_idx))
}
