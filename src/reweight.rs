use std::error::Error;


use rayon::prelude::*;

/// kernel is a m by r_p matrix
pub fn reweight(
    mass: &mut [f64],
    kernel: &mut [f64],
    r_p: usize,
    ldk: usize,
    c_k: usize,
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
        let _k_end = k_begin + r_k;
        // let okernel = &mut kernel[k_begin..];
        // let omass = &mass[offset..];
        let ocl = r_k - offset;
        let ocn = c_k - offset;
        debug_assert_eq!(kernel[offset * ldk..].len(), ocn * ldk);

        let mut update_k_increment_0 = |r: usize, ker: &mut [f64], mass: &mut [f64]| {
            debug_assert!(r < r_p);
            for c in 1..ocn {
                let denom = ker[k_begin + r];

                const CHUNK_SIZE: usize = 512;
                if denom != 0.0f64 {
                    let factor = -ker[k_begin + r + c * ldk] / denom;
                    // debug_assert!(factor.abs() <= 1.0f64);
                    let mut j = 0usize;

                    {
                        let rhs = ker[k_begin..k_begin + ocl].to_vec();
                        let start = k_begin + c * ldk;
                        let end = start + ocl;
                        ker[start..end]
                            .par_chunks_mut(CHUNK_SIZE)
                            .zip(rhs.par_chunks(CHUNK_SIZE))
                            .for_each(|(lhs, r)| {
                                lhs.iter_mut().zip(r.iter())
                                    .for_each(|(l, r)| *l += factor * (*r))
                            });

                    }

                //
                //     loop {
                //         let this_chunk_size = if j + CHUNK_SIZE < ocl {
                //             CHUNK_SIZE
                //         } else if j < ocl {
                //             ocl - j
                //         } else {
                //             break;
                //         };
                //         for jj in 0..this_chunk_size {
                //             ker[k_begin + c * ldk + j + jj] += factor * ker[k_begin + j + jj];
                //         }
                //         j += this_chunk_size;
                //     }
                }

                if r != 0usize {
                    ker[k_begin + c * ldk + r] = ker[k_begin + c * ldk]
                }
                ker[k_begin + c * ldk] = 0.0f64;
            }

            if r != 0 {
                mass[offset + r] = mass[offset];
                mass[offset] = 0.0f64;
                ker.swap(k_begin + r, k_begin);
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
            let mut curr_max = kernel[k_begin + r].abs();
            for i in 1..ocn {
                let test = k_begin + i * ldk + r;
                let test_val = kernel[test].abs();
                if test_val > curr_max {
                    max_col = i;
                    curr_max = test_val;
                }
            }
            if max_col != 0 {
                for i in 0..ocl {
                    kernel.swap(k_begin + i, k_begin + max_col * ldk + i)
                }
                c_idx.swap(max_col, 0usize);
            }
            update_k_increment_0(r, kernel, mass);
        } else {
            let mut found = offset;
            let mut test_val = mass[found] / kernel[k_begin];

            debug_assert!(!test_val.is_nan());

            for i in 1..ocl {
                let fract = mass[offset+i] / kernel[k_begin+i];
                if f64::abs(fract) < f64::abs(test_val) {
                    found = i + offset;
                    test_val = fract;
                }
            }
            let r = found - offset;
            let factor = -test_val;
            for i in 0..ocl {
                if i == r {
                    mass[i+offset] = 0.0f64;
                } else {
                    mass[i+offset] += factor * kernel[k_begin+i];
                }
                debug_assert!(mass[i+offset] >= 0.0f64 && mass[i+offset] <= 1.0f64);
            }

            update_k_increment_0(r, kernel, mass);

            zeros_in_P = mass[(offset + 1)..]
                .iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    if p.abs() == 0.0f64 {
                        Some(i + offset + 1)
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
