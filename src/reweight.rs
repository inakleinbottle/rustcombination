//!
//! kernel is a m by r_p matrix
pub fn reweight(
    mass: &mut [f64],
    kernel: &mut [f64],
    r_p: usize,
    c_k: usize,
    ldk: usize,
    tol: f64,
) {
    let r_k = r_p;

    let mut r_idx: Vec<usize> = (0..r_p).collect();
    let mut c_idx: Vec<usize> = (0..c_k).collect();

    let mut zeros_in_P: Vec<usize> = mass
        .iter()
        .enumerate()
        .filter_map(|(i, p)| if p.abs() == 0.0f64 { Some(i) } else { None })
        .rev()
        .collect();

    let mut offset = 0usize;
    while offset < c_k {
        let k_begin = offset * (ldk + 1);
        let k_end = k_begin + r_k - offset;
        let okernel = &mut kernel[k_begin..];
        let omass = &mass[offset..];
        let ocl = r_k - offset;
        let ocn = c_k - offset;

        let update_k_increment_0 = |r| {
            let mut c = 0usize;
        };

        if let Some(idx) = zeros_in_P.pop() {
            let r = idx;
            let mut max_col = 0;
            let mut curr_max = kernel[max_col].abs();
            for i in 0..ocn {
                let test = r + i * ldk;
                let test_val = kernel[test].abs();
                if test_val > curr_max {
                    max_col = i;
                    curr_max = test_val;
                }
            }
            if max_col != 0 {
                for i in 0..ocl {
                    okernel.swap(i, i + r + (max_col * ldk))
                }
                c_idx.swap(max_col, 0usize);
            }
        } else {
        }
    }
}

use std::collections::vec_deque;
