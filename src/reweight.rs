
pub fn reweight(
    mass: &mut [f64],
    kernel: &mut [f64],
    r_p: usize,
    c_k: usize,
    ldk: usize,
    tol: f64) {
    let r_k = r_p;

    let mut zeros_in_P: Vec<usize> = mass.iter()
        .enumerate()
        .filter_map(|(&i, p)| { if p.abs() == 0.0f64 { Some(i) } else { None } })
        .rev()
        .collect();

    let mut offset = 0usize;
    while offset < c_k {
        let okernel = &kernel[offset*(ldk+1)..];
        let omass = &mass[offset..];
        let ocl = r_k - offset;
        let ocn = c_k - offset;

        let update_k_increment_0 = |r| {
            let mut c = 0usize;
        };


        if let Some(idx) = zeros_in_P.pop() {
            let r = idx - offset;
            let found =
        }
    }
}


use std::collections::vec_deque;
