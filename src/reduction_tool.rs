use crate::{RecombineError};
use lapack::{dgelsd, dgesvd};

use std::cmp::Ordering;
use std::error::Error;
use std::mem;

use crate::reweight::reweight;

pub trait LinearAlgebraReductionTool {
    fn move_mass(
        &self,
        weights: &mut [f64],
        points: &[f64],
        no_points: usize,
    ) -> Result<Vec<usize>, Box<dyn Error>>;

    fn num_linalg_calls(&self) -> usize;
}

pub struct SVDReductionTool;

impl SVDReductionTool {
    const THRESHOLD: f64 = 10e-12f64;
    const PROB_ZERO_TOL: f64 = 0.0f64;

    pub fn new() -> SVDReductionTool {
        SVDReductionTool
    }

    fn find_kernel(
        &self,
        input: &[f64],
        input_rows: usize,
        lda: isize,
        ldk: isize,
    ) -> Result<Vec<f64>, Box<dyn Error>> {
        #[allow(non_snake_case)]
        let mut A = input.to_owned();
        let input_cols = input.len() / lda as usize;

        let mut s = vec![0.0f64; usize::min(input_rows, input_cols)];
        let mut u = vec![0.0f64];
        let ldu = 1;
        let mut vt = vec![0.0f64; input_cols * input_cols];
        let ldvt = input_cols as i32;
        let mut work = vec![0.0f64];
        let mut lwork = -1;

        let mut info: i32 = 0;
        unsafe {
            if lwork == -1 {
                dgesvd(
                    b'N',              //
                    b'A',              //
                    input_rows as i32, //
                    input_cols as i32, //
                    &mut A,             //
                    lda as i32,
                    &mut s,
                    &mut u,
                    ldu,
                    &mut vt,
                    ldvt,
                    &mut work,
                    lwork,
                    &mut info,
                );
                lwork = work[0] as i32;
                work.resize(lwork as usize, 0.0f64);
            }
            dgesvd(
                b'N',              //
                b'A',              //
                input_rows as i32, //
                input_cols as i32, //
                &mut A,             //
                lda as i32,        //
                &mut s,
                &mut u,
                ldu, // No return U
                &mut vt,
                ldvt,
                &mut work,
                lwork,
                &mut info,
            )
        }
        match info.cmp(&0) {
            Ordering::Less => {
                return Err(RecombineError::LinearAlgebraError(format!(
                    "Argument {} has incorrect value",
                    -info
                ))
                .into());
            }
            Ordering::Greater => {
                return Err(
                    RecombineError::LinearAlgebraError("svd failed to converge".into()).into(),
                );
            }
            Ordering::Equal => (),
        }

        let split_point = s.partition_point(|x| *x > Self::THRESHOLD);
        let mut result = vec![0.0f64; (ldk as usize) * (input_cols - split_point)];

        for i in split_point..input_cols {
            for j in 0..input_cols {
                result[j + (i - split_point) * (ldk as usize)] = vt[i + j * (ldvt as usize)];
            }
        }

        Ok(result)
    }

    fn sharpen_weights(
        &self,
        mut min_set: Vec<usize>,
        mut max_set: Vec<usize>,
        points: &[f64],
        weights: &mut [f64],
        m_cog: &[f64],
        no_coords: usize,
    ) -> Result<Vec<usize>, Box<dyn Error>> {
        let mut temp_min_set = Vec::<usize>::with_capacity(min_set.len());


        while temp_min_set.len() < min_set.len() {
            let mut avec = vec![0.0f64; min_set.len() * no_coords];
            let _wvec = vec![0.0f64; min_set.len()];
            let mut bvec = m_cog.to_owned();
            debug_assert!(bvec.len() >= min_set.len());

            for (i, &mi) in min_set.iter().enumerate() {
                for j in 0..no_coords {
                    avec[j + i * no_coords] = points[j + no_coords * mi];
                    // print!("{:>7.3} ", avec[j+i*no_coords]);
                }
                // print!("    {:>7.3}\n", bvec[i]);
            }
            // println!(" ");

            let m = no_coords as i32;
            let n = min_set.len() as i32;
            let nrhs = 1;
            let lda = m as i32;
            let ldb = m as i32;
            let mut lwork = -1;
            let mut info = 0;
            let mut work = vec![0.0f64];
            let mut iwork = vec![0i32];
            let mut rank: i32 = 0;
            let rcond = 0.0f64;

            let mut svec = vec![0.0f64; n as usize];

            unsafe {
                if lwork == -1 {
                    dgelsd(
                        m, n, nrhs, &mut avec, lda, &mut bvec, ldb, &mut svec, rcond, &mut rank,
                        &mut work, lwork, &mut iwork, &mut info,
                    );
                    lwork = work[0] as i32;
                    work.resize(lwork as usize, 0.0f64);
                    iwork.resize(iwork[0] as usize, 0);
                }
                dgelsd(
                    m, n, nrhs, &mut avec, lda, &mut bvec, ldb, &mut svec, rcond, &mut rank,
                    &mut work, lwork, &mut iwork, &mut info,
                );
            }
            match info.cmp(&0) {
                Ordering::Equal => (),
                Ordering::Less => {
                    return Err(
                        RecombineError::LinearAlgebraError(format!("Incorrect argument {} to DGELSD", info)).into()
                    );
                },
                Ordering::Greater => {
                    return Err(
                        RecombineError::LinearAlgebraError("DGELSD failed to converge".into()).into()
                    )
                }
            }

            temp_min_set.clear();
            for (i, &mi) in min_set.iter().enumerate() {
                if bvec[i] <= 0.0f64 {
                    weights[mi] = 0.0f64;
                    debug_assert!(!max_set.contains(&mi));
                    max_set.push(mi);
                } else {
                    weights[mi] = bvec[i];
                    debug_assert!(!temp_min_set.contains(&mi));
                    temp_min_set.push(mi);
                }
                debug_assert!(weights[mi] >= 0.0 && weights[mi] <= 1.0);
            }

            mem::swap(&mut min_set, &mut temp_min_set);
        }

        Ok(max_set)
    }
}

impl LinearAlgebraReductionTool for SVDReductionTool {
    fn move_mass(
        &self,
        weights: &mut [f64],
        points: &[f64],
        no_coords: usize,
    ) -> Result<Vec<usize>, Box<dyn Error>> {
        let mut min_set = Vec::<usize>::new();
        let mut max_set = Vec::<usize>::new();

        let no_points = weights.len();
        debug_assert!(weights.iter().all(|v| *v >= 0.0 && *v <= 1.0));

        let mut m_cog = vec![0.0f64; no_coords];
        for (i, &w) in weights.iter().enumerate() {
            debug_assert!(w <= 1.0 && w >= 0.0);
            for j in 0..no_coords {
                m_cog[j] += (points[j + i * no_coords] * w);
            }
        }

        let mut new_weights = weights.to_vec();

        let mut kernel =
            self.find_kernel(&points, no_coords, no_coords as isize, no_points as isize)?;

        debug_assert!(kernel.iter().all(|v| v.is_finite()));
        debug_assert!(!kernel.iter().any(|v| v.is_nan()));

        let r_p = no_points;
        let ldk = no_points;
        let c_k = kernel.len() / ldk;

        let (permute_r, _permute_c) = reweight(
            &mut new_weights,
            &mut kernel,
            r_p,
            ldk,
            c_k,
            Self::PROB_ZERO_TOL,
        )?;
        //
        // println!("kernel");
        // for i in 0..c_k {
        //     for j in 0..ldk {
        //         print!("{:>7.3} ", kernel[i*ldk + j]);
        //     }
        //     println!(" ");
        // }
        // println!("done");

        for (i, &v) in permute_r.iter().enumerate().take(c_k) {
            max_set.push(v);
            debug_assert_eq!(new_weights[i], 0.0);
            weights[v] = 0.0f64;
        }

        for i in c_k..r_p {
            let wi = permute_r[i];
            weights[wi] = new_weights[i];
            debug_assert!(new_weights[i] >= 0.0 && new_weights[i] <= 1.0);
            if weights[wi] == 0.0f64 {
                max_set.push(wi);
            } else {
                min_set.push(wi);
            }
        }

        self.sharpen_weights(min_set, max_set, points, weights, &m_cog, no_coords)
    }

    fn num_linalg_calls(&self) -> usize {
        0
    }
}
