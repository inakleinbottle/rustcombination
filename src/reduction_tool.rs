use lapack::{dgelsd, dgesvd};
use rayon::iter::split;
use std::error::Error;
use std::mem;

use crate::reweight::reweight;

pub trait LinearAlgebraReductionTool {
    fn move_mass(
        &self,
        weights: &mut [f64],
        points: &mut [f64],
        max_set: &mut Vec<usize>,
        no_points: usize,
    ) -> Result<(), Box<dyn Error>>;

    fn num_linalg_calls(&self) -> usize;
}

pub struct SVDReductionTool;

impl SVDReductionTool {
    const THRESHOLD: f64 = 10e-12f64;
    const PROB_ZERO_TOL: f64 = 0.0f64;

    fn find_kernel(
        &self,
        input: &mut [f64],
        input_rows: usize,
        lda: isize,
        ldk: isize,
    ) -> Result<Vec<f64>, Box<dyn Error>> {
        let input_cols = input.len() / lda as usize;

        let mut s = vec![0.0f64; input_rows.min(input_cols)];
        let mut vt = vec![0.0f64; input_cols * input_cols];
        let ldvt = input_cols as i32;
        let mut work = vec![1.0f64];
        let mut lwork = -1;

        let mut info: i32 = 0;
        unsafe {
            if (lwork == -1) {
                dgesvd(
                    b'N',              //
                    b'A',              //
                    input_cols as i32, //
                    input_rows as i32, //
                    input,             //
                    lda as i32,
                    &mut s,
                    &mut [0.0f64],
                    1,
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
                input_cols as i32, //
                input_rows as i32, //
                input,             //
                lda as i32,        //
                &mut s,
                &mut [0.0f64],
                1, // No return U
                &mut vt,
                ldvt,
                &mut [0.0f64],
                -1,
                &mut info,
            )
        }

        let split_point = s.partition_point(|x| *x > Self::THRESHOLD);

        let mut result = vec![0.0f64; (ldk as usize) * (input_cols - split_point)];

        for i in split_point..input_cols {
            for j in 0..input_cols {
                result[j + (i - split_point) * (ldk as usize)] = vt[i + j * (ldvt as usize)];
            }
        }

        Ok(s)
    }

    fn sharpen_weights(
        &self,
        mut min_set: Vec<usize>,
        max_set: &mut Vec<usize>,
        points: &[f64],
        weights: &mut [f64],
        m_cog: Vec<f64>,
        no_coords: usize,
    ) -> Result<(), Box<dyn Error>> {
        let mut temp_min_set = Vec::<usize>::new();

        while (temp_min_set.len() < min_set.len()) {
            let mut avec = vec![0.0f64; no_coords];
            let mut wvec = vec![0.0f64; min_set.len()];
            let mut bvec = m_cog.clone();

            for i in 0..min_set.len() {
                for j in 0..no_coords {
                    avec[j + i * no_coords] = points[j + no_coords * min_set[i]];
                }
            }

            let m = no_coords as i32;
            let n = min_set.len() as i32;
            let nrhs = 1;
            let lda = m as i32;
            let mut ldb = m as i32;
            let mut lwork = -1;
            let mut info = 0;
            let mut work = vec![1.0f64];
            let mut iwork = vec![1i32];
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

            temp_min_set.clear();
            for i in 0..min_set.len() {
                if bvec[i] <= 0.0f64 {
                    weights[min_set[i]] = 0.0f64;
                    max_set.push(min_set[i]);
                } else {
                    weights[min_set[i]] = bvec[i];
                    temp_min_set.push(min_set[i]);
                }
            }

            mem::swap(&mut min_set, &mut temp_min_set);
        }

        Ok(())
    }
}

impl LinearAlgebraReductionTool for SVDReductionTool {
    fn move_mass(
        &self,
        weights: &mut [f64],
        points: &mut [f64],
        max_set: &mut Vec<usize>,
        no_coords: usize,
    ) -> Result<(), Box<dyn Error>> {
        let min_set: Vec<usize> = Vec::new();
        max_set.clear();

        let no_points = weights.len();

        let mut m_cog = vec![0.0f64; no_coords];
        for i in 0..no_points {
            for j in 0..no_coords {
                m_cog[j] += points[j + i * no_coords] * weights[i];
            }
        }

        let mut new_weights = weights.to_vec();

        let mut kernel =
            self.find_kernel(points, no_coords, no_coords as isize, no_points as isize)?;

        let r_p = no_points;
        let c_k = kernel.len() / no_points;

        let (permute_r, permute_c) = reweight(
            &mut new_weights,
            &mut kernel,
            r_p,
            c_k,
            no_points,
            Self::PROB_ZERO_TOL,
        )?;

        for &v in &permute_r {
            max_set.push(v);
            weights[v] = 0.0f64;
        }

        let mut min_set: Vec<usize> = Vec::new();
        for i in c_k..r_p {
            let wi = permute_r[i];
            if weights[wi] == 0.0f64 {
                max_set.push(wi);
            } else {
                min_set.push(wi);
            }
        }

        self.sharpen_weights(min_set, max_set, points, weights, m_cog, no_coords)
    }

    fn num_linalg_calls(&self) -> usize {
        todo!()
    }
}
