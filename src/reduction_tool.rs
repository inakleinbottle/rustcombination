use lapack::dgesvd;
use std::error::Error;

pub trait LinearAlgebraReductionTool {
    fn move_mass(
        &self,
        weights: &mut [f64],
        points: &mut [f64],
        max_set: &mut Vec<usize>,
    ) -> Result<(), Box<dyn Error>>;
}

pub struct SVDReductionTool {
    no_coords: usize,
    no_points: usize,
}

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

        let mut info: i32 = 0;
        unsafe {
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

        let split_point = s.partition_point(|x| *x <= Self::THRESHOLD);
        s.shrink_to(split_point);

        Ok(s)
    }
}

impl LinearAlgebraReductionTool for SVDReductionTool {
    fn move_mass(
        &self,
        weights: &mut [f64],
        points: &mut [f64],
        max_set: &mut Vec<usize>,
    ) -> Result<(), Box<dyn Error>> {
        let min_set: Vec<usize> = Vec::new();
        max_set.clear();

        let mut m_cog = vec![0.0f64; self.no_coords];
        for i in 0..self.no_points {
            for j in 0..self.no_coords {
                m_cog[j] += points[j + i * self.no_coords] * weights[i];
            }
        }

        let mut new_weights = Vec::from(weights);

        let mut kernel = self.find_kernel(
            points,
            self.no_coords,
            self.no_coords as isize,
            self.no_points as isize,
        )?;

        let r_p = self.no_points;
        let c_k = kernel.len() / self.no_points;

        Ok(())
    }
}
