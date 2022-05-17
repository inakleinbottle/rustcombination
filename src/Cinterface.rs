#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use crate::ConditionerHelper;
use libc::{c_double, size_t};
use std::error::Error;
use std::ffi::c_void;

use super::{recombine, RecombineInterface};

pub type index_integer = size_t;
#[link_name = "expander"]
pub type ExpanderFunc = Option<extern "C" fn(*mut c_void, *mut c_double, *mut c_void)>;

#[link_name = "sCloud"]
#[repr(C)]
pub struct sCloud {
    NoActiveWeightsLocations: index_integer,
    WeightBuf: *mut c_double,
    LocationBuf: *mut c_void,
    end: *mut c_void,
}

#[link_name = "sRCloudInfo"]
#[repr(C)]
pub struct sRCloudInfo {
    No_KeptLocations: index_integer,
    NewWeightBuf: *mut c_double,
    KeptLocations: *mut index_integer,
    end: *mut c_void,
}

#[link_name = "RecombineInterface"]
#[repr(C)]
pub struct CRecombineInterface {
    pInCloud: *mut sCloud,
    pOutCloudInfo: *mut sRCloudInfo,
    degree: index_integer,
    func: ExpanderFunc,
    end: *mut c_void,
}

impl RecombineInterface for CRecombineInterface {
    fn points_in_cloud(&self) -> usize {
        todo!()
    }

    fn expand_points(
        &self,
        output: &mut [f64],
        helper: &ConditionerHelper,
    ) -> Result<(), Box<dyn Error>> {
        todo!()
    }

    fn degree(&self) -> usize {
        todo!()
    }

    fn weights(&self) -> &[f64] {
        todo!()
    }

    fn set_output(&mut self, locs: &[usize], weights: &[f64]) {
        todo!()
    }
}

pub extern "C" fn Recombine(interface: *mut c_void) {
    unsafe {
        recombine(&mut *(interface as *mut CRecombineInterface)).unwrap();
    }
}
