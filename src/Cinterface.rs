use std::ffi::{c_void};

use libc::{c_double, size_t};

use super::{recombine, RecombineInterface};

pub type index_integer = size_t;
#[link_name = "expander"]
pub type ExpanderFunc = Option<extern "C" fn(*mut c_void, *mut c_double, *mut c_void)>;


#[link_name = "sCloud"]
#[repr(C)]
pub struct sCloud
{
    NoActiveWeightsLocations: index_integer,
    WeightBuf: *mut c_double,
    LocationBuf: *mut c_void,
    end: *mut c_void
}

#[link_name = "sRCloudInfo"]
#[repr(C)]
pub struct sRCloudInfo
{
    No_KeptLocations: index_integrer,
    NewWeightBuf: *mut c_double,
    KeptLocations: *mut index_integer,
    end: *mut c_void;
}


#[link_name = "RecombineInterface"]
#[repr(C)]
pub struct CRecombineInterface {
    pInCloud: *mut sCloud,
    pOutCloudInfo: *mut sRCloudInfo,
    degree: index_integer,
    func: ExpanderFunc,
    end: *mut c_void
}

impl RecombineInterface for CRecombineInterface {}


pub extern "C" fn Recombine(interface: *mut c_void)
{
    unsafe {
        recombine(&*(interface as *mut CRecombineInterface));
    }
}
