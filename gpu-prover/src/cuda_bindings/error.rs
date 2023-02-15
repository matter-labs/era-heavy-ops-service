#[derive(Clone, Debug)]
pub enum GpuError {
    DeviceGetCountErr(u32),
    DeviceGetDeviceMemoryInfoErr(u32),
    CreateContextErr(u32),
    PermutationSetupErr(u32),
    SetBasesErr(u32),
    SchedulingErr(u32),
    GetExponentAddressErr(u32),
    GetResultAddressesErr(u32),
    StartProcessingErr(u32),
    FinishProcessingErr(u32),
    DestroyContextErr(u32),

    MallocErr(u32),
    MemFreeErr(u32),
    MemPoolCreateErr(u32),
    AsyncPoolMallocErr(u32),
    AsyncH2DErr(u32),
    AsyncMemFreeErr(u32),
    AsyncMemcopyErr(u32),
    NttExecErr(u32),
    StremCreateErr(u32),
    StreamDestroyErr(u32),
    StreamWaitEventErr(u32),
    StreamSyncErr(u32),

    FFAssignErr(u32),
    PermutationPolysErr(u32),
    MSMErr(u32),
    EvaluationErr(u32),
    NTTErr(u32),
    MultiGpuNTTErr(u32),
    MultiGpuLargeNTTErr(u32),
    BitReverseErr(u32),
    MultiGpuBitReverseErr(u32),
    ArithmeticErr(u32),
    DistributeOmegasErr(u32),

    EventCreateErr(u32),
    EventRecordErr(u32),
    EventDestroyErr(u32),
    EventSyncErr(u32),

    DevicePeerAccessErr(u32),
    MemPoolPeerAccessErr(u32),
    SetDeviceErr(u32),

    DeviceInUseErr(usize), // We can't allocate two Contexts with same device_id
    AssemblyError(String),
}

pub type GpuResult<T> = Result<T, GpuError>;
