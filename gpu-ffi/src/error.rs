#[derive(Clone, Debug)]
pub enum GpuError {
    DeviceGetCountErr,
    DeviceGetDeviceMemoryInfoErr,
    CreateContextErr,
    SetBasesErr,
    SchedulingErr,
    GetExponentAddressErr,
    GetResultAddressesErr,
    StartProcessingErr,
    FinishProcessingErr,
    DestroyContextErr,

    MemPoolCreateErr,
    AsyncPoolMallocErr,
    AsyncMemcopyErr,
    NttExecErr,
    StremCreateErr,
    StreamDestroyErr,
    StreamWaitEventErr,
    StreamSyncErr,

    EventCreateErr,
    EventRecordErr,
    EventDestroyErr,
    EventSyncErr,

    DevicePeerAccessErr,
    MemPoolPeerAccessErr,
    SetDeviceErr,
}
