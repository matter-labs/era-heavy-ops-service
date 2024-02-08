use super::*;

#[derive(Debug)]
pub enum CompressionError {
    GenerationCompressionSetupError(u8),
    GenerationCompressionForWrapperSetupError(u8),
    GenerationCompressionProofError(u8),
    GenerationCompressionForWrapperProofError(u8),
}

#[derive(Debug)]
pub enum WrapperError {
    Synthesis(SynthesisError),
    Gpu(GpuError),
    Compression(CompressionError),
}

pub type WrapperResult<T> = Result<T, WrapperError>;

impl From<SynthesisError> for WrapperError {
    fn from(error: SynthesisError) -> Self {
        Self::Synthesis(error)
    }
}

impl From<GpuError> for WrapperError {
    fn from(error: GpuError) -> Self {
        Self::Gpu(error)
    }
}

impl From<CompressionError> for WrapperError {
    fn from(error: CompressionError) -> Self {
        Self::Compression(error)
    }
}

impl From<ProvingError> for WrapperError {
    fn from(error: ProvingError) -> Self {
        match error {
            ProvingError::Synthesis(error) => Self::Synthesis(error),
            ProvingError::Gpu(error) => Self::Gpu(error),
        }
    }
}
