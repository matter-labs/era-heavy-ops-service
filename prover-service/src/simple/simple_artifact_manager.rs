use std::io::{BufReader, Read};

use crate::utils::get_artifacts_dir;

use super::*;

#[cfg(feature = "legacy")]
pub const SETUP_FILE_NAME: &str = "setup";
#[cfg(not(feature = "legacy"))]
pub const SETUP_FILE_NAME: &str = "minimal_setup";

pub struct SimpleArtifactManager;

impl ArtifactProvider for SimpleArtifactManager {
    type ArtifactError = std::io::Error;

    fn get_setup(&self, circuit_id: u8) -> Result<Box<dyn Read>, Self::ArtifactError> {
        let artifact_dir = get_artifacts_dir();
        let setup_file_path = format!(
            "{}/{}_{}.bin",
            artifact_dir.to_str().unwrap(),
            SETUP_FILE_NAME,
            circuit_id
        );
        let setup_file_path = std::path::Path::new(&setup_file_path);
        let file = std::fs::File::open(&setup_file_path)?;
        let buf_reader = BufReader::new(file);

        Ok(Box::new(buf_reader))
    }

    fn get_vk(&self, circuit_id: u8) -> Result<ZkSyncVerificationKey<Bn256>, Self::ArtifactError> {
        let artifact_dir = get_artifacts_dir();
        let vk_file_path = format!("{}/vk_{}.json", artifact_dir.to_str().unwrap(), circuit_id);
        let vk_file_path = std::path::Path::new(&vk_file_path);
        let vk_file = std::fs::File::open(&vk_file_path)?;
        let vk = serde_json::from_reader(&vk_file).expect(vk_file_path.to_str().unwrap());

        Ok(vk)
    }
}
