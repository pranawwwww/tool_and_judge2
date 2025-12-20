use std::{path::PathBuf, sync::LazyLock};

pub static BASE_DATASET_PATH: LazyLock<PathBuf> = LazyLock::new(|| PathBuf::from("tool/dataset"));
pub static BASE_RESULT_PATH: LazyLock<PathBuf> = LazyLock::new(|| PathBuf::from("tool/result"));