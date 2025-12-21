use serde::{Deserialize, Serialize};

use crate::config::{Language, ToolConfig, ToolExperiment, TranslateMode, TranslateOption};

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub enum OriginalDataset {
    #[serde(rename = "BFCL_v4_multiple")]
    BfclV4Multiple,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub enum DatasetLanguage {
    #[serde(rename = "en")]
    English,
    #[serde(rename = "zh")]
    Chinese,
    #[serde(rename = "hi")]
    Hindi,
    #[serde(rename = "igbo")]
    Igbo,
}

impl DatasetLanguage {
    pub fn from_config_experiment(exp: &ToolExperiment) -> Self {
        match &exp.translate_mode {
            TranslateMode::Translated {
                language,
                option: _,
            } => match language {
                Language::Chinese => DatasetLanguage::Chinese,
                Language::Hindi => DatasetLanguage::Hindi,
                Language::Igbo => DatasetLanguage::Igbo,
            },
            TranslateMode::NotTranslated {} => DatasetLanguage::English,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub enum TranslateLevel {
    #[serde(rename = "na")]
    NotApplicable,
    #[serde(rename = "fulltrans")]
    FullyTranslated,
    #[serde(rename = "parttrans")]
    PartiallyTranslated,
}
impl TranslateLevel {
    pub fn from_config_experiment(exp: &ToolExperiment) -> Self {
        match &exp.translate_mode {
            TranslateMode::Translated {
                language: _,
                option,
            } => match &option {
                TranslateOption::FullyTranslated => TranslateLevel::FullyTranslated,
                TranslateOption::PartiallyTranslated => TranslateLevel::PartiallyTranslated,
                TranslateOption::FullyTranslatedPreTranslate => TranslateLevel::FullyTranslated,
                TranslateOption::FullyTranslatedPromptTranslate => TranslateLevel::FullyTranslated,
                TranslateOption::FullyTranslatedPostTranslate => TranslateLevel::FullyTranslated,
            },
            TranslateMode::NotTranslated {} => TranslateLevel::NotApplicable,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub enum AddNoiseMode {
    #[serde(rename = "nonoise")]
    NoNoise,
    #[serde(rename = "para")]
    Paraphrase,
    #[serde(rename = "syno")]
    Synonym,
}

impl AddNoiseMode {
    pub fn from_config_experiment(exp: &ToolExperiment) -> Self {
        match &exp.add_noise_mode {
            crate::config::AddNoiseMode::NoNoise => AddNoiseMode::NoNoise,
            crate::config::AddNoiseMode::Paraphrase => AddNoiseMode::Paraphrase,
            crate::config::AddNoiseMode::Synonym => AddNoiseMode::Synonym,
        }
    }
}
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub enum PromptTranslateMode {
    #[serde(rename = "prompt")]
    PromptTranslate,
    #[serde(rename = "noprompt")]
    NoPromptTranslate,
}

impl PromptTranslateMode {
    pub fn from_config_experiment(exp: &ToolExperiment) -> Self {
        match &exp.translate_mode {
            TranslateMode::Translated {
                language: _,
                option,
            } => match &option {
                TranslateOption::FullyTranslatedPromptTranslate => {
                    PromptTranslateMode::PromptTranslate
                }
                TranslateOption::FullyTranslated => PromptTranslateMode::NoPromptTranslate,
                TranslateOption::PartiallyTranslated => PromptTranslateMode::NoPromptTranslate,
                TranslateOption::FullyTranslatedPreTranslate => {
                    PromptTranslateMode::NoPromptTranslate
                }
                TranslateOption::FullyTranslatedPostTranslate => {
                    PromptTranslateMode::NoPromptTranslate
                }
            },
            TranslateMode::NotTranslated {} => PromptTranslateMode::NoPromptTranslate,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub enum PreTranslateMode {
    #[serde(rename = "pretrans")]
    PreTranslate,
    #[serde(rename = "nopretrans")]
    NoPreTranslate,
}

impl PreTranslateMode {
    pub fn from_config_experiment(exp: &ToolExperiment) -> Self {
        match &exp.translate_mode {
            TranslateMode::Translated {
                language: _,
                option,
            } => match &option {
                TranslateOption::FullyTranslatedPreTranslate => PreTranslateMode::PreTranslate,
                TranslateOption::FullyTranslated => PreTranslateMode::NoPreTranslate,
                TranslateOption::PartiallyTranslated => PreTranslateMode::NoPreTranslate,
                TranslateOption::FullyTranslatedPromptTranslate => PreTranslateMode::NoPreTranslate,
                TranslateOption::FullyTranslatedPostTranslate => PreTranslateMode::NoPreTranslate,
            },
            TranslateMode::NotTranslated {} => PreTranslateMode::NoPreTranslate,
        }
    }
}
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub enum PostTranslateMode {
    #[serde(rename = "posttrans")]
    PostTranslate,
    #[serde(rename = "noposttrans")]
    NoPostTranslate,
}
impl PostTranslateMode {
    pub fn from_config_experiment(exp: &ToolExperiment) -> Self {
        match &exp.translate_mode {
            TranslateMode::Translated {
                language: _,
                option,
            } => match &option {
                TranslateOption::FullyTranslatedPostTranslate => PostTranslateMode::PostTranslate,
                TranslateOption::FullyTranslated => PostTranslateMode::NoPostTranslate,
                TranslateOption::PartiallyTranslated => PostTranslateMode::NoPostTranslate,
                TranslateOption::FullyTranslatedPreTranslate => PostTranslateMode::NoPostTranslate,
                TranslateOption::FullyTranslatedPromptTranslate => {
                    PostTranslateMode::NoPostTranslate
                }
            },
            TranslateMode::NotTranslated {} => PostTranslateMode::NoPostTranslate,
        }
    }
}
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct DatasetFileName(
    pub OriginalDataset,
    pub DatasetLanguage,
    pub TranslateLevel,
    pub AddNoiseMode,
);
impl DatasetFileName {
    pub fn from_config_experiment(exp: &ToolExperiment) -> Self {
        DatasetFileName(
            OriginalDataset::BfclV4Multiple,
            DatasetLanguage::from_config_experiment(exp),
            TranslateLevel::from_config_experiment(exp),
            AddNoiseMode::from_config_experiment(exp),
        )
    }
}
pub type PreTranslateFileName = DatasetFileName;
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct GenerateRawFileName(
    pub DatasetFileName,
    pub DatasetLanguage,
    pub TranslateLevel,
    pub PreTranslateMode,
    pub AddNoiseMode,
    pub PromptTranslateMode,
);
impl GenerateRawFileName {
    pub fn from_config_experiment(exp: &ToolExperiment) -> Self {
        GenerateRawFileName(
            DatasetFileName::from_config_experiment(exp),
            DatasetLanguage::from_config_experiment(exp),
            TranslateLevel::from_config_experiment(exp),
            PreTranslateMode::from_config_experiment(exp),
            AddNoiseMode::from_config_experiment(exp),
            PromptTranslateMode::from_config_experiment(exp),
        )
    }
}
pub type ParseOutputFileName = GenerateRawFileName;
pub type PostTranslateFileName = GenerateRawFileName;
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct EvaluateFileName(
    pub DatasetFileName,
    pub DatasetLanguage,
    pub TranslateLevel,
    pub PreTranslateMode,
    pub AddNoiseMode,
    pub PromptTranslateMode,
    pub PostTranslateMode,
);
impl EvaluateFileName {
    pub fn from_config_experiment(exp: &ToolExperiment) -> Self {
        EvaluateFileName(
            DatasetFileName::from_config_experiment(exp),
            DatasetLanguage::from_config_experiment(exp),
            TranslateLevel::from_config_experiment(exp),
            PreTranslateMode::from_config_experiment(exp),
            AddNoiseMode::from_config_experiment(exp),
            PromptTranslateMode::from_config_experiment(exp),
            PostTranslateMode::from_config_experiment(exp),
        )
    }
}
pub type CategorizeFileName = EvaluateFileName;
pub type StatisticsFileName = EvaluateFileName;
