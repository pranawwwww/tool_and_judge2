use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Preference{
    pub preferred_answer: usize,
    pub logprob_signed_difference: f32,
    pub logprob1: f32,
    pub logprob2: f32,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct PreferenceResultEntry{
    pub index: usize,
    pub preference: Result<Preference, String>,
    pub question: String,
    pub answer1: String,
    pub answer2: String,
    pub lang1: String,
    pub lang2: String,
    pub is_correct1: bool,
    pub is_correct2: bool,
    pub subject: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct PerplexityResultEntry{
    pub index: usize,
    pub perplexity: Result<f32, String>,
    pub question: String,
    pub answer: String,
    pub lang: String,
    pub is_correct: bool,
    pub subject: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct ResponseEntry{
    pub index: usize,
    pub question: String,
    pub response: String,
    pub lang: String,
    pub subject: String,
}