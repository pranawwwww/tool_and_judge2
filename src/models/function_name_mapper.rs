use std::collections::HashMap;

#[derive(Debug)]
pub struct FunctionNameMapper {
    pub original_to_sanitized: HashMap<String, String>,
    pub sanitized_to_original: HashMap<String, String>,
}
