#[derive(Clone)]
pub struct BfclDatasetEntry {
    pub id: String,
    pub question_content: String,
    pub functions: Vec<BfclFunctionDef>,
    pub raw_entry: serde_json::Value,
}

impl BfclDatasetEntry {
    pub fn try_from(raw_entry: serde_json::Value) -> Result<Self, String> {
        let id = raw_entry
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid 'id' field")?
            .to_string();

        let question_content = raw_entry
            .get("question")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid 'question[0][0].content' field")?
            .to_string();

        let functions_array = raw_entry
            .get("function")
            .and_then(|v| v.as_array())
            .ok_or("Missing or invalid 'function' field")?;

        let mut functions = Vec::new();
        for func_val in functions_array {
            let name = func_val
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid 'name' field in function")?
                .to_string();

            let description = func_val
                .get("description")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid 'description' field in function")?
                .to_string();

            let params_obj = func_val
                .get("parameters")
                .and_then(|v| v.get("properties"))
                .and_then(|v| v.as_object())
                .ok_or("Missing or invalid 'parameters.properties' field")?;
            
            let required_array = func_val
                .get("parameters")
                .and_then(|v| v.get("required"))
                .and_then(|v| v.as_array())
                .ok_or("Missing or invalid 'parameters.required' field")?;

            let required = required_array
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<String>>();

            let mut parameters = Vec::new();
            for (param_name, param_val) in params_obj {
                let param_type = param_val
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing or invalid 'type' field in parameter")?
                    .to_string();

                let param_description = param_val
                    .get("description")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing or invalid 'description' field in parameter")?
                    .to_string();

                parameters.push(BfclParameter {
                    name: param_name.clone(),
                    ty: param_type,
                    description: param_description,
                });
            }

            functions.push(BfclFunctionDef {
                name,
                description,
                parameters,
                required,
            });
        }

        Ok(BfclDatasetEntry {
            id,
            question_content,
            functions,
            raw_entry,
        })
    }
    pub fn modify_question_content(&self, new_content: &str) -> Result<serde_json::Value, String> {
        let mut raw_entry = self.raw_entry.clone();
        let question_array = raw_entry
            .get_mut("question")
            .and_then(|v| v.as_array_mut())
            .ok_or("Missing or invalid 'question' field")?;
        let first_question = question_array
            .get_mut(0)
            .and_then(|v| v.as_array_mut())
            .ok_or("Missing or invalid 'question[0]' field")?;
        let first_content = first_question
            .get_mut(0)
            .and_then(|v| v.as_object_mut())
            .ok_or("Missing or invalid 'question[0][0]' field")?;
        first_content.insert(
            "content".to_string(),
            serde_json::Value::String(new_content.to_string()),
        );
        Ok(raw_entry)
    }
}

#[derive(Clone)]
pub struct BfclFunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: Vec<BfclParameter>,
    pub required: Vec<String>,
}

#[derive(Clone)]
pub struct BfclParameter {
    pub name: String,
    pub ty: String,
    pub description: String,
}
