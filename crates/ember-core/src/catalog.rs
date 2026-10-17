use crate::{Error, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, path::Path};

pub struct Catalog {
    pub version: String,
    problems: HashMap<String, Value>,
    order: Vec<String>,
    starters: Value,
    solutions: Value,
    paths: Vec<Value>,
}
impl Catalog {
    pub fn load(root: &Path) -> Result<Self> {
        let bytes = fs::read(root.join("problems.json"))?;
        let data: Value = serde_json::from_slice(&bytes)?;
        let list = data["problems"]
            .as_array()
            .ok_or_else(|| Error::Invalid("Missing problem catalog".into()))?;
        let order = list
            .iter()
            .map(|p| p["id"].as_str().unwrap_or_default().to_owned())
            .collect();
        let problems = list
            .iter()
            .map(|p| (p["id"].as_str().unwrap_or_default().to_owned(), p.clone()))
            .collect();
        let load =
            |name| -> Result<Value> { Ok(serde_json::from_slice(&fs::read(root.join(name))?)?) };
        Ok(Self {
            version: hex::encode(Sha256::digest(&bytes)),
            problems,
            order,
            starters: load("starters.json")?,
            solutions: load("solutions.json")?,
            paths: load("paths.json")?["paths"]
                .as_array()
                .cloned()
                .unwrap_or_default(),
        })
    }
    pub fn problem(&self, id: &str) -> Result<Value> {
        let mut problem = self
            .problems
            .get(id)
            .cloned()
            .ok_or_else(|| Error::NotFound("Problem not found".into()))?;
        problem["starterCode"]=json!(format!("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport numpy as np\n\n{}",self.starters[id].as_str().unwrap_or("")));
        Ok(problem)
    }
    pub fn stats(&self) -> Value {
        let count = |difficulty: &str| {
            self.problems
                .values()
                .filter(|p| p["difficulty"] == difficulty)
                .count()
        };
        json!({"total":self.problems.len(),"easy":count("Easy"),"medium":count("Medium"),"hard":count("Hard")})
    }
    pub fn list(&self) -> Value {
        let problems: Vec<Value> = self
            .order
            .iter()
            .map(|id| {
                let p = &self.problems[id];
                let mut summary = serde_json::Map::new();
                for key in [
                    "id",
                    "title",
                    "titleZh",
                    "difficulty",
                    "functionName",
                    "hint",
                    "hintZh",
                    "descriptionEn",
                    "descriptionZh",
                    "category",
                ] {
                    summary.insert(key.into(), p[key].clone());
                }
                Value::Object(summary)
            })
            .collect();
        json!({"total":problems.len(),"problems":problems})
    }
    pub fn solution(&self, id: &str) -> Result<Value> {
        self.solutions
            .get(id)
            .cloned()
            .ok_or_else(|| Error::NotFound("Solution not found".into()))
    }
    pub fn paths(&self, progress: &Value) -> Value {
        json!({"paths":self.paths.iter().map(|p|self.enrich_path(p,progress,false)).collect::<Vec<_>>()})
    }
    pub fn path(&self, id: &str, progress: &Value) -> Result<Value> {
        self.paths
            .iter()
            .find(|p| p["id"] == id)
            .map(|p| self.enrich_path(p, progress, true))
            .ok_or_else(|| Error::NotFound("Path not found".into()))
    }
    fn enrich_path(&self, path: &Value, progress: &Value, detail: bool) -> Value {
        let mut result = path.clone();
        let ids = path["problems"].as_array().cloned().unwrap_or_default();
        result["total"] = json!(ids.len());
        result["solved"] = json!(ids
            .iter()
            .filter(|id| progress[id.as_str().unwrap_or_default()]["status"] == "solved")
            .count());
        if detail {
            result["problems"]=json!(ids.iter().map(|id| {let key=id.as_str().unwrap_or_default(); let p=self.problems.get(key).cloned().unwrap_or_default();json!({"id":id,"title":p["title"],"titleZh":p["titleZh"],"difficulty":p["difficulty"],"status":progress[key]["status"].as_str().unwrap_or("todo")})}).collect::<Vec<_>>());
        }
        result
    }
}
