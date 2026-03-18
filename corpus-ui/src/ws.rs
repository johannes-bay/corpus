use std::collections::HashMap;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

use corpus_associate::matcher;
use corpus_associate::{AxisRegistry, WeightedAxis};

use crate::server::AppState;

#[derive(Deserialize)]
struct LiveComposeRequest {
    seed: String,
    axes: HashMap<String, f64>,
    count: Option<usize>,
}

#[derive(Serialize)]
struct LiveMatchResult {
    path: String,
    filename: String,
    score: f64,
    explanation: Vec<String>,
}

#[derive(Serialize)]
struct LiveComposeResponse {
    matches: Vec<LiveMatchResult>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

pub async fn compose_live(
    State(state): State<AppState>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(mut socket: WebSocket, state: AppState) {
    while let Some(Ok(msg)) = socket.recv().await {
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };

        let req: LiveComposeRequest = match serde_json::from_str(&text) {
            Ok(r) => r,
            Err(e) => {
                let err = serde_json::to_string(&ErrorResponse {
                    error: format!("Invalid request: {e}"),
                })
                .unwrap_or_default();
                let _ = socket.send(Message::Text(err.into())).await;
                continue;
            }
        };

        let count = req.count.unwrap_or(10).min(100);
        let registry = AxisRegistry::new();

        let mut weighted: Vec<WeightedAxis> = Vec::new();
        for (name, weight) in &req.axes {
            if let Some(axis) = registry.get(name) {
                weighted.push(WeightedAxis {
                    axis,
                    weight: *weight,
                });
            }
        }

        // All DB work in a sync block so MutexGuard never crosses an .await
        let compose_result: Result<Vec<_>, String> = if weighted.is_empty() {
            Ok(Vec::new())
        } else {
            match state.db.lock() {
                Ok(db) => Ok(
                    matcher::find_matches(&db, &req.seed, &weighted, count)
                        .unwrap_or_default(),
                ),
                Err(e) => Err(format!("db lock: {e}")),
            }
        };

        let response = match compose_result {
            Err(err_msg) => {
                let err = serde_json::to_string(&ErrorResponse { error: err_msg })
                    .unwrap_or_default();
                let _ = socket.send(Message::Text(err.into())).await;
                continue;
            }
            Ok(scored) => LiveComposeResponse {
                matches: scored
                    .into_iter()
                    .map(|m| LiveMatchResult {
                        path: m.file.path.clone(),
                        filename: m.file.filename.clone(),
                        score: m.total_score,
                        explanation: m
                            .explanation
                            .axis_details
                            .iter()
                            .map(|(name, score, detail)| {
                                format!("{name}: {score:.2} ({detail})")
                            })
                            .collect(),
                    })
                    .collect(),
            },
        };

        let json = serde_json::to_string(&response).unwrap_or_default();
        if socket.send(Message::Text(json.into())).await.is_err() {
            break;
        }
    }
}
