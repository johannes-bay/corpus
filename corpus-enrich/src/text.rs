use std::process::Command;

use anyhow::Result;
use tracing::warn;

#[derive(Debug, Default)]
pub struct TextAnalysis {
    /// Number of pages (PDF only)
    pub page_count: Option<i64>,
    /// Word count (from extracted text)
    pub word_count: Option<i64>,
    /// Character count
    pub char_count: Option<i64>,
    /// Detected language (ISO 639-1, e.g. "en", "de")
    pub language: Option<String>,
    /// Title from document metadata
    pub title: Option<String>,
    /// Author from document metadata
    pub author: Option<String>,
}

/// Analyze a document file. Dispatches based on extension.
pub fn analyze(path: &str, ext: &str) -> TextAnalysis {
    match analyze_inner(path, ext) {
        Ok(a) => a,
        Err(e) => {
            warn!("Text analysis failed for {path}: {e}");
            TextAnalysis::default()
        }
    }
}

fn analyze_inner(path: &str, ext: &str) -> Result<TextAnalysis> {
    match ext {
        ".pdf" => analyze_pdf(path),
        ".txt" | ".rtf" | ".md" => analyze_plaintext(path),
        ".docx" | ".doc" | ".pages" | ".epub" | ".mobi" => analyze_via_textutil(path),
        _ => Ok(TextAnalysis::default()),
    }
}

/// Extract metadata and text from a PDF using poppler's pdfinfo and pdftotext.
fn analyze_pdf(path: &str) -> Result<TextAnalysis> {
    let mut analysis = TextAnalysis::default();

    // pdfinfo for metadata
    if let Ok(output) = Command::new("pdfinfo").arg(path).output()
        && output.status.success()
    {
        let info = String::from_utf8_lossy(&output.stdout);
        for line in info.lines() {
            if let Some(val) = line.strip_prefix("Pages:") {
                analysis.page_count = val.trim().parse().ok();
            } else if let Some(val) = line.strip_prefix("Title:") {
                let v = val.trim().to_string();
                if !v.is_empty() {
                    analysis.title = Some(v);
                }
            } else if let Some(val) = line.strip_prefix("Author:") {
                let v = val.trim().to_string();
                if !v.is_empty() {
                    analysis.author = Some(v);
                }
            }
        }
    }

    // pdftotext for word/char count and language detection
    if let Ok(output) = Command::new("pdftotext")
        .args(["-l", "5", path, "-"])
        .output()
        && output.status.success()
    {
        let text = String::from_utf8_lossy(&output.stdout);
        let text = text.as_ref();
        if !text.is_empty() {
            analysis.char_count = Some(text.chars().count() as i64);
            analysis.word_count = Some(count_words(text) as i64);
            analysis.language = detect_language(text);
        }
    }

    Ok(analysis)
}

/// Analyze a plain text file directly.
fn analyze_plaintext(path: &str) -> Result<TextAnalysis> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Ok(TextAnalysis::default()),
    };
    if text.is_empty() {
        return Ok(TextAnalysis::default());
    }

    // Cap at first 50KB for language detection
    let sample = safe_truncate(&text, 50_000);

    Ok(TextAnalysis {
        page_count: None,
        word_count: Some(count_words(&text) as i64),
        char_count: Some(text.chars().count() as i64),
        language: detect_language(sample),
        title: None,
        author: None,
    })
}

/// Use macOS textutil to convert doc/docx/pages/epub to plain text, then analyze.
fn analyze_via_textutil(path: &str) -> Result<TextAnalysis> {
    let output = Command::new("textutil")
        .args(["-convert", "txt", "-stdout", path])
        .output();

    let text = match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => return Ok(TextAnalysis::default()),
    };

    if text.is_empty() {
        return Ok(TextAnalysis::default());
    }

    let sample = safe_truncate(&text, 50_000);

    Ok(TextAnalysis {
        page_count: None,
        word_count: Some(count_words(&text) as i64),
        char_count: Some(text.chars().count() as i64),
        language: detect_language(sample),
        title: None,
        author: None,
    })
}

/// Truncate a string at the nearest char boundary at or before `max_bytes`.
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Simple language detection based on common word frequencies.
/// Only distinguishes English vs German vs a few others — good enough for scoring.
fn detect_language(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().take(500).collect();
    if words.len() < 10 {
        return None;
    }

    let en_words = ["the", "and", "is", "in", "to", "of", "a", "that", "it", "for", "was", "with", "as", "this", "but", "not", "are", "from"];
    let de_words = ["der", "die", "und", "das", "ist", "ein", "eine", "für", "mit", "den", "von", "auf", "nicht", "sich", "auch", "ich", "des", "dem"];
    let fr_words = ["le", "la", "les", "de", "des", "est", "une", "que", "dans", "pour", "pas", "qui", "sur", "avec", "son", "mais", "nous", "cette"];
    let es_words = ["el", "la", "los", "las", "de", "que", "en", "por", "con", "una", "del", "para", "como", "pero", "más", "ser", "este", "fue"];

    let total = words.len() as f64;
    let en_score = words.iter().filter(|w| en_words.contains(w)).count() as f64 / total;
    let de_score = words.iter().filter(|w| de_words.contains(w)).count() as f64 / total;
    let fr_score = words.iter().filter(|w| fr_words.contains(w)).count() as f64 / total;
    let es_score = words.iter().filter(|w| es_words.contains(w)).count() as f64 / total;

    let max = en_score.max(de_score).max(fr_score).max(es_score);
    if max < 0.02 {
        return None;
    }

    if max == en_score {
        Some("en".to_string())
    } else if max == de_score {
        Some("de".to_string())
    } else if max == fr_score {
        Some("fr".to_string())
    } else {
        Some("es".to_string())
    }
}
