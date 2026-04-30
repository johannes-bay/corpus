//! Cross-modal HTML renderer (Phase 2 — needs the web UI further along).
//!
//! Phase 1: emits a minimal self-contained HTML page that lists the members
//! grouped by domain. Useful as a quick-look composition export.

use std::path::{Path, PathBuf};

use crate::renderers::manifest::ManifestRenderer;
use crate::strategy::CompositionStrategy;
use crate::{ComposeError, Composition, PreviewData, Renderer, Result};

#[derive(Debug, Default, Clone)]
pub struct MixedRenderer;

impl Renderer for MixedRenderer {
    type Output = PathBuf;

    fn validate(&self, composition: &Composition) -> Result<()> {
        if composition.members.is_empty() {
            return Err(ComposeError::Empty);
        }
        match &composition.strategy {
            CompositionStrategy::CrossModal(_) => Ok(()),
            other => Err(ComposeError::DomainMismatch {
                strategy: other.kind().to_string(),
                domain: "mixed".to_string(),
            }),
        }
    }

    fn render(&self, composition: &Composition, output_dir: &Path) -> Result<PathBuf> {
        self.validate(composition)?;
        std::fs::create_dir_all(output_dir)?;

        let html = render_html(composition);
        let path = output_dir.join(format!("{}.html", composition.id));
        std::fs::write(&path, html)?;
        Ok(path)
    }

    fn preview(&self, composition: &Composition) -> Result<PreviewData> {
        let html = render_html(composition);
        Ok(PreviewData::Html(html))
    }
}

fn render_html(composition: &Composition) -> String {
    let mut body = String::new();
    body.push_str(&format!(
        "<!doctype html><html><head><meta charset='utf-8'><title>{}</title>\n\
         <style>body{{font:14px/1.4 sans-serif;background:#0e0e10;color:#e8e8ea;padding:24px}}\n\
         h1{{font-size:18px}} h2{{font-size:14px;margin-top:24px;color:#a8a8b0}}\n\
         .member{{padding:8px 0;border-bottom:1px solid #1f1f24}}\n\
         .role{{display:inline-block;font-size:11px;background:#23232a;padding:2px 6px;border-radius:3px;margin-right:8px}}</style>\n\
         </head><body>",
        html_escape(&composition.name),
    ));
    body.push_str(&format!(
        "<h1>{}</h1><p>Seed: <code>{}</code></p>\n",
        html_escape(&composition.name),
        html_escape(&composition.seed.path),
    ));
    body.push_str(&format!(
        "<p>Strategy: <code>{}</code></p>\n",
        composition.strategy.kind()
    ));

    let mut by_domain: std::collections::BTreeMap<String, Vec<&crate::CompositionMember>> =
        Default::default();
    for m in &composition.members {
        by_domain
            .entry(m.artifact.domain.clone())
            .or_default()
            .push(m);
    }
    for (domain, members) in &by_domain {
        body.push_str(&format!("<h2>{}</h2>\n", html_escape(domain)));
        for m in members {
            body.push_str(&format!(
                "<div class='member'><span class='role'>{}</span><code>{}</code> \
                 <span style='color:#888'>score {:.2}</span><br><small>{}</small></div>\n",
                m.role.as_str(),
                html_escape(&m.artifact.path),
                m.match_score,
                html_escape(&m.match_explanation),
            ));
        }
    }

    // Tack on the textual manifest at the bottom for completeness.
    body.push_str("<h2>Manifest</h2><pre>");
    body.push_str(&html_escape(&ManifestRenderer::new().render_text(composition)));
    body.push_str("</pre></body></html>");
    body
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
