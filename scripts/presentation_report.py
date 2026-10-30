"""Render a saved APP run into an offline HTML presentation report. No API calls."""
import argparse
import html
import json
from pathlib import Path


def esc(value):
    return html.escape(str(value), quote=True)


def probabilities(distribution):
    rows = ''.join(f'<tr><td>{esc(name)}</td><td><meter min="0" max="1" value="{float(p):.8f}"></meter></td>'
                   f'<td>{float(p):.1%}</td></tr>'
                   for name, p in sorted(distribution.items(), key=lambda x: -x[1]))
    return '<table><thead><tr><th>Disease</th><th>Model estimate</th><th>Probability</th></tr></thead><tbody>' + rows + '</tbody></table>'


def render(record):
    result = record['result']
    cfg = record.get('config', {})
    algo = cfg.get('algo', {})
    llm = cfg.get('llm', {})
    turns = result.get('turns', [])
    parts = [f'<h1>Ask Patients with Patience</h1><p class="lead">A saved research demonstration of grounded question selection.</p>',
        f'<p>Case <b>{esc(result["case_id"])}</b> · Model <b>{esc(llm.get("model", "unknown"))}</b> · '
        f'Provider {esc(llm.get("provider", "unknown"))} · Status {esc(result.get("status", "completed"))}</p>',
        '<aside>This is one demonstration, not an accuracy benchmark or a validated medical diagnosis. '
        'The bundled knowledge base is illustrative; model probabilities are uncalibrated. '
        'Question scores use the paper’s printed formula, not standard expected information gain.</aside>',
        '<h2>Experiment settings</h2>',
        f'<p>{len(turns)} completed rounds · Up to {esc(algo.get("num_candidate_questions", "?"))} candidate questions · '
        f'{esc(algo.get("min_responses_per_question", "?"))}–{esc(algo.get("max_responses_per_question", "?"))} hypothetical answers · '
        f'Retrieval: {esc(cfg.get("rag", {}).get("embedding_provider", "unknown"))} · '
        f'Belief update: {esc(result.get("belief_update", "unknown"))}</p>',
        '<h2>Opening conversation</h2>',
        f'<p class="doctor"><b>Dr.APP:</b> {esc(result.get("opening_question", ""))}</p>',
        f'<p class="patient"><b>Patient:</b> {esc(result.get("opening_response", ""))}</p>']
    if llm.get('provider') == 'mock':
        parts.insert(1, '<aside><b>MOCK RUN: structural demonstration only. These estimates do not measure model quality.</b></aside>')
    if result.get('status', 'completed') != 'completed':
        parts.insert(1, '<aside><b>INCOMPLETE RUN: the result below is a partial checkpoint.</b></aside>')
    if 'elapsed_seconds' in record:
        usage = record.get('provider_usage', {})
        parts.insert(4, f'<p>Recorded runtime: {float(record["elapsed_seconds"]) / 60:.1f} minutes. '
                     f'HTTP attempts: {esc(usage.get("calls", "unavailable"))}; '
                     f'cached completions: {esc(usage.get("cache_hits", 0))}. '
                     'Timing excludes patient-profile preparation.</p>')
    if turns:
        parts += ['<h2>Initial disease estimates</h2>', probabilities(turns[0]['prior_distribution'])]
    for turn in turns:
        parts += [f'<section><h2>Round {esc(turn["turn"])}</h2>',
            f'<p class="doctor"><b>Dr.APP:</b> {esc(turn["human_question"])}</p>',
            f'<p class="patient"><b>Patient:</b> {esc(turn["patient_response"])}</p>',
            f'<p>Clinical question: {esc(turn["clinical_question"])}</p>',
            '<h3>Why this question was selected</h3><table><thead><tr><th>Candidate</th><th>Paper entropy score</th><th>Selection</th></tr></thead><tbody>']
        for question, evidence in turn.get('question_evidence', {}).items():
            selected = question == turn['clinical_question']
            parts.append(f'<tr class="{"selected" if selected else ""}"><td>{esc(question)}</td>'
                         f'<td>{evidence["paper_entropy"]:.6f}</td><td>{"Selected" if selected else ""}</td></tr>')
        parts += ['</tbody></table>',
            '<details><summary>Inspect imagined answers, likelihoods and virtual posteriors</summary><pre>'
            + esc(json.dumps(turn.get('question_evidence', {}), ensure_ascii=False, indent=2)) + '</pre></details>',
            '<h3>Disease estimates after the answer</h3>', probabilities(turn.get('posterior_distribution', {})),
            f'<p>Observed entropy: {float(turn["entropy_after"]):.4f} nats.</p>', '</section>']
    parts += ['<h2>Final result</h2>',
        f'<p class="lead">{esc(result.get("final_diagnosis", "Unavailable"))}</p>',
        f'<p>Reference label: {esc(result.get("ground_truth", "Unavailable"))}</p>',
        probabilities(result.get('final_distribution', {})),
        '<h3>Entropy across the conversation</h3><p>' + ' → '.join(f'{h:.4f}' for h in result.get('entropy_trajectory', [])) + '</p>',
        '<p>Lower entropy means greater concentration of the model’s estimates. It does not prove correctness.</p>',
        '<details><summary>Knowledge used and source metadata</summary><pre>' +
        esc(json.dumps(turns[0].get('grounding', []) if turns else [], ensure_ascii=False, indent=2)) + '</pre></details>',
        '<details><summary>Patient profile (evaluation-only; not supplied to the doctor)</summary><pre>' +
        esc(json.dumps(record.get('profile', {}), ensure_ascii=False, indent=2)) + '</pre></details>',
        '<p>Saved report: opening this file makes no model requests and requires no API key.</p>']
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>APP research demonstration</title>
<style>body{font:16px/1.65 system-ui,sans-serif;background:#f1f5f9;color:#172334;margin:0}main{max-width:1000px;margin:auto;padding:40px 28px;background:white}h1{font-size:36px;margin-bottom:0}h2{margin-top:32px;color:#153d5c}h3{font-size:18px}.lead{font-size:21px}aside{padding:18px;border-left:4px solid #c68b25;background:#fff6de}.doctor,.patient{padding:16px;border-radius:8px}.doctor{background:#e9f3fb}.patient{background:#eef6ee}table{width:100%;border-collapse:collapse;margin:16px 0}th,td{text-align:left;padding:10px;border-bottom:1px solid #dae3eb}th{background:#eef2f6}meter{width:100%;min-width:100px}.selected{background:#e0f3e9}details{margin:16px 0;padding:12px;border:1px solid #d8e2ec}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}section{border-top:2px solid #dde6ee;margin-top:28px}@media print{body{background:white}main{padding:0}section{break-inside:avoid}details{display:none}}@media(max-width:600px){main{padding:20px 12px}h1{font-size:28px}td,th{padding:6px;font-size:14px}}</style></head><body><main>''' + '\n'.join(parts) + '</main></body></html>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', default='outputs/presentation/report.html')
    args = parser.parse_args()
    record = json.loads(Path(args.input).read_text(encoding='utf-8'))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(record), encoding='utf-8')
    print(f'Offline presentation report: {output.resolve()}')


if __name__ == '__main__':
    main()
