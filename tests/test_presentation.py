import copy
import importlib.util
import json
from pathlib import Path

from askpatients.config import Config
from askpatients.dialogue.app_agent import DrAPP
from askpatients.dialogue.orchestrator import DialogueOrchestrator
from askpatients.llm.factory import build_llm
from askpatients.medical.knowledge_base import KnowledgeBase
from askpatients.patient.simulator import PatientSimulator
from askpatients.patient.store import get_profile
from askpatients.data.remedi_loader import load_remedi_raw
from askpatients.rag.lexical import LexicalRetriever

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoints_are_ordered_and_finalized(tmp_path):
    cfg = Config()
    llm = build_llm(cfg.llm)
    kb = KnowledgeBase.from_json_file(ROOT / 'data/knowledge_base/sample_diseases.json')
    case = load_remedi_raw(ROOT / 'data/remedi/synthetic_sample.jsonl')[0]
    patient = PatientSimulator(llm, get_profile(llm, case, tmp_path))
    app = DrAPP(llm, kb, LexicalRetriever(kb), cfg.algo)
    checkpoints = []
    result = DialogueOrchestrator(app, 2, lambda r: checkpoints.append(copy.deepcopy(r))).run(
        case.case_id, case.opening_doctor_turn(), patient, case.ground_truth_diagnosis)
    assert [len(r.turns) for r in checkpoints] == [0, 1, 2, 2]
    assert [r.status for r in checkpoints] == ['running', 'running', 'running', 'completed']
    assert checkpoints[-1].final_distribution == result.final_distribution


def test_report_escapes_model_text_and_excludes_config_secrets():
    spec = importlib.util.spec_from_file_location('report', ROOT / 'scripts/presentation_report.py')
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    record = {'config': {'api_key': 'DO_NOT_RENDER', 'llm': {'model': 'test'}},
              'result': {'case_id': 'x', 'opening_response': '<script>bad()</script>',
                         'turns': [], 'final_distribution': {}, 'entropy_trajectory': []}}
    rendered = report.render(record)
    assert '<script>bad()' not in rendered
    assert '&lt;script&gt;' in rendered
    assert 'DO_NOT_RENDER' not in rendered
    assert 'not an accuracy benchmark' in rendered
