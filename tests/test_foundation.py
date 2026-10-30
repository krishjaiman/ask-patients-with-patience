"""Regression checks for scientific validity and demo integration fixes."""
import json
from pathlib import Path
from dataclasses import asdict
import pytest
from pydantic import ValidationError

from askpatients.config import Config
from askpatients.data.remedi_loader import ReMeDiCase, ReMeDiTurn, _coerce_case, filter_diagnosis_only
from askpatients.medical.schema import ConditionalProbabilityResult, DiseaseProbabilityResult, PossibleResponsesResult, DiagnosisJudgeResult
from askpatients.llm.base import BaseLLM, LLMResponse, LLMMessage, NonRetryableError
from askpatients.llm.factory import build_llm
from askpatients.patient.profile import build_profile_from_case
from askpatients.patient.store import get_profile
from askpatients.medical.knowledge_base import KnowledgeBase
from askpatients.rag.lexical import LexicalRetriever
from askpatients.dialogue.app_agent import DrAPP
from askpatients.dialogue.orchestrator import DialogueOrchestrator
from askpatients.patient.simulator import PatientSimulator
from askpatients.evaluation.run_eval import EvaluationRunner

ROOT = Path(__file__).resolve().parents[1]


def sample_case():
    return ReMeDiCase('demo', '', 'Lactose Intolerance', [
        ReMeDiTurn('patient', 'I have bloating.'),
        ReMeDiTurn('doctor', 'Does dairy trigger it?', 'Diagnosis'),
        ReMeDiTurn('patient', 'Yes, especially after ice cream.')])


class QueueLLM(BaseLLM):
    def __init__(self, payloads):
        super().__init__(model='test', max_retries=1, retry_base_delay=0)
        self.payloads = iter(payloads)
        self.calls = 0

    def _complete(self, messages):
        self.calls += 1
        return LLMResponse(json.dumps(next(self.payloads)))


@pytest.mark.parametrize('value', [-1, 1.01, float('nan'), float('inf')])
def test_invalid_likelihoods_rejected(value):
    with pytest.raises(ValidationError):
        ConditionalProbabilityResult(probabilities={'A': value})


def test_missing_disease_triggers_structured_retry():
    llm = QueueLLM([{'probabilities': {'A': .5}}, {'probabilities': {'A': .5, 'B': .2}}])
    result = llm.complete_structured([], ConditionalProbabilityResult,
                                     validation_context={'diseases': ['A', 'B']})
    assert llm.calls == 2 and result.probabilities['B'] == .2


def test_zero_prior_and_invalid_judge_score_rejected():
    with pytest.raises(ValidationError):
        DiseaseProbabilityResult(probabilities={'A': 0})
    with pytest.raises(ValidationError):
        DiagnosisJudgeResult(score=1)


def test_duplicate_or_placeholder_response_padding_not_allowed():
    with pytest.raises(ValidationError):
        PossibleResponsesResult(responses=['Yes', ' yes ', ''])


def test_neutral_opening_does_not_replay_doctor_diagnosis():
    case = sample_case()
    case.turns[1].text = 'You have Lactose Intolerance.'
    assert 'Lactose' not in case.opening_doctor_turn()


def test_later_patient_fact_retained_without_invented_age(tmp_path):
    cfg = Config()
    llm = build_llm(cfg.llm)
    case = sample_case()
    first = get_profile(llm, case, tmp_path)
    assert 'ice cream' in first.as_persona_text()
    assert first.age is None
    assert case.ground_truth_diagnosis not in first.as_persona_text()
    llm.complete_structured = lambda *a, **kw: pytest.fail('Cached profile was regenerated')
    assert asdict(get_profile(llm, case, tmp_path)) == asdict(first)


def test_profile_rejects_untraceable_fact():
    llm = QueueLLM([{'clinical_facts': [{'category': 'symptom',
                        'statement': 'fever', 'evidence': 'I have a fever'}]}])
    with pytest.raises(ValueError, match='source quote'):
        build_profile_from_case(llm, '1', 'Patient: bloating', 'bloating', [], 'X')


def test_official_adapter_preserves_multiple_actions_and_ambiguous_labels():
    raw = {'dialogue': 42, 'information': [
        {'role': 'patient', 'sentence': 'complaint', 'actions': []},
        {'role': 'doctor', 'sentence': 'assessment', 'actions': [
            {'intent': 'Diagnosis', 'slot': 'disease', 'value1': 'A'},
            {'intent': 'Inquire', 'slot': 'symptom', 'value1': 'pain'}]}]}
    case = _coerce_case(raw)
    assert case.case_id == '42' and case.ground_truth_diagnosis == 'A'
    assert case.doctor_actions() == ['Diagnosis', 'Inquiring']
    assert filter_diagnosis_only([case]) == []
    assert filter_diagnosis_only([case], 'single_diagnosis') == [case]
    raw['information'][1]['actions'].append({'intent': 'Diagnosis', 'slot': 'disease', 'value1': 'B'})
    assert _coerce_case(raw).ground_truth_diagnosis == ''
    assert ReMeDiCase.from_dict(case.to_dict()) == case


def test_trace_retains_human_question_probabilities_and_evidence(tmp_path):
    cfg = Config()
    llm = build_llm(cfg.llm)
    kb = KnowledgeBase.from_json_file(ROOT / 'data/knowledge_base/sample_diseases.json')
    retriever = LexicalRetriever(kb)
    assert 'Lactose Intolerance' in {c.disease for c, _ in retriever.query('milk ice cream dairy bloating')}
    app = DrAPP(llm, kb, retriever, cfg.algo)
    case = sample_case()
    patient = PatientSimulator(llm, get_profile(llm, case, tmp_path))
    result = DialogueOrchestrator(app, 1).run(case.case_id, case.opening_doctor_turn(), patient, case.ground_truth_diagnosis)
    turn = result.turns[0]
    assert turn.question_evidence and turn.grounding
    assert abs(sum(turn.posterior_distribution.values()) - 1) < 1e-9
    state = app.initialize('Hello', 'bloating')
    app.observe_response(state, 'Plain question', 'I do not know')
    assert 'Doctor: Plain question' in state.dialogue_text


def test_evaluation_records_failure_and_does_not_score_it(tmp_path):
    cfg = Config()
    llm = build_llm(cfg.llm)
    kb = KnowledgeBase.from_json_file(ROOT / 'data/knowledge_base/sample_diseases.json')
    runner = EvaluationRunner(cfg, llm, llm, kb, LexicalRetriever(kb), profile_cache_dir=tmp_path)
    def fail(case):
        raise ValueError('bad provider output')
    runner.run_app = fail
    results = runner.run_all([sample_case()], ['app'])
    summary = runner.summarize(results)['app']
    assert results[0].status == 'failed'
    assert summary['n_failed'] == 1 and summary['mean_accuracy'] is None


def test_compatible_adapter_cache_and_shared_request_budget(monkeypatch, tmp_path):
    import io
    import askpatients.llm.compatible_llm as module
    module._USAGE.clear()
    calls = []
    def fake_open(req, **kwargs):
        calls.append(req)
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'reply'}, 'finish_reason': 'stop'}]}).encode())
    monkeypatch.setattr(module, 'urlopen', fake_open)
    a = module.CompatibleLLM(base_url='http://localhost:1234/v1', model='local',
                            max_api_calls=1, response_cache=str(tmp_path))
    messages = [LLMMessage('user', 'hello')]
    assert a.complete(messages).text == 'reply'
    b = module.CompatibleLLM(base_url=a.base_url, model='local', max_api_calls=1, response_cache=str(tmp_path))
    assert b.complete(messages).raw['cached']
    with pytest.raises(NonRetryableError, match='budget'):
        b.complete([LLMMessage('user', 'different')])
    assert len(calls) == 1
    assert calls[0].get_header('User-agent') == 'AskPatients/0.1'


def test_missing_config_does_not_silently_run_mock(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.load(str(tmp_path / 'missing.yaml'))


def test_empty_symptom_mapping_retrieves_complaint_not_file_order():
    from askpatients.medical.diagnosis import DiagnosisEngine
    kb = KnowledgeBase.from_json_file(ROOT / 'data/knowledge_base/sample_diseases.json')
    engine = DiagnosisEngine(None, kb, LexicalRetriever(kb), top_n_diseases=5)
    diseases = engine.candidate_disease_set([], "I've been having stomach cramps and bloating lately.")
    assert diseases[0] == 'Lactose Intolerance'
    assert 'Orthostatic Hypotension' not in diseases
    with pytest.raises(ValueError, match='patient evidence'):
        engine.candidate_disease_set([])
    with pytest.raises(ValueError, match='No relevant'):
        engine.candidate_disease_set(['zzzzzz'])


def test_patient_cannot_generate_unrecorded_denials(tmp_path):
    llm = build_llm(Config().llm)
    profile = get_profile(llm, sample_case(), tmp_path)
    selector = QueueLLM([{'fact_indices': [], 'unknown': True,
                         'answer': 'I do not have diarrhea.'},
                        {'fact_indices': [0], 'unknown': True}])
    patient = PatientSimulator(selector, profile)
    assert patient.answer('Do you have diarrhea?', '') == "I don't know."
    answer = patient.answer('Tell me about bloating and diarrhea.', '')
    assert profile.clinical_facts[0]['statement'] in answer
    assert 'diarrhea' not in answer
    assert "don't know" in answer


def test_patient_rejects_invented_fact_index():
    from askpatients.patient.simulator import PatientAnswerResult
    with pytest.raises(ValidationError):
        PatientAnswerResult.model_validate({'fact_indices': [2], 'unknown': False},
                                           context={'fact_count': 2})
