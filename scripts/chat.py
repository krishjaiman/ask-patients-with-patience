"""Interactive research demo. Run with --mock for an offline plumbing check."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from _common import add_common_args, load_config, build_llm_from_config, build_knowledge_base, build_retriever
from askpatients.dialogue.app_agent import DrAPP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument('--out', default='outputs/chat/session.json')
    parser.add_argument('--turns', type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args)
    turns = args.turns if args.turns is not None else cfg.algo.max_dialogue_turns
    if turns < 0:
        parser.error('--turns must be nonnegative')
    print(f'Research demo | provider={cfg.llm.provider} | model={cfg.llm.model}')
    if cfg.llm.provider == 'mock':
        print('MOCK MODE: probabilities and dialogue are placeholders, not diagnostic results.')
    print('Enter /quit to stop. Use a fictional case for this demonstration.')
    opening = 'Hi, could you please tell me what has been bothering you today?'
    print('Dr.APP:', opening)
    complaint = input('Patient: ').strip()
    if not complaint or complaint == '/quit':
        return
    kb = build_knowledge_base(cfg, args.kb)
    app = DrAPP(build_llm_from_config(cfg), kb, build_retriever(cfg, kb), cfg.algo)
    state = app.initialize(opening, complaint)
    record = {'config': asdict(cfg), 'opening': complaint, 'turns': []}
    try:
        for _ in range(turns):
            print('Scoring candidate questions...')
            step = app.propose_next_question(state)
            print('Dr.APP:', step.human_question)
            print('Candidate scores (paper formula):')
            for q, score in step.question_entropies.items():
                print(f'  {score:.4f}  {q}')
            answer = input('Patient: ').strip()
            if answer == '/quit':
                break
            if not answer:
                answer = 'I do not know.'
            app.observe_response(state, step.human_question, answer)
            record['turns'].append({'selection': asdict(step), 'answer': answer,
                                    'posterior': dict(state.distribution)})
            print('Current disease estimates:', state.distribution)
    except (EOFError, KeyboardInterrupt):
        print('\nSession stopped.')
    finally:
        record['final_state'] = asdict(state)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Session and evidence saved to {path}')


if __name__ == '__main__':
    main()
