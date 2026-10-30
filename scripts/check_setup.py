"""Check demo readiness without making API calls or printing credentials."""
import argparse
import importlib.util
import os

from _common import add_common_args, load_config, build_knowledge_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args)
    missing = []
    for role in ['llm', 'profile_llm', 'patient_llm', 'judge_llm']:
        model = getattr(cfg, role) or cfg.llm
        print(f'{role}: {model.provider} / {model.model}')
        env = {'openai': 'OPENAI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY'}.get(model.provider)
        if model.provider == 'compatible':
            from urllib.parse import urlparse
            if not model.base_url:
                missing.append(f'{role}: base_url')
            elif urlparse(model.base_url).hostname not in {'localhost', '127.0.0.1', '::1'}:
                env = model.api_key_env
        if env and not os.environ.get(env):
            missing.append(env)
        package = {'openai': 'openai', 'anthropic': 'anthropic'}.get(model.provider)
        if package and not importlib.util.find_spec(package):
            missing.append(f'Python package: {package}')
    kb = build_knowledge_base(cfg, args.kb)
    print(f'Knowledge base: {len(kb.entries)} diseases; {sum(bool(e.source_url) for e in kb.entries)} source URLs')
    print(f'Retrieval: {cfg.rag.embedding_provider}; belief update: {cfg.algo.belief_update}')
    print(f'Demo: {cfg.algo.max_dialogue_turns} rounds, {cfg.algo.num_candidate_questions} candidate questions')
    if missing:
        raise SystemExit('Missing: ' + ', '.join(sorted(set(missing))))
    print('Configuration prerequisites present. Live model quality and access are not yet verified.')


if __name__ == '__main__':
    main()
