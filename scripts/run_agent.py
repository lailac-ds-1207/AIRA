"""Command-line entry to run the RecSys research agent."""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from aira.graph import AgentConfig, ResearchAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RecSys research agent end-to-end")
    parser.add_argument("--pdf", nargs="+", type=Path, required=True, help="Path(s) to PDF papers or surveys")
    parser.add_argument("--objective", required=True, help="Business objective, e.g., retention uplift")
    parser.add_argument("--kpi", required=True, help="Primary KPI, e.g., NDCG@10 or latency budget")
    parser.add_argument("--data", required=True, help="Summary of available data and schema")
    parser.add_argument("--constraints", required=True, help="Infrastructure or cost constraints")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.pdf:
        raise SystemExit("--pdf 경로를 하나 이상 제공해주세요.")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY가 설정되지 않았습니다. .env나 환경 변수를 확인하세요.")
    config = AgentConfig(
        pdf_paths=args.pdf,
        business_objective=args.objective,
        kpi=args.kpi,
        data_description=args.data,
        constraints=args.constraints,
        model=args.model,
        temperature=args.temperature,
    )
    agent = ResearchAgent(config)
    result = agent.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
