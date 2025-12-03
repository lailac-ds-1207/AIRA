from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, TypedDict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.runnables import Runnable  # type: ignore


@dataclass
class AgentConfig:
    """Configuration for the research agent run."""

    pdf_paths: List[Path]
    business_objective: str
    kpi: str
    data_description: str
    constraints: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_input_tokens: int = 4_000  # summarization / taxonomy 입력 토큰 상한 (대략치)


class AgentState(TypedDict):
    """Shared state passed between graph nodes."""

    documents: List[Document]
    paper_taxonomy: Dict[str, Any]
    roadmap: Dict[str, Any]


class ResearchAgent:
    """
    Orchestrates the multi-step research pipeline using LangGraph.

    v0.1 구조:
    1) build_taxonomy  : 논문 내용을 기반으로 paper_taxonomy (문제 축/모델/데이터셋/미래연구) 생성
    2) plan_roadmap    : paper_taxonomy + business config 를 이용해 paper-grounded roadmap 생성
       - roadmap 안에 legacy_outputs(research_summary / requirements / architecture / experiments)도 포함
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm: Runnable = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
        )
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the graph and return the final compiled plan."""
        initial_state: AgentState = {
            "documents": self._load_documents(self.config.pdf_paths),
            "paper_taxonomy": {},
            "roadmap": {},
        }
        app = self.graph.compile()
        final_state: AgentState = app.invoke(initial_state)

        config_dict = {
            "objective": self.config.business_objective,
            "kpi": self.config.kpi,
            "data_description": self.config.data_description,
            "constraints": self.config.constraints,
            "model": self.config.model,
            "temperature": self.config.temperature,
        }

        roadmap = final_state["roadmap"] or {}
        legacy_outputs = roadmap.get("legacy_outputs", {})

        # JSON 직렬화 가능한 형태로만 반환
        return {
            "config": config_dict,
            "paper_taxonomy": final_state["paper_taxonomy"],
            "roadmap": roadmap,
            # 기존 구조와 최대한 호환되도록 legacy_outputs도 outputs에 매핑
            "outputs": {
                "research_summary": legacy_outputs.get("research_summary", ""),
                "requirements_analysis": legacy_outputs.get("requirements_analysis", ""),
                "architecture": legacy_outputs.get("architecture", ""),
                "experiments": legacy_outputs.get("experiments", ""),
            },
        }

    # ------------------------------------------------------------------
    # Graph definition
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("build_taxonomy", self._build_taxonomy)
        graph.add_node("plan_roadmap", self._plan_roadmap)

        graph.set_entry_point("build_taxonomy")
        graph.add_edge("build_taxonomy", "plan_roadmap")
        graph.add_edge("plan_roadmap", END)

        return graph

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def _load_documents(self, pdf_paths: Iterable[Path]) -> List[Document]:
        """Load & split PDF into chunks."""
        documents: List[Document] = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
        )
        validated_paths = self._validate_paths(pdf_paths)
        print("[DEBUG] Validated PDF paths:", [str(p) for p in validated_paths])

        for path in validated_paths:
            loader = PyPDFLoader(str(path))
            for doc in loader.load():
                documents.extend(splitter.split_documents([doc]))

        print("[DEBUG] Loaded chunks:", len(documents))
        return documents

    def _validate_paths(self, pdf_paths: Iterable[Path]) -> Sequence[Path]:
        validated: List[Path] = []
        for path in pdf_paths:
            if not path.exists():
                raise FileNotFoundError(f"PDF not found: {path}")
            if not path.is_file():
                raise ValueError(f"Path is not a file: {path}")
            validated.append(path)
        if not validated:
            raise ValueError("At least one PDF path must be provided")
        return validated

    # ------------------------------------------------------------------
    # Helper: JSON 파서 (```json ... ``` 래핑 제거)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_content(raw: str) -> Dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            # ```json ... ``` 혹은 ``` ... ``` 형태 제거
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        return json.loads(text)

    # ------------------------------------------------------------------
    # Node 1: build_taxonomy
    # ------------------------------------------------------------------

    def _build_taxonomy(self, state: AgentState) -> AgentState:
        """
        논문 텍스트를 기반으로 paper_taxonomy를 생성.
        - meta (title, year, domain, primary_task, summary)
        - problem_axes (data_modeling / encoding / training.main / training.auxiliary)
        - model_catalog (각 모델별 축 정보, 복잡도, 노트)
        - datasets
        - future_directions
        """
        docs = state["documents"]
        if not docs:
            raise ValueError("No documents available for taxonomy building")

        # 대략적인 토큰 상한을 문자 길이로 제어 (4 chars ≒ 1 token heuristic)
        char_limit = self.config.max_input_tokens * 4
        selected_texts: List[str] = []
        current_chars = 0

        for d in docs:
            c = d.page_content
            if current_chars + len(c) > char_limit:
                break
            selected_texts.append(c)
            current_chars += len(c)

        combined_text = "\n\n".join(selected_texts)
        print("[DEBUG] first 400 chars from doc input:", combined_text[:400])

        system_prompt = (
            "You are an expert research assistant for recommender systems, "
            "specialized in multi-behavior and modern recommendation.\n\n"
            "You will receive the full text of a single survey or algorithm paper, "
            "already chunked and concatenated.\n\n"
            "Your job:\n"
            "1) Read ONLY this paper and build a structured taxonomy of its content.\n"
            "2) Do NOT invent algorithms or datasets that are not mentioned in the paper.\n"
            "3) Focus on:\n"
            "   - problem definition\n"
            "   - main axes / taxonomy of methods\n"
            "   - catalog of representative models\n"
            "   - datasets\n"
            "   - future directions / open challenges\n\n"
            "Output format:\n"
            "Return a SINGLE valid JSON object with the following top-level keys:\n\n"
            "{\n"
            '  "meta": {\n'
            '    "title": str,\n'
            '    "year": int | null,\n'
            '    "domain": [str, ...],\n'
            '    "primary_task": str,\n'
            '    "summary": str\n'
            "  },\n"
            '  "problem_axes": {\n'
            '    "data_modeling": [str, ...],\n'
            '    "encoding": [str, ...],\n'
            '    "training": {\n'
            '      "main": [str, ...],\n'
            '      "auxiliary": [str, ...]\n'
            "    }\n"
            "  },\n"
            '  "model_catalog": [\n'
            "    {\n"
            '      "name": str,\n'
            '      "year": int | null,\n'
            '      "type": [str, ...],\n'
            '      "data_modeling": [str, ...],\n'
            '      "encoding": [str, ...],\n'
            '      "training": {\n'
            '        "main": str | null,\n'
            '        "auxiliary": [str, ...]\n'
            "      },\n"
            '      "complexity": "low" | "medium" | "high" | null,\n'
            '      "notes": str\n'
            "    }\n"
            "  ],\n"
            '  "datasets": [\n'
            "    {\n"
            '      "name": str,\n'
            '      "behaviors": [str, ...],\n'
            '      "target_behavior": str | null,\n'
            '      "size_interactions": float | int | null,\n'
            '      "notes": str\n'
            "    }\n"
            "  ],\n"
            '  "future_directions": [str, ...]\n'
            "}\n\n"
            "Constraints:\n"
            "- All model names, dataset names, and axes MUST come from the given paper.\n"
            "- The problem_axes should reflect how this paper organizes the space.\n"
            "- JSON 키 이름과 축 이름, 모델/데이터셋 이름은 논문에 나온 영어 표기를 그대로 사용하십시오.\n"
            "- 하지만 \"summary\", \"notes\", \"future_directions\" 같은 자유 텍스트 값은 모두 **한국어로** 작성하십시오.\n"
            "- 응답은 JSON 한 개만 포함해야 하며, JSON 이외의 설명은 쓰지 마십시오."
        )


        user_content = (
            "Below is the concatenated text of the paper.\n"
            "Read it carefully and construct the requested JSON taxonomy.\n\n"
            "=== PAPER TEXT BEGIN ===\n"
            f"{combined_text}\n"
            "=== PAPER TEXT END ===\n"
        )

        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        resp: AIMessage = self.llm.invoke(messages)  # type: ignore[arg-type]
        taxonomy = self._parse_json_content(resp.content)

        return {**state, "paper_taxonomy": taxonomy}

    # ------------------------------------------------------------------
    # Node 2: plan_roadmap
    # ------------------------------------------------------------------

    def _plan_roadmap(self, state: AgentState) -> AgentState:
        """
        paper_taxonomy + business config를 기반으로
        - approach_selection (axes / candidate_models / justification)
        - experiment_plan (offline / online)
        - architecture_guidance (offline / realtime / data_notes)
        - legacy_outputs (기존 text 기반 요약 4개 섹션)
        을 생성.
        """
        taxonomy = state["paper_taxonomy"]
        if not taxonomy:
            raise ValueError("paper_taxonomy is empty; build_taxonomy must run first.")

        config_json = json.dumps(
            {
                "objective": self.config.business_objective,
                "kpi": self.config.kpi,
                "data_description": self.config.data_description,
                "constraints": self.config.constraints,
            },
            ensure_ascii=False,
            indent=2,
        )
        taxonomy_json = json.dumps(taxonomy, ensure_ascii=False, indent=2)

        system_prompt = (
            "You are an expert RecSys architect.\n\n"
            "You will receive:\n"
            "- A structured taxonomy of a paper (paper_taxonomy)\n"
            "- A business & data config (config) for a real-world recommender system\n\n"
            "Your job:\n"
            "- Propose paper-grounded modeling roadmaps and architectures.\n"
            "- You MUST explicitly use the axes and model names from paper_taxonomy.\n"
            "- Do NOT propose generic CF/CBF/DL descriptions without tying them back to\n"
            "  the paper's taxonomy.\n\n"
            "Output: A SINGLE valid JSON object with the following structure:\n"
            "...\n"
            "Constraints:\n"
            "- recommended_axes.* values must be taken from paper_taxonomy.problem_axes.\n"
            "- reference_models must be a subset of paper_taxonomy.model_catalog[*].name.\n"
            "- When proposing pipelines, explicitly mention how multi-behavior information\n"
            "  is used (e.g., view-specific graphs, view-unified sequences, auxiliary losses).\n"
            "- Connect every major design choice to something in the paper_taxonomy\n"
            "  (either an axis, a model, a dataset, or a future direction).\n"
            "- JSON 키, 축 이름, 모델/데이터셋 이름 등은 영어로 유지하십시오.\n"
            "- 하지만 \"justification\", \"notes_from_paper\", \"architecture_guidance\" 내 문장들, "
            "  그리고 \"legacy_outputs\"(research_summary, requirements_analysis, architecture, experiments) "
            "  필드는 모두 **한국어 문장**으로 작성하십시오.\n"
            "- 응답은 JSON 한 개만 포함해야 하며, JSON 이외의 설명은 쓰지 마십시오."
        )

        user_content = (
            "You are given the following JSON inputs.\n"
            "- config: business requirements and system constraints\n"
            "- paper_taxonomy: structured summary of a research paper\n\n"
            "Use them to construct a paper-grounded roadmap.\n\n"
            "=== CONFIG JSON ===\n"
            f"{config_json}\n\n"
            "=== PAPER_TAXONOMY JSON ===\n"
            f"{taxonomy_json}\n"
        )

        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        resp: AIMessage = self.llm.invoke(messages)  # type: ignore[arg-type]
        roadmap = self._parse_json_content(resp.content)

        return {**state, "roadmap": roadmap}


__all__ = ["AgentConfig", "ResearchAgent", "AgentState"]
