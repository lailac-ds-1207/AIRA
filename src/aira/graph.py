"""LangGraph-based research and recommendation agent for RecSys architecture work."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, TypedDict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


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
    max_input_tokens: int = 4_000


class AgentState(TypedDict):
    """Shared state passed between graph nodes."""

    documents: List[Document]
    research_summary: str
    requirements_analysis: str
    architecture: str
    experiments: str


class ResearchAgent:
    """Orchestrates the multi-step research pipeline using LangGraph."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm: Runnable = ChatOpenAI(model=config.model, temperature=config.temperature)
        self.graph = self._build_graph()

    def run(self) -> Dict[str, Any]:
        """Execute the graph and return the final compiled plan."""
        initial_state: AgentState = {
            "documents": self._load_documents(self.config.pdf_paths),
            "research_summary": "",
            "requirements_analysis": "",
            "architecture": "",
            "experiments": "",
        }
        app = self.graph.compile()
        return app.invoke(initial_state)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("summarize", self._summarize_documents)
        graph.add_node("align_requirements", self._align_requirements)
        graph.add_node("design_architecture", self._design_architecture)
        graph.add_node("plan_experiments", self._plan_experiments)

        graph.set_entry_point("summarize")
        graph.add_edge("summarize", "align_requirements")
        graph.add_edge("align_requirements", "design_architecture")
        graph.add_edge("design_architecture", "plan_experiments")
        graph.add_edge("plan_experiments", END)
        return graph

    def _load_documents(self, pdf_paths: Iterable[Path]) -> List[Document]:
        documents: List[Document] = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        validated_paths = self._validate_paths(pdf_paths)
        for path in validated_paths:
            loader = PyPDFLoader(str(path))
            for doc in loader.load():
                documents.extend(splitter.split_documents([doc]))
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

    def _summarize_documents(self, state: AgentState) -> AgentState:
        if not state["documents"]:
            raise ValueError("No documents available for summarization")

        selected_chunks = state["documents"][:15]
        document_text = "\n\n".join(doc.page_content for doc in selected_chunks)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    "You are a research assistant that distills RecSys papers into structured insights. "
                    "Summarize motivation, methods, key results, and deployment notes with concise bullets."
                ),
                HumanMessage(
                    "Summarize the following document chunks. Respond in Korean with section headers: "
                    "Motivation, Method, Results, Deployment. Limit to 12 bullets total."
                ),
                HumanMessage("{documents}"),
            ]
        )
        messages: List[BaseMessage] = prompt.format_messages(documents=document_text)
        summary: AIMessage = self.llm.invoke(messages)  # type: ignore[arg-type]
        return {**state, "research_summary": summary.content}

    def _align_requirements(self, state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    "You align research insights with business goals and data constraints for recommender systems."
                ),
                HumanMessage(
                    "Given the business objective, KPI, data description, constraints, and research summary, "
                    "produce a numbered plan covering: (1) KPI-to-approach mapping, (2) data readiness and gaps, "
                    "(3) algorithm candidates with rationale, and (4) latency/cost considerations."
                ),
                HumanMessage(
                    "Business objective: {objective}\nKPI: {kpi}\nData: {data}\nConstraints: {constraints}\n"
                    "Research summary:\n{summary}"
                ),
            ]
        )
        messages: List[BaseMessage] = prompt.format_messages(
            objective=self.config.business_objective,
            kpi=self.config.kpi,
            data=self.config.data_description,
            constraints=self.config.constraints,
            summary=state["research_summary"],
        )
        analysis: AIMessage = self.llm.invoke(messages)  # type: ignore[arg-type]
        return {**state, "requirements_analysis": analysis.content}

    def _design_architecture(self, state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    "You design practical recommender system architectures that balance accuracy, latency, and ops."
                ),
                HumanMessage(
                    "Using the requirements analysis, sketch an offline + realtime architecture. Include data flow, "
                    "embedding/feature store choices, retrieval, ranking, reranking, feedback loop, and cost-aware deployment."
                ),
                HumanMessage("Requirements analysis:\n{analysis}"),
            ]
        )
        messages: List[BaseMessage] = prompt.format_messages(
            analysis=state["requirements_analysis"]
        )
        architecture: AIMessage = self.llm.invoke(messages)  # type: ignore[arg-type]
        return {**state, "architecture": architecture.content}

    def _plan_experiments(self, state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage("You design lean experiment plans for recommender systems."),
                HumanMessage(
                    "Create an experiment roadmap with baselines, dataset splits, metrics, A/B rollout steps, and risk mitigations. "
                    "Keep latency and cost constraints in mind. Provide concise tables or bullet lists."
                ),
                HumanMessage("Architecture proposal:\n{architecture}"),
            ]
        )
        messages: List[BaseMessage] = prompt.format_messages(architecture=state["architecture"])
        experiments: AIMessage = self.llm.invoke(messages)  # type: ignore[arg-type]
        return {**state, "experiments": experiments.content}


__all__ = ["AgentConfig", "ResearchAgent", "AgentState"]
