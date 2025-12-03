import sys
import ast
from pathlib import Path
import os
import streamlit as st
import tempfile
import os
import json
import subprocess

st.set_page_config(page_title="AIRA Chat", layout="wide")
st.title("AIRA: Recsys 리서치/설계 Assistant")

def call_aira(pdf_files, objective, kpi, data_desc, constraints, user_question=None):
    if not pdf_files:
        raise ValueError("최소 1개 이상의 PDF를 업로드해주세요.")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_paths = []
        for f in pdf_files:
            path = os.path.join(tmpdir, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            pdf_paths.append(path)

        repo_root = Path(__file__).resolve().parent
        run_agent_path = repo_root / "scripts" / "run_agent.py"

        cmd = [
            sys.executable,
            str(run_agent_path),
            "--pdf", *pdf_paths,
            "--objective", objective,
            "--kpi", kpi,
            "--data", data_desc,
            "--constraints", constraints,
        ]

        if user_question:
            objective_with_q = objective + f"\n\n[사용자 추가 질문]\n{user_question}"
            cmd[cmd.index("--objective") + 1] = objective_with_q

        # 자식 프로세스는 UTF-8로 출력하게
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # bytes로 받고 나중에 우리가 디코딩
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        raw_stdout = (result.stdout or b"").decode("utf-8", errors="ignore")
        raw_stderr = (result.stderr or b"").decode("utf-8", errors="ignore")

        if result.returncode != 0:
            # run_agent.py 내부 에러는 stderr 그대로 보여주자
            raise RuntimeError(
                f"run_agent.py 실행 실패 (code={result.returncode})\n\nSTDERR:\n{raw_stderr}"
            )

        text = raw_stdout.strip()
        if not text:
            raise RuntimeError("run_agent.py stdout이 비어 있습니다.")

        # stdout 전체에서 { ... } 블록만 한 번 시도
        start = text.find("{")
        end = text.rfind("}")

        payload = None

        if start != -1 and end != -1 and end > start:
            json_like = text[start:end + 1]
            # 1차: JSON 시도
            try:
                payload = json.loads(json_like)
            except Exception:
                # JSON 아니면 그냥 포기하고 아래에서 raw 텍스트로 처리
                payload = None

        # 그래도 payload를 못 만들었으면, 그냥 raw 텍스트 통째로 반환
        if payload is None:
            payload = {"raw": text}

        return payload, raw_stdout.splitlines()

st.sidebar.header("실험 설정")

uploaded_pdfs = st.sidebar.file_uploader(
    "논문 PDF 업로드 (여러 개 가능)",
    type=["pdf"],
    accept_multiple_files=True,
)

default_objective = "고가치 사용자 유지율 개선을 위한 next-item 추천"
default_kpi = "NDCG@10 + latency < 80ms"
default_data = "GA4 이벤트, 상품 마스터, 1.8억 트랜잭션"
default_constraints = "Vertex AI 실시간 서빙, 비용 월 5천불 한도"

objective = st.sidebar.text_area("Objective", value=default_objective)
kpi = st.sidebar.text_input("KPI", value=default_kpi)
data_desc = st.sidebar.text_area("Data description", value=default_data)
constraints = st.sidebar.text_area("Constraints", value=default_constraints)

show_raw = st.sidebar.checkbox("raw stdout 보기", value=False)

# ---------- 세션 상태 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메시지 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 입력 ----------
user_input = st.chat_input("AIRA에게 질문해 보세요 (Multi-Turn 적용 예정).")

if user_input:
    # 1) 유저 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) AIRA 호출
    try:
        with st.chat_message("assistant"):
            with st.spinner("AIRA 에이전트 실행 중..."):
                payload, raw_logs = call_aira(
                    uploaded_pdfs,
                    objective=objective,
                    kpi=kpi,
                    data_desc=data_desc,
                    constraints=constraints,
                    user_question=user_input,
                )

            # --- 응답 구조 파싱 ---
            config = payload.get("config", {})
            meta = payload.get("paper_taxonomy", {}).get("meta", {})
            roadmap = payload.get("roadmap", {})

            # outputs 우선, 없으면 roadmap.legacy_outputs, 그래도 없으면 result/payload
            data = (
                payload.get("outputs")
                or roadmap.get("legacy_outputs")
                or payload.get("result")
                or payload
            )

            sections = []

            # 0. 설정 / 메타 정보 간단 요약
            header_lines = []
            if config.get("objective"):
                header_lines.append(f"- **Objective**: {config['objective']}")
            if config.get("kpi"):
                header_lines.append(f"- **KPI**: {config['kpi']}")
            if meta.get("title"):
                header_lines.append(f"- **논문 제목**: {meta['title']}")
            if meta.get("year"):
                header_lines.append(f"- **연도**: {meta['year']}")
            if header_lines:
                sections.append("#### ⚙️ 설정 / 메타\n" + "\n".join(header_lines))

            # 1. 논문 요약 / 요구사항 / 아키텍처 / 실험 플랜
            if isinstance(data, dict):
                if "research_summary" in data:
                    sections.append("### 🔍 논문 요약\n" + data["research_summary"])
                if "requirements_analysis" in data:
                    sections.append("### 🎯 요구사항 정렬\n" + data["requirements_analysis"])
                if "architecture" in data:
                    sections.append("### 🏗 아키텍처 제안\n" + data["architecture"])
                if "experiments" in data:
                    sections.append("### 🧪 실험 플랜\n" + data["experiments"])

            # 2. 로드맵에서 참고 모델 / 아키텍처 가이드 / 근거 / 미래 방향도 보여주기
            ref_models = roadmap.get("reference_models") or []
            if ref_models:
                sections.append("### 🧩 참고 모델\n" + "\n".join(f"- {m}" for m in ref_models))

            if roadmap.get("architecture_guidance"):
                sections.append("### 🏛 아키텍처 가이드\n" + roadmap["architecture_guidance"])

            if roadmap.get("justification"):
                sections.append("### 📌 설계 근거\n" + roadmap["justification"])

            future_dirs = roadmap.get("future_directions") or []
            if future_dirs:
                sections.append("### 🔭 Future Work 제안\n" + "\n".join(f"- {d}" for d in future_dirs))

            # 최종 출력 조합
            if sections:
                answer_text = "\n\n".join(sections)
            else:
                # 혹시 위에서 다 실패하면 그냥 전체 payload를 JSON으로 보여주기
                answer_text = "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"

            st.markdown(answer_text)

            if show_raw:
                with st.expander("디버그: raw stdout"):
                    st.code("\n".join(raw_logs), language="bash")

        # 3) assistant 메시지를 대화 히스토리에 추가
        st.session_state.messages.append({"role": "assistant", "content": answer_text})

    except Exception as e:
        error_msg = f"에러 발생: {e}"
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
