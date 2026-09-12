import os
from typing import Any

import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .citation {
        font-size: 0.85rem;
        color: #666;
        background-color: #fafafa;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #ff9800;
    }
    .stButton > button {
        width: 100%;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
</style>
""",
    unsafe_allow_html=True,
)


def check_api_health() -> dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return dict(response.json())
    except Exception as e:
        return {"status": "error", "error": str(e)}


def upload_document(file: UploadedFile) -> dict[str, Any]:
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/documents/upload", files=files, timeout=60)
        return dict(response.json())
    except Exception as e:
        return {"error": str(e)}


def query_documents(
    question: str, top_k: int = 5, include_citations: bool = True
) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "top_k": top_k, "include_citations": include_citations},
            timeout=60,
        )
        return dict(response.json())
    except Exception as e:
        return {"error": str(e)}


def summarize_documents(document_ids: list[str], max_length: int = 500) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/summarize",
            json={"document_ids": document_ids, "max_length": max_length},
            timeout=60,
        )
        return dict(response.json())
    except Exception as e:
        return {"error": str(e)}


def research_topic(topic: str, max_steps: int = 5, max_sources_per_step: int = 3) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/research",
            json={
                "topic": topic,
                "max_steps": max_steps,
                "max_sources_per_step": max_sources_per_step,
            },
            timeout=180,
        )
        return dict(response.json())
    except Exception as e:
        return {"error": str(e)}


def get_document(document_id: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE_URL}/documents/{document_id}", timeout=10)
        return dict(response.json())
    except Exception as e:
        return {"error": str(e)}


def delete_document(document_id: str) -> dict[str, Any]:
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{document_id}", timeout=10)
        return dict(response.json())
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    st.markdown('<div class="main-header">🔬 AI Research Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        (
            '<div class="sub-header">Document Ingestion, RAG Q&A, '
            "Summarization & Agentic Research</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ System Status")
        health = check_api_health()
        if health.get("status") == "healthy":
            st.success("✅ All systems operational")
            for service, info in health.get("services", {}).items():
                status = info.get("status", "unknown")
                if status == "healthy":
                    st.write(f"✅ {service.capitalize()}")
                else:
                    st.write(f"❌ {service.capitalize()}: {info.get('error', 'Unknown error')}")
        else:
            st.error(f"❌ System degraded: {health.get('error', 'Unknown error')}")

        st.divider()
        st.header("📊 Settings")
        top_k = st.slider("Top-K Results", 1, 20, 5)
        include_citations = st.checkbox("Include Citations", True)
        max_steps = st.slider("Max Research Steps", 1, 10, 5)
        max_sources = st.slider("Sources per Step", 1, 5, 3)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📤 Upload Documents",
            "💬 Chat / Query",
            "📝 Summarize",
            "🔬 Deep Research",
            "📚 Document Library",
        ]
    )

    with tab1:
        st.subheader("Upload Documents")
        st.write("Supported formats: PDF, TXT, MD, DOCX (max 50MB)")

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    result = upload_document(file)
                if "error" in result:
                    st.error(f"Failed to upload {file.name}: {result['error']}")
                else:
                    st.success(f"✅ {file.name} uploaded successfully!")
                    st.json(result)

    with tab2:
        st.subheader("Chat with Your Documents")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("citations"):
                    for citation in message["citations"]:
                        citation_html = (
                            f'<div class="citation">📄 {citation["document_id"]}:'
                            f"{citation['chunk_index']} "
                            f"(score: {citation['score']:.3f})<br>{citation['content']}</div>"
                        )
                        st.markdown(citation_html, unsafe_allow_html=True)

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = query_documents(
                        prompt, top_k=top_k, include_citations=include_citations
                    )

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {result['error']}"}
                    )
                else:
                    st.markdown(result["answer"])
                    if result.get("citations"):
                        for citation in result["citations"]:
                            citation_html = (
                                f'<div class="citation">📄 {citation["document_id"]}:'
                                f"{citation['chunk_index']} "
                                f"(score: {citation['score']:.3f})<br>{citation['content']}</div>"
                            )
                            st.markdown(citation_html, unsafe_allow_html=True)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "citations": result.get("citations", []),
                        }
                    )

    with tab3:
        st.subheader("Summarize Documents")

        document_ids = (
            st.text_area(
                "Document IDs (one per line)",
                placeholder="doc-id-1\ndoc-id-2\n...",
                help="Enter document IDs to summarize, one per line",
            )
            .strip()
            .split("\n")
        )
        document_ids = [d.strip() for d in document_ids if d.strip()]

        max_length = st.slider("Max Summary Length", 100, 2000, 500)

        if st.button("Generate Summary", type="primary"):
            if not document_ids:
                st.warning("Please enter at least one document ID")
            else:
                with st.spinner("Generating summary..."):
                    result = summarize_documents(document_ids, max_length)
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.markdown("### Summary")
                    st.write(result["summary"])
                    st.caption(
                        f"Model: {result['model_used']} | Documents: {len(result['document_ids'])}"
                    )

    with tab4:
        st.subheader("Agentic Deep Research")
        st.write(
            "The agent will perform multi-step research on your topic, "
            "iteratively querying documents and synthesizing findings."
        )

        topic = st.text_input(
            "Research Topic",
            placeholder="e.g., Impact of transformer architectures on NLP",
        )
        col1, col2 = st.columns(2)
        with col1:
            max_steps = st.slider("Max Steps", 1, 10, max_steps)
        with col2:
            max_sources = st.slider("Sources per Step", 1, 5, max_sources)

        if st.button("Start Research", type="primary"):
            if not topic:
                st.warning("Please enter a research topic")
            else:
                with st.spinner("Conducting research... This may take a few minutes."):
                    result = research_topic(topic, max_steps, max_sources)

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.markdown("### Final Answer")
                    st.write(result["final_answer"])

                    st.markdown("### Research Steps")
                    for step in result["steps"]:
                        with st.expander(f"Step {step['step']}: {step['action']}"):
                            st.write(f"**Query:** {step['query']}")
                            st.write(f"**Result:** {step['result']}")
                            if step.get("citations"):
                                st.write("**Sources:**")
                                for citation in step["citations"]:
                                    citation_html = (
                                        f'<div class="citation">📄 {citation["document_id"]}:'
                                        f"{citation['chunk_index']} "
                                        f"(score: {citation['score']:.3f})</div>"
                                    )
                                    st.markdown(citation_html, unsafe_allow_html=True)

                    if result.get("all_citations"):
                        st.markdown("### All Citations")
                        for citation in result["all_citations"]:
                            citation_html = (
                                f'<div class="citation">📄 {citation["document_id"]}:'
                                f"{citation['chunk_index']} "
                                f"(score: {citation['score']:.3f})"
                                f"<br>{citation['content']}</div>"
                            )
                            st.markdown(citation_html, unsafe_allow_html=True)

    with tab5:
        st.subheader("Document Library")
        st.info(
            "Document listing functionality coming soon. Use the API directly to manage documents."
        )


if __name__ == "__main__":
    main()
