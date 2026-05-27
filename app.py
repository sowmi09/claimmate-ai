import streamlit as st
from src.pipeline import ClaimMatePipeline
from src.config import EMBEDDING_MODEL_NAME, OLLAMA_MODEL

st.set_page_config(page_title="ClaimMate AI", page_icon="🧾", layout="wide")

st.title("🧾 ClaimMate AI")
st.caption("Local RAG assistant for warranty, return, refund, and service-claim support")

with st.sidebar:
    st.header("Project Settings")
    st.write("**Retriever:** FAISS")
    st.write(f"**Embedding model:** `{EMBEDDING_MODEL_NAME}`")
    st.write(f"**Local LLM:** Ollama `{OLLAMA_MODEL}`")
    st.info("Run `python build_index.py` before starting the app.")

@st.cache_resource
def load_pipeline():
    return ClaimMatePipeline()

example = (
    "My laptop stopped working after 8 months. I have invoice and serial number. "
    "The service center rejected it saying physical damage, but there is no visible damage. Draft an email."
)

query = st.text_area("Enter your warranty/service claim question", value=example, height=130)

top_k = st.slider("Number of policy clauses to retrieve", min_value=2, max_value=8, value=4)

if st.button("Analyze Claim", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            pipeline = load_pipeline()
            result = pipeline.run(query, top_k=top_k)

            st.subheader("Detected Query Type")
            st.code(result["route"])

            st.subheader("ClaimMate Answer")
            st.write(result["answer"])

            st.subheader("Retrieved Policy Evidence")
            for i, item in enumerate(result["retrieved"], start=1):
                with st.expander(f"Evidence {i}: {item['policy_section']} | score={item['score']:.4f}"):
                    st.write(f"**Product Category:** {item['product_category']}")
                    st.write(f"**Clause Type:** {item['clause_type']}")
                    st.write(f"**Clause Text:** {item['clause_text']}")
                    st.write(f"**Required Documents:** {item['required_documents']}")
        except FileNotFoundError as e:
            st.error(str(e))
            st.code("python build_index.py")
        except Exception as e:
            st.exception(e)
