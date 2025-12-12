import streamlit as st
import os
import tempfile
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from dotenv import load_dotenv
load_dotenv()
from tavily import TavilyClient


# ===============================================
# Backend configuration for chroma vectorestore, llm client, and models (llm + embeddings)
# ===============================================

# Clear ChromaDB system cache at startup
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Initialize client with API (put Cornell API key in OPENAI_API_KEY env variable)
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url="https://api.ai.it.cornell.edu",
)

# Model configuration (we can change this or allow the user to select)
LLM_MODEL = "openai.gpt-4o-mini"
EMBEDDING_MODEL = "openai.text-embedding-3-small"

# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

def internet_search(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Search error: missing TAVILY_API_KEY."

    try:
        tavily_client = TavilyClient(api_key=api_key)
        response = tavily_client.search(query, max_results=3)

        items = response.get("results", [])
        if not items:
            return "No results found."

        return "\n".join(
            f"- {it.get('title', 'N/A')}: {it.get('url','')} — {it.get('content', 'N/A')}"
            for it in items
        )
    except Exception as e:
        return f"Search error: {e}"


# ===============================================
# System prompting for agents
# ===============================================

INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for a research paper assistant. Classify the user's message into exactly one of these categories:

- "summarize" - User wants to summarize papers, extract key findings, or get an overview.
- "question" - User is asking a specific question about the paper content.
- "research" - User wants research gap analysis, future directions, or research suggestions.

Examples:
- "Summarize this paper" = summarize
- "What are the key findings?" = summarize
- "Give me an overview" = summarize
- "What methods did they use?" = question
- "How did they collect data?" = question
- "What were the results?" = question
- "What research gaps exist?" = research
- "Suggest future research directions" = research
- "What limitations does this research have?" = research

Respond with only the category word: summarize, question, or research
If uncertain, respond with: question"""

READER_AGENT_PROMPT = """You are a Research Paper Reader Agent. Your role is to:

1. Extract and identify key sections from research papers (abstract, introduction, 
   methodology, results, discussion, conclusion)
2. Summarize the main findings and contributions of each paper
3. Identify the research questions, hypotheses, and objectives
4. Extract key data, statistics, and experimental results
5. Note the limitations acknowledged by the authors

When analyzing a paper, be thorough but concise. Focus on the most important 
information that would help a researcher understand the paper's contribution 
to the field.

Format your response with clear sections:
- **Summary**: Brief overview of the paper
- **Key Findings**: Main results and discoveries
- **Methodology**: Research methods used
- **Contributions**: Novel contributions to the field
- **Limitations**: Acknowledged limitations

Always base your analysis on the provided context from the papers."""

QA_AGENT_PROMPT = """You are a Research Paper Question-Answering Agent. Your role is to:

1. Understand and interpret user questions about uploaded research papers
2. Search through the paper content to find relevant information
3. Provide accurate, well-reasoned answers based on the paper content
4. Always cite specific passages or sections that support your answer
5. Acknowledge when information is not available in the papers

Guidelines:
- Be precise and factual. Only state what is supported by the paper content.
- If a question cannot be fully answered from the available papers, say so.
- Distinguish between what the paper states and your interpretation.
- Use direct quotes when appropriate, with proper attribution.
- If multiple papers are available, synthesize information across them when relevant.

Always base your answers on the provided context from the papers."""

RESEARCH_ADVISOR_PROMPT = """You are a Research Advisor Agent. Your role is to:

1. Synthesize insights and findings across multiple research papers.
2. Identify gaps, limitations, and open questions in the current research.
3. Suggest promising directions for future research.
4. Connect ideas across different papers and identify potential synergies.
5. Evaluate the significance and potential impact of research directions.

Guidelines:
- Base suggestions on concrete evidence from the papers.
- Clearly distinguish between what papers state and your inferences.
- Consider practical feasibility of suggested research directions.
- Identify both incremental improvements and potentially transformative ideas.
- When suggesting new directions, explain the rationale and potential impact.
- Highlight gaps that could be addressed.
- Consider interdisciplinary connections when relevant.

Format your response with clear sections:
- Research Gaps: Identified gaps in current research.
- Future Directions: Promising research directions.
- Potential Impact: Why these directions matter.
- Connections: Links between papers or ideas.

Always base your analysis on the provided context from the papers."""


RESEARCH_VERIFIER_PROMPT = """You are a Research Idea Verifier Agent.

Input you receive will include:
(1) Proposed research directions produced from the uploaded papers.
(2) Web search results (snippets) for targeted queries.

Your goals:
1. For each proposed direction, assess whether it appears already explored or closely related work exists.
2. If related work exists, explain briefly how it overlaps and how to adjust the idea to be more novel or precise.
3. Flag claims that require citation or that the web snippets contradict.
4. Produce a revised set of directions that are stronger, better scoped, and clearly positioned.

Rules:
- Only use evidence from the provided web snippets; do not invent paper titles, authors, or numbers.
- If the snippets are insufficient, say “unclear from search results” and suggest a better query.
- Keep the final output structured and actionable.

Format:
For each idea:
- Status: Likely novel / Possibly known / Likely known / Unclear
- Evidence: 1–2 bullets referencing the web snippets
- Revision: improved idea + positioning
- Next search query: (if unclear)

Output constraints:
- Verify at most 4 ideas (pick the most promising / most uncertain).
- Max 120 words per idea.
- Do not include extra sections beyond the required fields.

"""

# ===============================================
# Session State Initialization
# ===============================================

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Please upload research papers using the sidebar then ask questions about them. I can summarize papers, answer specific questions, or suggest research directions."}
    ]

if "current_files" not in st.session_state:
    st.session_state["current_files"] = []

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "file_chunks" not in st.session_state:
    st.session_state["file_chunks"] = []

# ===============================================
# Analyze user message and determine what type of task they want
# ===============================================

def classify_intent(user_message: str) -> str:
    """
    Classify user intent to route to the appropriate agent.
    Returns: 'summarize', 'question', or 'research'
    Defaults to 'question' if uncertain.
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=20,
            temperature=0
        )
        intent = response.choices[0].message.content.strip().lower()
        
        # Validate intent is one of expected values
        if intent in ["summarize", "question", "research"]:
            return intent
        return "question"  # Default fallback
    except Exception:
        return "question"  # Default on error


def get_agent_info(intent: str) -> tuple[str, str]:
    """
    Get agent name and system prompt based on intent.
    Returns: (agent_name, system_prompt)
    """
    if intent == "summarize":
        return "Reader Agent", READER_AGENT_PROMPT
    elif intent == "research":
        return "Research Advisor", RESEARCH_ADVISOR_PROMPT
    elif intent == "verify":
        return "Research Verifier", RESEARCH_VERIFIER_PROMPT
    else:
        return "QA Agent", QA_AGENT_PROMPT
# ===============================================
# Run the selected agent with context
# ===============================================

def run_agent(user_message: str, intent: str, context: str) -> str:
    """
    Run the appropriate agent based on intent with retrieved context.
    Returns the agent's response as a streaming generator.
    """
    agent_name, system_prompt = get_agent_info(intent)
    
    # Build the system message with context
    if context:
        system_with_context = f"""{system_prompt}


Here is relevant context from the uploaded research papers:

{context}
"""
    else:
        system_with_context = f"""{system_prompt}

Note: No relevant context was retrieved. This may indicate an issue with document processing or retrieval.
"""
    
    # Create streaming response
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_with_context},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]],  # Last 10 messages for context
            {"role": "user", "content": user_message}
        ],
        stream=True
    )
    
    return stream

def run_agent_nonstream(user_message: str, intent: str, context: str) -> str:
    agent_name, system_prompt = get_agent_info(intent)

    if context:
        system_with_context = f"""{system_prompt}

Context:

{context}
"""
    else:
        system_with_context = system_prompt

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_with_context},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2,
        max_tokens=900
    )
    return resp.choices[0].message.content or ""


# =============================================================================
# Process uploaded PDF files
# =============================================================================

def process_uploaded_files(uploaded_files) -> bool:
    """
    Process uploaded PDF files: extract text, chunk, and create vector store.
    Returns True if successful.
    """
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        try:
            # Save to temp file for PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.getvalue())
                temp_pdf_path = temp_pdf.name
            
            # Load PDF
            pdf_loader = PyPDFLoader(temp_pdf_path)
            pages = pdf_loader.load()
            file_content = "\n".join([page.page_content for page in pages])
            
            # Clean up temp file
            os.unlink(temp_pdf_path)
            
            # Chunk the content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            file_chunks = text_splitter.split_text(file_content)
            
            # Add metadata to track source
            for chunk in file_chunks:
                all_chunks.append(chunk)
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            return False
    
    # Store chunks in session state
    st.session_state["file_chunks"] = all_chunks
    
    try:
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url="https://api.ai.it.cornell.edu/v1",
            model=EMBEDDING_MODEL
        )
        
        # Create Chroma vectorstore
        chroma_client = chromadb.EphemeralClient(
            settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
        vectorstore = Chroma.from_texts(
            texts=all_chunks,
            embedding=embeddings,
            client=chroma_client,
            collection_name="research_papers"
        )
        
        st.session_state["vectorstore"] = vectorstore
        st.success(f"Vectorstore created with {len(all_chunks)} chunks")
        return True
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return False


def retrieve_context(query: str, k: int = 5) -> str:
    """
    Retrieve relevant context from vector store for the given query.
    """
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        st.error("Vectorstore not found. Please re-upload your papers.")
        return ""
    
    try:
        relevant_docs = vectorstore.similarity_search(query, k=k)
        if not relevant_docs:
            st.warning("No relevant documents found for your query.")
            return ""
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        return context
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return ""

# ===============================================
# Streamlit layout
# ===============================================

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Multi-Agent Research Assistant")
st.caption("Upload papers, ask questions, get insights")

# ===============================================
# Sidebar items
# ===============================================

with st.sidebar:
    st.header("Upload Papers")
    
    uploaded_files = st.file_uploader(
        "Upload Research Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF research papers to analyze"
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) selected")
        
        # Check if files have changed
        uploaded_filenames = [f.name for f in uploaded_files]
        if st.session_state["current_files"] != uploaded_filenames:
            with st.status("Processing documents...", expanded=True) as status:
                st.write("Extracting text from PDFs...")
                st.write("Chunking documents...")
                st.write("Creating embeddings...")
                
                if process_uploaded_files(uploaded_files):
                    st.session_state["current_files"] = uploaded_filenames
                    status.update(label="Documents processed!", state="complete")
                else:
                    status.update(label="Processing failed", state="error")
    
    # Display uploaded papers
    if st.session_state["current_files"]:
        st.divider()
        st.subheader("Loaded Papers")
        for filename in st.session_state["current_files"]:
            st.text(f"• {filename}")
        
        chunks_count = len(st.session_state.get('file_chunks', []))
        vectorstore_status = "✅ Ready" if st.session_state.get("vectorstore") else "❌ Not Found"
        st.caption(f"{chunks_count} chunks indexed • Vectorstore: {vectorstore_status}")
    
    st.divider()
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Summary", use_container_width=True, disabled=not st.session_state["current_files"]):
            st.session_state["quick_action"] = "Summarize all the uploaded papers. Provide key findings, methodology, and contributions."
    
    with col2:
        if st.button("Research Ideas", use_container_width=True, disabled=not st.session_state["current_files"]):
            st.session_state["quick_action"] = "Analyze the uploaded papers and suggest future research directions based on gaps and limitations."
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. Ask me anything about your research papers."}
        ]
        st.rerun()

# ===============================================
# Chat interface
# ===============================================

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show agent badge if present
        if msg.get("agent"):
            st.caption(f"{msg['agent']}")
        st.markdown(msg["content"])

# Handle quick actions
user_typed = st.chat_input(
    "Ask about your research papers...",
    disabled=not st.session_state["current_files"]
)

qa = st.session_state.get("quick_action")
if qa:
    question = qa
    st.session_state["quick_action"] = None
else:
    question = user_typed


# Process user input
if question:
    # Check if documents are loaded
    handled = False
    if not st.session_state.get("vectorstore"):
        st.error("Please upload research papers first!")
        handled = True
    else:
    
        # Display user message
        st.chat_message("user").write(question)
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Classify intent
        with st.spinner("Analyzing intent..."):
            intent = classify_intent(question)
        
        agent_name, _ = get_agent_info(intent)
        
        # Retrieve relevant context
        context = retrieve_context(question, k=5)

        if intent == "research":
            # Step 1: generate research ideas (non-streaming)
            ideas_text = run_agent_nonstream(question, intent="research", context=context)

            # Show the immediate Research Advisor output
            

            with st.chat_message("assistant"):
                st.caption("Research Advisor")
                st.markdown(ideas_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": ideas_text,
                "agent": "Research Advisor"
            })

            # Extract a few candidate idea lines to search (keeps searches small)
            bullet_lines = [
                ln.strip("-• ").strip()
                for ln in ideas_text.splitlines()
                if ln.strip().startswith(("-", "•"))
            ]
            queries = [f"{b} related work arxiv" for b in bullet_lines[:3]]

            # Run Tavily searches
            web_snippets = []
            for q in queries:
                web_snippets.append(f"QUERY: {q}\n{internet_search(q)}")

            verifier_context = (
                "PROPOSED IDEAS:\n" + ideas_text +
                "\n\nWEB SEARCH RESULTS:\n" + "\n\n".join(web_snippets)
            )

            # Step 2: verifier output
            final_text = run_agent_nonstream(
                "Vet and revise these ideas based on the web results.",
                intent="verify",
                context=verifier_context
            )

            with st.chat_message("assistant"):
                st.caption("Research Verifier")
                st.markdown(final_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_text,
                "agent": "Research Verifier"
            })

            handled = True

    
        # Generate response
        if not handled:
            with st.chat_message("assistant"):
                st.caption(f"{agent_name}")
                
                # Show source chunks in expander
                if context:
                    with st.expander("Source Chunks Used", expanded=False):
                        chunks = context.split("\n\n---\n\n")
                        for i, chunk in enumerate(chunks, 1):
                            st.text_area(
                                f"Chunk {i}",
                                chunk[:500] + "..." if len(chunk) > 500 else chunk,
                                height=100,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                            if i < len(chunks):
                                st.divider()
                
                # Stream response
                try:
                    stream = run_agent(question, intent, context)
                    response = st.write_stream(stream)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "agent": f"{agent_name}"
                    })
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "agent": f"{agent_name}"
                    })
