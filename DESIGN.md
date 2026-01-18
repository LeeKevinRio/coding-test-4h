# Design Document: Multimodal Document Chat System

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Design Choices](#design-choices)
3. [Evaluation Pipeline Design](#evaluation-pipeline-design)
4. [Prompt Versioning Strategy](#prompt-versioning-strategy)
5. [Technical Decisions & Trade-offs](#technical-decisions--trade-offs)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Upload  │  │   Chat   │  │ Document │  │   Conversation   │ │
│  │   Page   │  │Interface │  │  Viewer  │  │     History      │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
└───────┼─────────────┼────────────┼──────────────────┼───────────┘
        │             │            │                  │
        ▼             ▼            ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     API Endpoints                           │ │
│  │  POST /api/documents/upload  │  POST /api/chat             │ │
│  │  GET /api/documents          │  GET /api/conversations     │ │
│  └────────────────────┬─────────────────────┬─────────────────┘ │
│                       │                     │                    │
│  ┌────────────────────▼─────────────────────▼─────────────────┐ │
│  │                    Service Layer                            │ │
│  │  ┌─────────────────┐ ┌─────────────┐ ┌──────────────────┐  │ │
│  │  │   Document      │ │   Vector    │ │      Chat        │  │ │
│  │  │   Processor     │ │    Store    │ │     Engine       │  │ │
│  │  │   (Docling)     │ │  (pgvector) │ │   (RAG + LLM)    │  │ │
│  │  └────────┬────────┘ └──────┬──────┘ └────────┬─────────┘  │ │
│  │           │                 │                  │            │ │
│  └───────────┼─────────────────┼──────────────────┼────────────┘ │
└──────────────┼─────────────────┼──────────────────┼──────────────┘
               │                 │                  │
               ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │   PostgreSQL     │  │     Redis        │  │  File Storage  │ │
│  │   + pgvector     │  │   (Caching)      │  │  (Images/PDFs) │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
PDF Upload → Docling Parse → Extract (Text, Images, Tables)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Text Chunks      Images          Tables
                    │               │               │
                    ▼               │               │
              Embeddings ←─────────┴───────────────┘
              (sentence-transformers)    (Metadata linking)
                    │
                    ▼
              pgvector Storage
                    │
                    ▼
User Query → Vector Search → Context Retrieval → LLM → Response
```

---

## Design Choices

### 1. Chunking Strategy

**Choice: Paragraph-based chunking with semantic overlap**

#### Why this approach?

```python
CHUNK_CONFIG = {
    "chunk_size": 512,      # ~512 tokens per chunk
    "chunk_overlap": 50,    # 50 tokens overlap for context continuity
    "strategy": "paragraph" # Split by paragraphs, then sentences
}
```

**Rationale:**
- **Paragraph boundaries** preserve semantic coherence better than fixed-size splitting
- **512 tokens** is optimal for most embedding models (all-MiniLM-L6-v2, OpenAI ada)
- **50-token overlap** ensures context continuity across chunk boundaries
- Maintains readability for retrieved content display

**Alternatives considered:**
| Strategy | Pros | Cons | Decision |
|----------|------|------|----------|
| Fixed-size (character) | Simple, predictable | Breaks mid-sentence | ❌ Rejected |
| Sentence-based | Natural boundaries | Too small for context | ❌ Rejected |
| Paragraph-based | Semantic coherence | Variable size | ✅ Selected |
| Semantic chunking (LLM) | Best coherence | Expensive, slow | Future enhancement |

**Trade-offs:**
- Chunk sizes vary (some paragraphs > 512 tokens) → We split long paragraphs at sentence boundaries
- Short paragraphs may lack context → We merge consecutive short paragraphs

### 2. Multimodal Linking Strategy

**Choice: Spatial proximity + explicit metadata references**

#### Linking approach:

```python
# Each text chunk stores references to related media
chunk_metadata = {
    "related_images": [image_ids],  # Images on same page
    "related_tables": [table_ids],  # Tables on same page
    "page_number": int,
    "bbox": [x1, y1, x2, y2]        # Optional: spatial position
}
```

**Why spatial proximity?**
- PDF documents naturally group related content by page
- Images/tables typically appear near their referencing text
- Simple and effective for most academic/technical documents

**Linking strategies evaluated:**
| Method | Implementation | Accuracy | Complexity |
|--------|---------------|----------|------------|
| Same-page linking | Filter by page_number | Good | Low |
| Bbox proximity | Calculate distances | Better | Medium |
| Caption matching | NLP similarity | Best | High |
| Explicit references | Parse "Figure 1" refs | Best | High |

**Current implementation: Same-page linking** (pragmatic choice for 4-hour scope)

**Future enhancements:**
```python
# Caption-based linking (TODO)
def link_by_caption(text_chunk, images):
    for img in images:
        if references_figure(text_chunk, img.caption):
            yield img.id

# Explicit reference parsing (TODO)
def parse_references(text):
    # Match patterns like "Figure 1", "Table 2", "see Fig. 3"
    patterns = [r"Figure\s+(\d+)", r"Table\s+(\d+)", r"Fig\.\s*(\d+)"]
    ...
```

### 3. Embedding Model Selection

**Choice: sentence-transformers (all-MiniLM-L6-v2) as default**

| Model | Dimension | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | Free |
| text-embedding-3-small | 1536 | Medium | Better | $0.02/1M |
| nomic-embed-text (Ollama) | 768 | Fast | Good | Free |

**Rationale:**
- Free and runs locally (no API dependency)
- 384 dimensions provide good balance of quality vs. storage
- Fast enough for real-time processing

---

## Evaluation Pipeline Design

### Quality Metrics Framework

```python
class RAGEvaluator:
    """
    Evaluation pipeline using RAGAS-style metrics.
    """
    
    def evaluate(self, test_cases: List[TestCase]) -> EvalResults:
        return {
            "faithfulness": self.measure_faithfulness(test_cases),
            "answer_relevance": self.measure_relevance(test_cases),
            "context_precision": self.measure_precision(test_cases),
            "context_recall": self.measure_recall(test_cases)
        }
```

### Metric Definitions

#### 1. Faithfulness (Grounded in context)
```python
def measure_faithfulness(question, answer, context):
    """
    Measures if the answer is grounded in retrieved context.
    
    Method: LLM-as-a-judge
    - Extract claims from answer
    - Verify each claim against context
    - Score = verified_claims / total_claims
    """
    prompt = f"""
    Given the context: {context}
    And the answer: {answer}
    
    Extract all factual claims from the answer.
    For each claim, determine if it can be verified from the context.
    
    Output:
    - Total claims: N
    - Verified claims: M
    - Unverified claims: [list]
    """
    return verified_claims / total_claims
```

#### 2. Answer Relevance
```python
def measure_answer_relevance(question, answer):
    """
    Measures if the answer addresses the question.
    
    Method: Embedding similarity
    - Generate embedding for question
    - Generate embedding for answer
    - Compute cosine similarity
    """
    q_emb = embed(question)
    a_emb = embed(answer)
    return cosine_similarity(q_emb, a_emb)
```

#### 3. Context Precision
```python
def measure_context_precision(question, retrieved_chunks, relevant_chunks):
    """
    Measures precision of retrieval.
    
    precision = relevant_retrieved / total_retrieved
    """
    relevant_retrieved = set(retrieved_chunks) & set(relevant_chunks)
    return len(relevant_retrieved) / len(retrieved_chunks)
```

#### 4. Context Recall
```python
def measure_context_recall(question, retrieved_chunks, ground_truth_chunks):
    """
    Measures recall of retrieval.
    
    recall = relevant_retrieved / total_relevant
    """
    relevant_retrieved = set(retrieved_chunks) & set(ground_truth_chunks)
    return len(relevant_retrieved) / len(ground_truth_chunks)
```

### Test Cases for "Attention Is All You Need"

```python
TEST_CASES = [
    {
        "question": "Can it find Figure 1?",
        "expected": "Reference to Transformer architecture diagram",
        "expected_media": ["figure_1_transformer_architecture"]
    },
    {
        "question": "What are the BLEU scores?",
        "expected": "Table with BLEU scores (28.4 EN-DE, 41.0 EN-FR)",
        "expected_media": ["table_2_bleu_scores"]
    },
    {
        "question": "Explain Self-Attention",
        "expected": "Description from Section 3.2",
        "expected_sections": ["section_3.2"]
    }
]
```

### LLM-as-a-Judge Approach

```python
JUDGE_PROMPT = """
You are evaluating a RAG system's response.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}
Ground Truth: {ground_truth}

Evaluate on:
1. Correctness (0-10): Is the answer factually correct?
2. Completeness (0-10): Does it fully answer the question?
3. Grounding (0-10): Is every claim supported by context?
4. Relevance (0-10): Is retrieved context relevant?

Provide scores and brief justification.
"""
```

---

## Prompt Versioning Strategy

### Current Implementation

```python
# Hardcoded prompts (current state)
SYSTEM_PROMPT = """You are a helpful AI assistant..."""
RAG_PROMPT_TEMPLATE = """Based on the following context..."""
```

### Proposed Architecture

```python
# prompt_registry.py
class PromptRegistry:
    """
    Centralized prompt management with versioning.
    """
    
    def __init__(self, storage_backend="database"):
        self.backend = storage_backend
        self.cache = {}
    
    def get_prompt(self, name: str, version: str = "latest") -> Prompt:
        """
        Retrieve a prompt by name and version.
        """
        cache_key = f"{name}:{version}"
        if cache_key not in self.cache:
            self.cache[cache_key] = self._load_from_backend(name, version)
        return self.cache[cache_key]
    
    def register_prompt(self, name: str, template: str, metadata: dict) -> str:
        """
        Register a new prompt version.
        Returns version ID.
        """
        version = self._generate_version()
        self._save_to_backend(name, version, template, metadata)
        return version
```

### Prompt Schema

```python
@dataclass
class Prompt:
    name: str
    version: str
    template: str
    variables: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    
    # A/B testing support
    experiment_group: Optional[str] = None
    
    def render(self, **kwargs) -> str:
        return self.template.format(**kwargs)
```

### Version Control Strategy

```yaml
# prompts/system_prompt.yaml
name: system_prompt
versions:
  - version: "1.0.0"
    template: |
      You are a helpful AI assistant...
    created: "2024-01-01"
    
  - version: "1.1.0"
    template: |
      You are a document analysis expert...
    created: "2024-01-15"
    changes:
      - "Added citation instructions"
      - "Improved grounding guidance"
```

### Scaling Considerations

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| Hardcoded | Simple, fast | No versioning | MVP/Prototypes |
| Config files | Git-trackable | Deployment overhead | Small teams |
| Database | Dynamic, A/B testing | Complexity | Production |
| External service | Full MLOps | Infrastructure cost | Enterprise |

**Recommendation for scale:** Database-backed with Redis caching

```python
# Production architecture
PromptRegistry(
    primary_storage="postgresql",
    cache="redis",
    ttl=3600,  # 1 hour cache
    enable_ab_testing=True
)
```

---

## Technical Decisions & Trade-offs

### 1. Docling vs. Alternatives

| Library | PDF Quality | Table Extraction | Image Extraction | Speed |
|---------|-------------|------------------|------------------|-------|
| Docling | Excellent | Excellent | Good | Medium |
| PyMuPDF | Good | Poor | Good | Fast |
| pdfplumber | Good | Good | Poor | Medium |
| Unstructured.io | Excellent | Excellent | Excellent | Slow |

**Decision:** Docling - best balance for academic papers

### 2. pgvector vs. Dedicated Vector DB

| Option | Scalability | Operational Complexity | Cost |
|--------|-------------|----------------------|------|
| pgvector | Good (millions) | Low (existing PG) | Low |
| Pinecone | Excellent | Medium | Medium |
| Weaviate | Excellent | High | Medium |
| Milvus | Excellent | High | Low |

**Decision:** pgvector - simplicity and sufficient scale for this use case

### 3. LLM Provider Strategy

```python
# Fallback chain for LLM
LLM_PROVIDERS = [
    ("openai", "gpt-4o-mini"),      # Best quality
    ("groq", "llama-3.1-70b"),      # Free tier available
    ("gemini", "gemini-1.5-flash"), # Free tier available
    ("ollama", "llama3.2"),         # Local, free
]
```

**Rationale:** Provider-agnostic design allows:
- Cost optimization
- Fallback resilience
- Local development without API keys

---

## Performance Optimizations

### Implemented
- [x] Async processing for document upload
- [x] Batch embedding generation
- [x] Connection pooling (SQLAlchemy)

### Planned
- [ ] Redis caching for embeddings
- [ ] Background job queue (Celery)
- [ ] Streaming responses
- [ ] Chunk deduplication

---

## Conclusion

This design prioritizes:
1. **Simplicity** - Minimal dependencies, clear architecture
2. **Flexibility** - Provider-agnostic, easy to swap components
3. **Scalability** - Can grow with pgvector, ready for dedicated vector DB
4. **Maintainability** - Clean separation of concerns, documented decisions

The 4-hour scope focused on core RAG functionality. Future iterations should add:
- Semantic chunking
- Advanced multimodal linking
- Comprehensive evaluation pipeline
- Production-grade prompt management
