"""
Rankify Web Playground - Interactive UI for Model Comparison

Launch an interactive Gradio interface to:
- Try different retrievers and rerankers
- Compare model outputs side-by-side
- Export code for selected configuration

Start:
    >>> from rankify.ui import launch_playground
    >>> launch_playground(port=7860)
"""

from typing import List, Optional, Dict, Any, Tuple
import json

# Try to import Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


# Available models
RETRIEVERS = ["bm25", "dpr", "bge", "ance", "colbert", "contriever"]
RERANKERS = ["flashrank", "monot5", "rankgpt", "inranker", "colbert_ranker", "upr"]
RAG_METHODS = ["basic-rag", "chain-of-thought-rag", "self-consistency-rag", "zero-shot"]


def create_playground_app():
    """Create Gradio playground application."""
    
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is required. Install with: pip install gradio")
    
    # Cache for initialized components
    _cache = {}
    
    def get_retriever(method: str, n_docs: int):
        """Get or create a cached retriever."""
        key = f"retriever_{method}_{n_docs}"
        if key not in _cache:
            from rankify.retrievers.retriever import Retriever
            _cache[key] = Retriever(method=method, n_docs=n_docs)
        return _cache[key]
    
    def get_reranker(method: str, model_name: Optional[str] = None):
        """Get or create a cached reranker."""
        key = f"reranker_{method}_{model_name}"
        if key not in _cache:
            from rankify.models.reranking import Reranking
            _cache[key] = Reranking(method=method, model_name=model_name)
        return _cache[key]
    
    def retrieve(query: str, retriever: str, n_docs: int) -> Tuple[str, str]:
        """Retrieve documents."""
        try:
            from rankify.dataset.dataset import Document, Question, Answer
            
            ret = get_retriever(retriever, n_docs)
            doc = Document(
                question=Question(query),
                answers=Answer([]),
                contexts=[],
            )
            results = ret.retrieve([doc])
            contexts = results[0].contexts[:10]
            
            # Format results
            output = []
            for i, ctx in enumerate(contexts):
                output.append(f"**{i+1}. {getattr(ctx, 'title', 'Document')}**")
                output.append(f"{ctx.text[:300]}...")
                output.append("")
            
            code = f'''from rankify import pipeline

# Create search pipeline
search = pipeline("search", retriever="{retriever}", n_docs={n_docs})
results = search("{query}")

for ctx in results.documents[0].contexts[:10]:
    print(ctx.text)
'''
            return "\n".join(output), code
            
        except Exception as e:
            return f"Error: {str(e)}", ""
    
    def rerank(query: str, documents: str, reranker: str, top_k: int) -> Tuple[str, str]:
        """Rerank documents."""
        try:
            from rankify.dataset.dataset import Document, Question, Answer, Context
            
            # Parse documents (one per line)
            doc_texts = [d.strip() for d in documents.strip().split("\n") if d.strip()]
            if not doc_texts:
                return "Please enter documents (one per line)", ""
            
            contexts = [
                Context(text=text, id=str(i))
                for i, text in enumerate(doc_texts)
            ]
            
            rr = get_reranker(reranker)
            doc = Document(
                question=Question(query),
                answers=Answer([]),
                contexts=contexts,
            )
            results = rr.rank([doc])
            reranked = (results[0].reorder_contexts or results[0].contexts)[:top_k]
            
            # Format results
            output = []
            for i, ctx in enumerate(reranked):
                score = getattr(ctx, 'score', 'N/A')
                output.append(f"**{i+1}. (Score: {score})**")
                output.append(ctx.text[:200])
                output.append("")
            
            code = f'''from rankify import pipeline

# Create rerank pipeline
rerank = pipeline("rerank", reranker="{reranker}")

documents = [
    {json.dumps(doc_texts[:3], indent=4)}
]

results = rerank.rerank("{query}", documents, top_k={top_k})
'''
            return "\n".join(output), code
            
        except Exception as e:
            return f"Error: {str(e)}", ""
    
    def rag(
        query: str,
        retriever: str,
        reranker: str,
        rag_method: str,
        n_contexts: int,
    ) -> Tuple[str, str, str]:
        """Run RAG pipeline."""
        try:
            from rankify import pipeline
            
            rag_pipe = pipeline(
                "rag",
                retriever=retriever,
                reranker=reranker,
                generator=rag_method,
                top_k=n_contexts,
            )
            
            result = rag_pipe(query)
            
            # Format answer
            answer = result.answers[0] if result.answers else "No answer generated"
            
            # Format contexts
            contexts_text = []
            for i, ctx in enumerate(result.documents[0].contexts[:n_contexts]):
                contexts_text.append(f"**Context {i+1}:**")
                contexts_text.append(ctx.text[:200] + "...")
                contexts_text.append("")
            
            code = f'''from rankify import pipeline

# Create RAG pipeline
rag = pipeline(
    "rag",
    retriever="{retriever}",
    reranker="{reranker}",
    generator="{rag_method}",
    top_k={n_contexts},
)

result = rag("{query}")
print(result.answers[0])
'''
            return answer, "\n".join(contexts_text), code
            
        except Exception as e:
            return f"Error: {str(e)}", "", ""
    
    def generate_pipeline_code(
        task: str,
        retriever: str,
        reranker: str,
        rag_method: str,
    ) -> str:
        """Generate code for selected configuration."""
        if task == "Search":
            return f'''from rankify import pipeline

# Search pipeline
search = pipeline("search", retriever="{retriever}")
results = search("Your query here")

for ctx in results.documents[0].contexts[:10]:
    print(ctx.text)
'''
        elif task == "Rerank":
            return f'''from rankify import pipeline

# Rerank pipeline  
rerank = pipeline("rerank", retriever="{retriever}", reranker="{reranker}")
results = rerank("Your query here")

for ctx in results.documents[0].reorder_contexts[:10]:
    print(ctx.text)
'''
        else:
            return f'''from rankify import pipeline

# RAG pipeline
rag = pipeline(
    "rag",
    retriever="{retriever}",
    reranker="{reranker}",
    generator="{rag_method}",
)

result = rag("Your question here")
print(result.answers[0])
'''
    
    # Create Gradio interface
    with gr.Blocks(
        title="Rankify Playground",
        theme=gr.themes.Soft(),
    ) as app:
        
        gr.Markdown("""
        # 🚀 Rankify Playground
        
        Interactive interface for testing retrieval, reranking, and RAG pipelines.
        """)
        
        with gr.Tabs():
            # Search Tab
            with gr.Tab("🔍 Search"):
                with gr.Row():
                    with gr.Column():
                        search_query = gr.Textbox(
                            label="Query",
                            placeholder="Enter your search query...",
                        )
                        search_retriever = gr.Dropdown(
                            choices=RETRIEVERS,
                            value="bm25",
                            label="Retriever",
                        )
                        search_n_docs = gr.Slider(
                            minimum=10, maximum=100, value=50,
                            label="Number of Documents",
                        )
                        search_btn = gr.Button("Search", variant="primary")
                    
                    with gr.Column():
                        search_results = gr.Markdown(label="Results")
                        search_code = gr.Code(language="python", label="Code")
                
                search_btn.click(
                    retrieve,
                    inputs=[search_query, search_retriever, search_n_docs],
                    outputs=[search_results, search_code],
                )
            
            # Rerank Tab
            with gr.Tab("📊 Rerank"):
                with gr.Row():
                    with gr.Column():
                        rerank_query = gr.Textbox(
                            label="Query",
                            placeholder="Enter query for reranking...",
                        )
                        rerank_docs = gr.Textbox(
                            label="Documents (one per line)",
                            placeholder="Enter documents to rerank...",
                            lines=10,
                        )
                        rerank_method = gr.Dropdown(
                            choices=RERANKERS,
                            value="flashrank",
                            label="Reranker",
                        )
                        rerank_k = gr.Slider(
                            minimum=1, maximum=20, value=5,
                            label="Top K",
                        )
                        rerank_btn = gr.Button("Rerank", variant="primary")
                    
                    with gr.Column():
                        rerank_results = gr.Markdown(label="Results")
                        rerank_code = gr.Code(language="python", label="Code")
                
                rerank_btn.click(
                    rerank,
                    inputs=[rerank_query, rerank_docs, rerank_method, rerank_k],
                    outputs=[rerank_results, rerank_code],
                )
            
            # RAG Tab
            with gr.Tab("🤖 RAG"):
                with gr.Row():
                    with gr.Column():
                        rag_query = gr.Textbox(
                            label="Question",
                            placeholder="Ask a question...",
                        )
                        rag_retriever = gr.Dropdown(
                            choices=RETRIEVERS,
                            value="bm25",
                            label="Retriever",
                        )
                        rag_reranker = gr.Dropdown(
                            choices=RERANKERS,
                            value="flashrank",
                            label="Reranker",
                        )
                        rag_method = gr.Dropdown(
                            choices=RAG_METHODS,
                            value="basic-rag",
                            label="RAG Method",
                        )
                        rag_contexts = gr.Slider(
                            minimum=1, maximum=10, value=5,
                            label="Number of Contexts",
                        )
                        rag_btn = gr.Button("Generate Answer", variant="primary")
                    
                    with gr.Column():
                        rag_answer = gr.Textbox(label="Answer", lines=5)
                        rag_contexts_out = gr.Markdown(label="Retrieved Contexts")
                        rag_code = gr.Code(language="python", label="Code")
                
                rag_btn.click(
                    rag,
                    inputs=[rag_query, rag_retriever, rag_reranker, rag_method, rag_contexts],
                    outputs=[rag_answer, rag_contexts_out, rag_code],
                )
            
            # Code Generator Tab
            with gr.Tab("💻 Code Generator"):
                gr.Markdown("Generate code for your selected configuration")
                
                with gr.Row():
                    gen_task = gr.Radio(
                        choices=["Search", "Rerank", "RAG"],
                        value="RAG",
                        label="Task",
                    )
                    gen_retriever = gr.Dropdown(
                        choices=RETRIEVERS,
                        value="bm25",
                        label="Retriever",
                    )
                    gen_reranker = gr.Dropdown(
                        choices=RERANKERS,
                        value="flashrank",
                        label="Reranker",
                    )
                    gen_rag = gr.Dropdown(
                        choices=RAG_METHODS,
                        value="basic-rag",
                        label="RAG Method",
                    )
                
                gen_code = gr.Code(language="python", label="Generated Code")
                
                # Update code when selections change
                for input_comp in [gen_task, gen_retriever, gen_reranker, gen_rag]:
                    input_comp.change(
                        generate_pipeline_code,
                        inputs=[gen_task, gen_retriever, gen_reranker, gen_rag],
                        outputs=gen_code,
                    )
        
        gr.Markdown("""
        ---
        **Rankify** - A Comprehensive Python Toolkit for Retrieval, Re-Ranking, and RAG
        
        [Documentation](https://rankify.readthedocs.io) | [GitHub](https://github.com/DataScienceUIBK/Rankify)
        """)
    
    return app


def launch_playground(
    port: int = 7860,
    share: bool = False,
    **kwargs,
):
    """
    Launch the Rankify playground.
    
    Args:
        port: Port to run on
        share: Create public share link
        **kwargs: Additional Gradio launch arguments
    """
    app = create_playground_app()
    app.launch(
        server_port=port,
        share=share,
        **kwargs,
    )


if __name__ == "__main__":
    launch_playground()
