
# docurag/app.py
import gradio as gr
import numpy as np
import os
from dotenv import load_dotenv
from src.document_processing import extract_text, clean_text, chunk_text, detect_language, generate_chunk_summaries
from src.hybrid_retrieval import HybridRetrieval
from src.generation import SummaryGenerator
from src.graph_summary import generate_keyword_table, generate_bar_chart
from src.rag_pipeline import RAGPipeline
from transformers import pipeline
import torch
from datetime import datetime
import tempfile

# Load environment variables
load_dotenv("config/api_keys.env")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# Initialize components
hybrid_retriever = HybridRetrieval()
summarizer = SummaryGenerator()
rag_pipeline = RAGPipeline(hybrid_retriever)
notes_output = gr.Textbox(
    label="Saved Notes",
    interactive=False,  # Make it read-only
    lines=10,
    elem_classes="notes-box"
)
# analyzer = DocumentAnalyzer()
# result = analyzer.full_analysis(your_text)
# print(result['analysis'])

processing_outputs = [
    gr.Textbox(label="Processing Status"),        # Index 0
    gr.Textbox(label="Detected Language"),       # Index 1
    gr.Markdown(label="Document Summary"),       # Index 2 (LLM summary)
    gr.DataFrame(label="Key Terms",              # Index 3
                headers=["Term", "Frequency"],
                datatype=["str", "number"]),
    gr.Image(label="Visualization"),              # Index 4
    gr.Textbox(label="Document Domain")           # Index 5
]

# Add to event handlers
def save_to_notes(chat_history, current_notes):
    """Format notes as text"""
    new_notes = current_notes or ""
    if chat_history:
        last_entry = chat_history[-1]
        note_content = last_entry[1] if isinstance(last_entry, tuple) else last_entry.get("content", "")
        new_notes += f"Note saved at {datetime.now().strftime('%H:%M')}:\n{note_content}\n\n"
    return new_notes

def save_notes_to_file(notes):
    # Check if notes is empty or just whitespace
    if not notes or not notes.strip():
        return None
    
    # Create directory if it doesn’t exist
    os.makedirs("user_notes", exist_ok=True)
    
    # Generate a unique filename with timestamp
    filename = f"user_notes/notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        # Write notes to file with UTF-8 encoding
        with open(filename, "w", encoding="utf-8") as f:
            f.write(notes)
        return filename
    except Exception as e:
        print(f"Error saving notes to file: {e}")
        return None
    
def classify_domain(text):
    """Classify document domain using zero-shot classification"""
    classifier = pipeline("zero-shot-classification",
                        # model="facebook/bart-large-mnli"
                        model="MoritzLaurer/deberta-v3-base-zeroshot-v1",  # Smaller model
                        device=-1 )
    
    candidate_labels = ["education", "health", "technology", "legal", "weather", "sports", "finance", "entertainment", "news","other"]
    result = classifier(text[:1000], candidate_labels)  
    return result['labels'][0]  

def chunk_summarization(text, summarizer, chunk_size=500):
    """Memory-safe chunk-based summarization"""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    
    for chunk in chunks:
        try:
            summary = summarizer.generate_structured_summary(chunk) #generate_structured_summary #generate
            summaries.append(summary)
            torch.cuda.empty_cache()
        except Exception as e:
            summaries.append(f"[Chunk error: {str(e)}]")
    
    return "\n".join(summaries)

def process_basic(file):
    """Handle basic processing only"""
    try:
        text = extract_text(file.name)
        cleaned_text = clean_text(text)
        
        return [
            "Basic processing completed!",  # Index 0
            f"Language: {detect_language(cleaned_text[:1000])}",  # Index 1
            f"Domain: {classify_domain(cleaned_text)}"  # Index 5
        ]
    
    except Exception as e:
        return [
            f"Error: {str(e)}",  # Index 0
            "Language detection failed",  # Index 1
            "Domain classification failed"  # Index 5
        ]

def process_analysis(file):
    """Handle analysis-only processing"""
    try:
        text = extract_text(file.name)
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)
        hybrid_retriever.index_documents(chunks)
        
        keywords = hybrid_retriever.extract_keywords(cleaned_text)
        counts = [cleaned_text.lower().count(kw.lower()) for kw in keywords]
        chart_path = generate_bar_chart(keywords[:10], counts[:10])
        summary = summarizer.generate_structured_summary(cleaned_text) #generate_structured_summary , generate
        
        return [
            summary,  # Index 2
            [[kw, cnt] for kw, cnt in zip(keywords, counts)][:10],  # Index 3
            chart_path  # Index 4
        ]
    
    except Exception as e:
        return [
            "Summary unavailable",  # Index 2
            [["Error", 0]],  # Index 3
            None  # Index 4
        ]

def handle_user_query(query, chat_history):
    """Process queries with strict message formatting"""
    try:
        # Generate response from RAG system
        response = rag_pipeline.generate_response(query)
        
        # Append messages in correct format
        updated_history = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
        
        return updated_history
    
    except Exception as e:
        error_msg = f"System Error: {str(e)}"
        return chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_msg}
        ]

def create_interface():
    with gr.Blocks(title="DocRAG", css=".markdown {max-width: 1200px} footer {display: none !important;}") as app:
        gr.Markdown("# 📄 DocRAG - Intelligent Document Analysis")
        
        with gr.Row():
            # Document Upload Column
            with gr.Column(scale=1):
                gr.Markdown("## Step 1: Document Processing")
                file_input = gr.File(label="Upload PDF/TXT", file_types=[".pdf", ".txt"])
                upload_btn = gr.Button("Process Document", variant="primary")
                processing_outputs[0].render()
                processing_outputs[1].render()
                processing_outputs[5].render()
                analysis_btn = gr.Button("Analyze Document", variant="primary")
                
            # Chat Interface Column
            with gr.Column(scale=3):
                gr.Markdown("## Step 2: Interactive Analysis")
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": "Ready to analyze your document!"}],
                    label="Analysis Dialogue",
                    height=500,
                    render_markdown=True,
                    type="messages",  
                    avatar_images=(None, None)
                )
                query_input = gr.Textbox(
                    label="Query Input",
                    placeholder="Type your question here...",
                    lines=1
                )
                submit_btn = gr.Button("Generate Analysis", variant="primary")
            
            # Results Column
            with gr.Column(scale=2):
                gr.Markdown("## Analysis Results")
                with gr.Tab("Summary"):
                    processing_outputs[2].render()
                with gr.Tab("Key Terms"):
                    processing_outputs[3].render()
                with gr.Tab("Visualization"):
                    processing_outputs[4].render()
                # Add to interface layout
                with gr.Tab("Notes"):
                    notes_output.render()
                    note_btn = gr.Button("Add to Notes", variant="secondary") 
                    # copy_btn = gr.Button("Copy Notes", variant="secondary", elem_classes="copy-btn")
                    clear_notes_btn = gr.Button("Clear Notes", variant="secondary")
                    save_btn = gr.Button("Save to File", variant="secondary")
                saved_file = gr.File(label="Download Notes", visible=False)
        # Add state for notes
        notes_state = gr.State(value="")
        
        # Event chain for note saving
        note_btn.click(
            save_to_notes,
            inputs=[chatbot, notes_state],
            outputs=[notes_state]
        ).then(
            lambda x: x,
            inputs=[notes_state],
            outputs=[notes_output]
        )
        
        # Clear notes handler
        clear_notes_btn.click(
            lambda: ("", ""),  # Clear both state and textbox
            outputs=[notes_state, notes_output]
        )
            
        # Event handlers
        upload_btn.click(
            process_basic,
            inputs=file_input,
            outputs= [
                processing_outputs[0],  # Status
                processing_outputs[1],  # Language
                processing_outputs[5]   # Domain
            ]
        )
        analysis_btn.click(
            process_analysis,
            inputs=file_input,
            outputs=[
                processing_outputs[2],  # Summary
                processing_outputs[3],  # Key Terms
                processing_outputs[4]   # Visualization
            ]
        )
        
        submit_btn.click(
            handle_user_query,
            inputs=[query_input, chatbot],
            outputs=[chatbot]  
        ).then(
            lambda: "",  
            inputs=None,
            outputs=query_input
        )
        save_btn.click(
            fn=save_notes_to_file,
            inputs=notes_state,
            outputs=notes_output
        )

    return app

if __name__ == "__main__":
    # Set memory optimization flags
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    
    app = create_interface()
    app.launch(
        server_name="localhost",
        server_port=7861,
        share=False,
        auth=("admin", os.getenv("APP_PASSWORD", "docurag")),
        auth_message="Enter admin credentials:"
    )