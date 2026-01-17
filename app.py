from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
import tempfile
import json
import csv
from datetime import datetime

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global variables for models
chroma = None
rag_retriever = None
whisper_model = None
whisper_processor = None
level_model = None
level_tokenizer = None
queue_model = None
queue_tokenizer = None
type_model = None
type_tokenizer = None
llm_model = None
chat_prompt_template = None
models_loaded = False

# Load models during startup
@app.on_event("startup")
async def startup_event():
    global chroma, rag_retriever, whisper_model, whisper_processor
    global level_model, level_tokenizer, queue_model, queue_tokenizer, type_model, type_tokenizer
    global llm_model, chat_prompt_template
    global models_loaded
    
    print("Loading models...")
    
    # Load RAG
    try:
        model_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        path_db = "./database_vec"
        chroma = Chroma(
            collection_name="TicketRAG",
            embedding_function=model_embeddings,
            persist_directory=path_db
        )
        rag_retriever = chroma.as_retriever(k=8)
        print("RAG loaded successfully")
    except Exception as e:
        print(f"Error loading RAG: {e}")
    
    # Load LLM
    try:
        llm_model = OllamaLLM(model="llama3:latest")
        prompt = prompt = """
You are a professional Customer Support Assistant at Reynaldi Company. 
Analyze the provided Document Context to answer the user's inquiry.

REPLACEMENT RULES:
1. If the context or your answer contains ANY company name, replace it with: Reynaldi Company.
2. If the context or your answer contains ANY phone number, replace it with: +86-13028896826.
3. If the context or your answer contains ANY email address, replace it with: gregoriusreynaldi@gmail.com.

RESPONSE RULES:
1. START the response with a professional greeting: "Dear Customer,".
2. Use ONLY the information provided in the "Document Context".
3. If the answer is NOT in the context, state: "I'm sorry, I don't have specific information about that in our records."
4. END the response with:
   "Regards,
   Customer Service Reynaldi Company"

Document Context:
{context}

Question:
{question}

Your Answer:
"""
        chat_prompt_template = ChatPromptTemplate.from_template(template=prompt)
        print("LLM loaded successfully")
    except Exception as e:
        print(f"Error loading LLM: {e}")
    
    # Load Whisper model
    try:
        whisper_model = WhisperForConditionalGeneration.from_pretrained("./audio_model")
        whisper_processor = WhisperProcessor.from_pretrained("./audio_processor")
        print("Whisper model loaded successfully")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
    
    # Load classification models
    try:
        # Level classification
        level_tokenizer = AutoTokenizer.from_pretrained("./level_classification")
        level_model = AutoModelForSequenceClassification.from_pretrained("./level_classification")
        
        # Queue classification
        queue_tokenizer = AutoTokenizer.from_pretrained("./queue_classification")
        queue_model = AutoModelForSequenceClassification.from_pretrained("./queue_classification")
        
        # Type classification
        type_tokenizer = AutoTokenizer.from_pretrained("./type_classification")
        type_model = AutoModelForSequenceClassification.from_pretrained("./type_classification")
        
        print("Classification models loaded successfully")
    except Exception as e:
        print(f"Error loading classification models: {e}")
    
    models_loaded = True
    print("All models loaded successfully")

# Check if models are loaded
@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": models_loaded}

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Process chat query
@app.post("/chat")
async def chat(query: str = Form(...)):
    if not chroma or not llm_model or not chat_prompt_template:
        return JSONResponse({"error": "RAG or LLM models not loaded"}, status_code=503)
    
    try:
        # Classify the query first
        classifications = await classify_text(query)
        
        # Use similarity search with relevance scores
        threshold = 0.4
        result = chroma.similarity_search_with_relevance_scores(query, k=6)
        relevan_docs = [jawaban for jawaban, score in result if score > threshold]
        
        # Generate response based on relevant documents
        if not relevan_docs:
            final_answer = (
                "Dear Customer,\n\n"
                "I apologize, but I couldn't find specific information regarding your request in our records. "
                "I have forwarded this to our team for further review.\n\n"
                "Regards,\n"
                "Customer Service Reynaldi Company"
            )
            # Log to CSV with classification results
            log_to_csv(query, classifications)
        else:
            # Create chain and generate response
            from_docs = "\n".join([doc.page_content for doc in relevan_docs])
            chain = chat_prompt_template | llm_model
            final_answer = chain.invoke({"question": query, "context": from_docs})
        
        return JSONResponse({
            "response": final_answer,
            "classifications": classifications
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Classification label mappings
level_map = {0: 'high', 1: 'low', 2: 'medium'}
queue_map = {0: 'Customer Service', 1: 'IT Support', 2: 'Product Support', 3: 'Technical Support', 4: 'Others'}
type_map = {0: 'Change', 1: 'Incident', 2: 'Problem', 3: 'Requests'}

# Log to CSV function
def log_to_csv(query, classifications):
    log_file = "customer_email.csv"
    file_exists = os.path.exists(log_file)
    
    with open(log_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(["timestamp", "query", "level", "queue", "type"])
        
        # Map classification numbers to labels
        level_label = level_map.get(classifications.get("level"), "-")
        queue_label = queue_map.get(classifications.get("queue"), "-")
        type_label = type_map.get(classifications.get("type"), "-")
        
        # Write data
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            level_label,
            queue_label,
            type_label
        ])

# Process audio file
@app.post("/audio")
async def audio(file: UploadFile = File(...)):
    if not whisper_model or not whisper_processor:
        return JSONResponse({"error": "Whisper model not loaded"}, status_code=503)
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load audio
        audio, sr = librosa.load(temp_file_path, sr=16000)
        
        # Process audio
        inputs = whisper_processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        
        # Generate transcription
        with torch.no_grad():
            generate_ids = whisper_model.generate(**inputs)
        
        transcription = whisper_processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        transcription = transcription.strip()
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Classify the transcription
        classifications = await classify_text(transcription)
        
        return JSONResponse({
            "transcription": transcription,
            "classifications": classifications
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Classify text
async def classify_text(text: str):
    classifications = {}
    
    # Level classification
    if level_model and level_tokenizer:
        try:
            inputs = level_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = level_model(**inputs)
            level = torch.argmax(outputs.logits, dim=1).item()
            classifications["level"] = level
        except Exception as e:
            classifications["level"] = f"Error: {str(e)}"
    
    # Queue classification
    if queue_model and queue_tokenizer:
        try:
            inputs = queue_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = queue_model(**inputs)
            queue = torch.argmax(outputs.logits, dim=1).item()
            classifications["queue"] = queue
        except Exception as e:
            classifications["queue"] = f"Error: {str(e)}"
    
    # Type classification
    if type_model and type_tokenizer:
        try:
            inputs = type_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = type_model(**inputs)
            type_ = torch.argmax(outputs.logits, dim=1).item()
            classifications["type"] = type_
        except Exception as e:
            classifications["type"] = f"Error: {str(e)}"
    
    return classifications

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
