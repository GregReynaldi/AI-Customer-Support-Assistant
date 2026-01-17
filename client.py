from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import chroma
import csv
from datetime import datetime

model = OllamaLLM(model = "llama3:latest")

prompt = """
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

chat_prompt_template = ChatPromptTemplate.from_template(template = prompt)
threshold = 0.4

q = "Do you know what is Harbin Institute of Technology Shenzhen??"

result = chroma.similarity_search_with_relevance_scores(q, k = 6)
relevan_docs = [jawaban for jawaban,score in result if score>threshold]

if not relevan_docs : 
    final_answer = (
        "Dear Customer,\n\n"
        "I apologize, but I couldn't find specific information regarding your request in our records. "
        "I have forwarded this to our team for further review.\n\n"
        "Regards,\n"
        "Customer Service Reynaldi Company"
    )
    log_file = "customer_email.csv"
    with open(log_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), q])
else : 
    chain = chat_prompt_template | model
    from_docs = ""
    for docs in relevan_docs : 
        from_docs+=f"{docs.page_content}\n"
    final_answer = chain.invoke({"question":q,"context":from_docs})

print(final_answer)