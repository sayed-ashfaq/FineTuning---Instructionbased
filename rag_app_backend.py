from dotenv import load_dotenv
load_dotenv()
import os
import json
import requests
from uuid import uuid4

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_dim = len(embeddings.embed_query("hello"))

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

index = faiss.IndexFlatL2(embedding_dim)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

from langchain_core.documents import Document

from langchain_core.documents import Document

docs = [
    # --------- ORDER STATUS ----------
    Document(
        page_content=(
            "You can check your order status by logging into your account and "
            "visiting the 'My Orders' section. The status of each order will be "
            "displayed along with estimated delivery dates."
        ),
        metadata={"source": "customer_support", "topic": "order-status"}
    ),
    Document(
        page_content=(
            "If your order has been shipped, you will receive a tracking number "
            "via email which you can use to track the delivery in real-time."
        ),
        metadata={"source": "customer_support", "topic": "order-status"}
    ),

    # --------- CANCEL OR RETURN ORDER ----------
    Document(
        page_content=(
            "To cancel an order, go to 'My Orders', select the order you want to "
            "cancel, and click 'Cancel Order'. Cancellations are only allowed before "
            "the order has been shipped."
        ),
        metadata={"source": "customer_support", "topic": "cancel-order"}
    ),
    Document(
        page_content=(
            "To return a product, visit the 'Returns' section, select the order, "
            "and follow the instructions to generate a return label. Ensure the "
            "item is unused and in original packaging."
        ),
        metadata={"source": "customer_support", "topic": "return-order"}
    ),

    # --------- PAYMENT ISSUES ----------
    Document(
        page_content=(
            "If you encounter a payment issue, first check if your card details are "
            "correct. Contact your bank if the payment fails repeatedly. You can "
            "also reach out to our support team for assistance."
        ),
        metadata={"source": "customer_support", "topic": "payment-issues"}
    ),
    Document(
        page_content=(
            "Refunds are processed within 5-7 business days after a successful "
            "cancellation or return. You will be notified via email once the refund "
            "is completed."
        ),
        metadata={"source": "customer_support", "topic": "payment-issues"}
    ),

    # --------- ACCOUNT & LOGIN ----------
    Document(
        page_content=(
            "If you forget your password, click on 'Forgot Password' at the login page "
            "and follow the instructions to reset it via your registered email."
        ),
        metadata={"source": "customer_support", "topic": "account-login"}
    ),
    Document(
        page_content=(
            "To update your account details such as email or phone number, go to "
            "'Account Settings' and edit your personal information."
        ),
        metadata={"source": "customer_support", "topic": "account-login"}
    ),

    # --------- GENERAL FAQ ----------
    Document(
        page_content=(
            "Our customer support is available 24/7 via chat, email, or phone. "
            "Response times may vary depending on query volume."
        ),
        metadata={"source": "customer_support", "topic": "general-faq"}
    ),
    Document(
        page_content=(
            "Shipping charges vary depending on location and delivery speed. "
            "You can view estimated shipping fees at checkout."
        ),
        metadata={"source": "customer_support", "topic": "shipping-info"}
    ),
]


vector_store.add_documents(docs)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

RAG_PROMPT = """
You are a helpful AI assistant.

Use ONLY the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

def call_finetuned_llm(prompt: str) -> str:
    # Safety guard – don’t ever send an empty prompt to the model
    if not prompt.strip():
        return "Error: empty prompt was generated before calling the LLM."

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY  # if you later protect the API with an API key

    payload = {"inputs": prompt}

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except Exception as e:
        return f"HTTP error from API: {e} | body={resp.text}"

    outer = resp.json()

    # Handle both: proxy-wrapped {"statusCode":200,"body":"..."}
    # and direct {"result":[...]} styles, just in case.
    if isinstance(outer, dict) and "statusCode" in outer and "body" in outer:
        body_str = outer["body"]
        if isinstance(body_str, str):
            try:
                inner = json.loads(body_str)
            except json.JSONDecodeError:
                return f"Unexpected 'body' from API: {body_str}"
        else:
            inner = body_str
    else:
        inner = outer

    # inner should be {"result": [...]}
    result = inner.get("result", inner)

    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        return str(first)

    return str(result)
    
def generate_answer(question: str):

    # Retrieve
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Build prompt
    final_prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    # LLM call
    answer = call_finetuned_llm(final_prompt)

    return {
        "question": question,
        "context": context,
        "answer": answer
    }