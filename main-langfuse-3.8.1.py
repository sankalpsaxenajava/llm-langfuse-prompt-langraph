from dotenv import load_dotenv
from langfuse import get_client
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv()

langfuse = get_client()
logging.info("Langfuse client initialized")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)
logging.info("ChatGroq LLM initialized")

class GraphState(TypedDict):
    question: str
    answer: str

def ask_ai(state: GraphState):
    question = state["question"]
    logging.info(f"Asking AI: {question}")

    try:
        prompt_version = langfuse.get_prompt("assistant-prompt")
        prompt_text = prompt_version.prompt.format(question=question)
        logging.info("Using custom prompt from Langfuse")
    except Exception:
        prompt_text = f"Answer the following question clearly:\n{question}"
        logging.info("Using default prompt")

    generation = langfuse.start_generation(
        name="ask_ai-node",
        input=prompt_text
    )

    response = llm.invoke(prompt_text)
    answer = getattr(response, "content", None) or str(response)
    logging.info(f"LLM response: {answer}")

    generation.text = answer
    generation.end()
    langfuse.flush()

    return {"answer": answer}

graph = StateGraph(GraphState)
graph.add_node("ask_ai", ask_ai)
graph.set_entry_point("ask_ai")
graph.add_edge("ask_ai", END)
workflow = graph.compile()
logging.info("StateGraph workflow compiled")

question = "What is Langfuse and why is it useful?"

span = langfuse.start_span(name="ask_ai-workflow")

result = workflow.invoke({"question": question})

final_gen = langfuse.start_generation(
    name="final-answer",
    input=question
)
final_gen.text = result["answer"]
final_gen.end()

span.end()
langfuse.flush()

logging.info("Generations successfully logged to Langfuse (v3.8.1)")
print(f"Q: {question}")
print(f"A: {result['answer']}")
