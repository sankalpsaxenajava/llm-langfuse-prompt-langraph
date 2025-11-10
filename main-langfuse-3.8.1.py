from dotenv import load_dotenv
from langfuse import get_client
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict
import os

load_dotenv()
langfuse = get_client()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

class GraphState(TypedDict):
    question: str
    answer: str

def ask_ai(state: GraphState):
    question = state["question"]

    try:
        prompt_version = langfuse.get_prompt("assistant-prompt")
        prompt_text = prompt_version.prompt.format(question=question)
    except Exception:
        prompt_text = f"Answer the following question clearly:\n{question}"

    generation = langfuse.start_generation(
        name="ask_ai-node",
        input=prompt_text,
    )

    response = llm.invoke(prompt_text)

    generation.output = response.content
    generation.end()

    return {"answer": response.content}


graph = StateGraph(GraphState)
graph.add_node("ask_ai", ask_ai)
graph.set_entry_point("ask_ai")
graph.add_edge("ask_ai", END)
workflow = graph.compile()

question = "What is Langfuse and why is it useful?"
result = workflow.invoke({"question": question})

final_gen = langfuse.start_generation(
    name="final-answer",
    input=question,
)
final_gen.output = result["answer"]
final_gen.end()

langfuse.flush()

print("Generations successfully logged to Langfuse (v3.8.1)")
print(f"Q: question}")
print(f"A: {result['answer']}")
