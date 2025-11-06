from dotenv import load_dotenv
from langfuse import Langfuse
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict
import os

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)


llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # use a model you have access to
    api_key=os.getenv("GROQ_API_KEY")
)

class GraphState(TypedDict):
    question: str
    answer: str

def ask_ai(state: GraphState):
    question = state["question"]
    prompt_version = langfuse.get_prompt("assistant-prompt")
    prompt_text = prompt_version.prompt.format(question=question)

    generation = langfuse.generation(
        name="ask_ai-node",
        trace_id=current_trace.id,  # Use the global trace
        input=prompt_text,
        prompt_name=prompt_version.name,
        prompt_version=prompt_version.version,
    )
    response = llm.invoke(prompt_text)

    generation.end(output=response.content)

    return {"answer": response.content}

graph = StateGraph(GraphState)
graph.add_node("ask_ai", ask_ai)
graph.set_entry_point("ask_ai")
graph.add_edge("ask_ai", END)
workflow = graph.compile()


current_trace = langfuse.trace(name="groq-langgraph-observability")

question = "What is Langfuse and why is it useful?"
result = workflow.invoke({"question": question})


final_generation = langfuse.generation(
    name="final-answer",
    trace_id=current_trace.id,
    input=question,
    output=result["answer"],
)
final_generation.end()

langfuse.flush()

print(" Trace, prompt, and generation successfully linked in Langfuse!")
print(f"Q: {question}")
print(f"A: {result['answer']}")
