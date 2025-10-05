from model_utils import load_vectorstore, load_llm, build_rag_chain

def answer_question(question, persist_directory="vectorstore", model_name="TheBloke/Hermes-2-Pro-Mistral-7B-GPTQ"):
    vectorstore = load_vectorstore(persist_directory=persist_directory)
    llm = load_llm(model_name=model_name)
    qa_chain = build_rag_chain(vectorstore, llm)
    return qa_chain.run(question)

if __name__ == "__main__":
    while True:
        q = input("Enter your question (or 'exit' to quit): ")
        if q.lower() == "exit":
            break
        answer = answer_question(q)
        print("Answer:", answer)
