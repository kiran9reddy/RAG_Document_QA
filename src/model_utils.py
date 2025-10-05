from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_vectorstore(persist_directory="vectorstore"):
    return Chroma(persist_directory=persist_directory, embedding_function=None)

def load_llm(model_name="TheBloke/Hermes-2-Pro-Mistral-7B-GPTQ"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain
