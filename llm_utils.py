from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

def get_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4)

def get_answer(question, vectorstore, chat_history):
    retriever = vectorstore.as_retriever(search_type="similarity", k=4)
    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        return_source_documents=False
    )
    history = [(msg["content"], "") for msg in chat_history if msg["role"] == "user"]
    return chain.run({"question": question, "chat_history": history})
