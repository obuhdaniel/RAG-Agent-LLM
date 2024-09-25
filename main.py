from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
import pinecone
from dotenv import load_dotenv
import os

class ChatBot():
  load_dotenv()
  loader = TextLoader('./info.txt')
  documents = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
  docs = text_splitter.split_documents(documents)

  embeddings = HuggingFaceEmbeddings()

  pinecone.init(
      api_key= os.getenv('PINECONE_API_KEY'),
      environment='gcp-starter'
  )

  index_name = "langchain-demo"

  if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric="cosine", dimension=768)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
  else:
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

  repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
  llm = HuggingFaceHub(
      repo_id=repo_id, model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50}, huggingfacehub_api_token=os.getenv('HUGGINGFACE_ACCESS_TOKEN')
  )

  from langchain import PromptTemplate

  template = """
  You are a knowledgeable resource person in the Department of Computer Engineering. Individuals will ask you questions regarding their academic pursuits, career options, and departmental resources. Use the provided context to formulate your answers. If you're unsure about something, simply state that you don't know.If you believe the answer is beyond the context given you can respond with a fact backed answer. Ensure your responses are clear and concise, limited to two sentences.

  Context: {context}
  Question: {question}
  Answer: 

  """

  prompt = PromptTemplate(template=template, input_variables=["context", "question"])

  from langchain.schema.runnable import RunnablePassthrough
  from langchain.schema.output_parser import StrOutputParser

  rag_chain = (
    {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
  )
  
  def _update_pinecone(self):
    """Update the Pinecone index with new embeddings."""
    self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)

  def retrain(self, new_file_path):
    """Retrain the model with a new document."""
    # Load the new file
    self.loader = TextLoader(new_file_path)
    documents = self.loader.load()

    # Re-split the new documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    self.docs = text_splitter.split_documents(documents)

    # Re-embed and update Pinecone
    self._update_pinecone()

    # You can add logging here to confirm the retraining process
    print(f"Retrained with new file: {new_file_path}")