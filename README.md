# LangChain Document Processing and QA

This project demonstrates how to use LangChain and OpenAI to process a document, create embeddings, and perform a question-answering task using a retrieval-based approach.

## Requirements

- Python 3.7+
- `langchain-openai`
- `langchain`
- `langchain-community`
- `faiss-cpu`
- `numpy`
- `torch`
- `transformers`

### Importing Necessary Libraries

```python
import os
from langchain_openai import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
```

### Setting Up the OpenAI API Key

Set your OpenAI API key:

```python
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
```

### Initializing the Language Model

Create an instance of the OpenAI chat model:

```python
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
```

### Loading the Document

Load the document you want to process:

```python
loader = UnstructuredFileLoader("/content/CheatSheet.pdf")
documents = loader.load()
```

### Creating Text Chunks

Split the document into chunks for better processing:

```python
text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)
text_chunks = text_splitter.split_documents(documents)
```

### Creating Embeddings

Load the embedding model and create vector embeddings for the text chunks:

```python
embeddings = HuggingFaceEmbeddings()
knowledge_base = FAISS.from_documents(text_chunks, embeddings)
```

### Setting Up the Retrieval QA Chain

Create a chain for question-answer retrieval:

```python
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=knowledge_base.as_retriever()
)
```

### Asking a Question

Ask a question to the chain and print the response:

```python
question = "What is this document about?"
response = qa_chain.invoke({"query": question})
print(response["result"])
```


