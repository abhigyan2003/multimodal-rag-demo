import streamlit as st
       import pymupdf4llm
       import os
       import base64
       from langchain_community.vectorstores import FAISS
       from langchain_community.embeddings import HuggingFaceEmbeddings
       from langchain_google_genai import ChatGoogleGenerativeAI
       from langchain_core.documents import Document
       from langchain.text_splitter import RecursiveCharacterTextSplitter
       from PIL import Image
       import io
       import uuid

       # Initialize session state
       if "vector_store" not in st.session_state:
           st.session_state.vector_store = None
       if "image_store" not in st.session_state:
           st.session_state.image_store = {}

       # Streamlit app title
       st.title("Multi-Modal RAG PDF Q&A Demo")

       # Sidebar for PDF upload
       with st.sidebar:
           st.header("Upload PDF")
           uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

       # Function to encode image for Gemini
       def encode_image(image):
           buffered = io.BytesIO()
           image.save(buffered, format="PNG")
           return base64.b64encode(buffered.getvalue()).decode("utf-8")

       # Process PDF and extract text and images
       if uploaded_file:
           # Save uploaded file temporarily
           with open("temp.pdf", "wb") as f:
               f.write(uploaded_file.getbuffer())
           
           # Extract text and images using PyMuPDF4LLM
           pdf_data = pymupdf4llm.to_markdown("temp.pdf")
           documents = [Document(page_content=pdf_data)]
           
           # Extract images
           pdf_document = pymupdf4llm.open("temp.pdf")
           image_store = {}
           for page_num in range(len(pdf_document)):
               page = pdf_document[page_num]
               images = page.get_images(full=True)
               for img_index, img in enumerate(images):
                   xref = img[0]
                   base_image = pdf_document.extract_image(xref)
                   image_bytes = base_image["image"]
                   image = Image.open(io.BytesIO(image_bytes))
                   image_id = str(uuid.uuid4())
                   image_store[image_id] = {"image": image, "page": page_num}
                   # Associate image with nearby text (simplified)
                   documents.append(Document(page_content=f"Image on page {page_num}", metadata={"image_id": image_id}))
           
           # Split text into chunks
           text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
           splits = text_splitter.split_documents(documents)
           
           # Create vector store
           embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
           st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
           st.session_state.image_store = image_store
           st.success("PDF processed successfully!")

       # Question input
       question = st.text_input("Ask a question about the PDF:")
       if question and st.session_state.vector_store:
           # Retrieve relevant documents
           retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
           relevant_docs = retriever.get_relevant_documents(question)
           
           # Initialize Gemini model
           llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
           
           # Prepare context and images
           context = "\n".join([doc.page_content for doc in relevant_docs])
           image_ids = [doc.metadata.get("image_id") for doc in relevant_docs if doc.metadata.get("image_id")]
           images = [st.session_state.image_store.get(img_id) for img_id in image_ids if img_id in st.session_state.image_store]
           
           # Prepare prompt for Gemini
           prompt = f"Question: {question}\nContext: {context}\nAnswer the question based on the context. If relevant, describe any images."
           image_inputs = [{"mime_type": "image/png", "data": encode_image(img["image"])} for img in images]
           
           # Query Gemini
           response = llm.invoke(prompt, images=image_inputs)
           
           # Display answer
           st.write("**Answer:**")
           st.write(response.content)
           
           # Display relevant images
           if images:
               st.write("**Relevant Images:**")
               for img_data in images:
                   st.image(img_data["image"], caption=f"Image from page {img_data['page']}")

       # Clean up
       if os.path.exists("temp.pdf"):
           os.remove("temp.pdf")
