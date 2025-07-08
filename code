"""
Backend logic for AI Chatbot with PDF processing capabilities
Using LangChain with Ollama (free alternative to OpenAI)
"""

import os
import tempfile
from typing import List, Dict, Any
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAgent:
    """
    Main AI Agent class that handles conversation, PDF processing, and summarization
    """
    
    def __init__(self, model_name: str = "llama2"):
        """
        Initialize the AI Agent with Ollama model
        
        Args:
            model_name: Name of the Ollama model to use (default: llama2)
        """
        self.model_name = model_name
        self.llm = None
        self.embeddings = None
        self.memory = None
        self.conversation_chain = None
        self.vectorstore = None
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all LangChain components"""
        try:
            # Initialize Ollama LLM
            self.llm = Ollama(model=self.model_name, temperature=0.7)
            
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(model=self.model_name)
            
            # Initialize memory for conversation
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=10  # Keep last 10 exchanges
            )
            
            logger.info(f"AI Agent initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing AI Agent: {str(e)}")
            raise
    
    def chat(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Handle regular conversation
        
        Args:
            message: User message
            session_id: Session identifier for conversation tracking
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Create a simple prompt for conversation
            prompt = PromptTemplate(
                input_variables=["question"],
                template="""You are a helpful AI assistant. Respond to the user's question in a friendly and informative way.
                
                Question: {question}
                
                Answer:"""
            )
            
            # Get response from LLM
            formatted_prompt = prompt.format(question=message)
            response = self.llm(formatted_prompt)
            
            # Store in memory
            self.memory.save_context(
                {"input": message},
                {"output": response}
            )
            
            return {
                "response": response,
                "session_id": session_id,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                "response": f"I'm sorry, I encountered an error: {str(e)}",
                "session_id": session_id,
                "status": "error"
            }
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process PDF and create vector store for Q&A
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with processing status and metadata
        """
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(
                chunks,
                self.embeddings
            )
            
            # Create conversation chain for PDF Q&A
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                return_source_documents=True
            )
            
            logger.info(f"PDF processed successfully: {len(chunks)} chunks created")
            
            return {
                "status": "success",
                "message": f"PDF processed successfully. Created {len(chunks)} text chunks.",
                "num_pages": len(documents),
                "num_chunks": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing PDF: {str(e)}"
            }
    
    def chat_with_pdf(self, question: str) -> Dict[str, Any]:
        """
        Chat with PDF using RAG (Retrieval Augmented Generation)
        
        Args:
            question: User question about the PDF
            
        Returns:
            Dictionary with response and source documents
        """
        try:
            if not self.conversation_chain:
                return {
                    "response": "Please upload and process a PDF first.",
                    "status": "error"
                }
            
            # Get response from conversation chain
            result = self.conversation_chain({
                "question": question
            })
            
            # Extract source information
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    sources.append({
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    })
            
            return {
                "response": result["answer"],
                "sources": sources,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in PDF chat: {str(e)}")
            return {
                "response": f"Error processing question: {str(e)}",
                "status": "error"
            }
    
    def summarize_text(self, text: str, summary_type: str = "stuff") -> Dict[str, Any]:
        """
        Summarize given text
        
        Args:
            text: Text to summarize
            summary_type: Type of summarization (stuff, map_reduce, refine)
            
        Returns:
            Dictionary with summary
        """
        try:
            # Create document from text
            from langchain.schema import Document
            doc = Document(page_content=text)
            
            # Create summarization chain
            if summary_type == "stuff":
                prompt_template = """Write a concise summary of the following text:

{text}

CONCISE SUMMARY:"""
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = load_summarize_chain(
                    llm=self.llm,
                    chain_type="stuff",
                    prompt=prompt
                )
            else:
                chain = load_summarize_chain(
                    llm=self.llm,
                    chain_type=summary_type
                )
            
            # Generate summary
            summary = chain.run([doc])
            
            return {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "status": "error"
            }
    
    def summarize_pdf(self, pdf_path: str, summary_type: str = "map_reduce") -> Dict[str, Any]:
        """
        Summarize PDF document
        
        Args:
            pdf_path: Path to PDF file
            summary_type: Type of summarization
            
        Returns:
            Dictionary with summary
        """
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Create summarization chain
            if summary_type == "stuff" and len(documents) == 1:
                prompt_template = """Write a comprehensive summary of the following document:

{text}

SUMMARY:"""
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = load_summarize_chain(
                    llm=self.llm,
                    chain_type="stuff",
                    prompt=prompt
                )
            else:
                # Use map_reduce for longer documents
                chain = load_summarize_chain(
                    llm=self.llm,
                    chain_type="map_reduce"
                )
            
            # Generate summary
            summary = chain.run(documents)
            
            return {
                "summary": summary,
                "num_pages": len(documents),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error summarizing PDF: {str(e)}")
            return {
                "summary": f"Error generating PDF summary: {str(e)}",
                "status": "error"
            }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history
        
        Returns:
            List of conversation exchanges
        """
        try:
            if not self.memory.chat_memory.messages:
                return []
            
            history = []
            messages = self.memory.chat_memory.messages
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    history.append({
                        "user": messages[i].content,
                        "assistant": messages[i + 1].content
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    def clear_memory(self):
        """Clear conversation memory"""
        try:
            self.memory.clear()
            logger.info("Conversation memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the AI agent is working properly
        
        Returns:
            Health status dictionary
        """
        try:
            # Test LLM
            test_response = self.llm("Hello")
            
            return {
                "status": "healthy",
                "model": self.model_name,
                "llm_working": True,
                "embeddings_working": self.embeddings is not None,
                "memory_working": self.memory is not None
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Utility functions
def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary directory
    
    Args:
        uploaded_file: Uploaded file object
        
    Returns:
        Path to saved file
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

def cleanup_temp_file(file_path: str):
    """
    Clean up temporary file
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")
