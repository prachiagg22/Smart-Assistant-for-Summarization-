"""
Flask API endpoints for AI Chatbot
Provides REST API for frontend interaction
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from backend import AIAgent, save_uploaded_file, cleanup_temp_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Global AI agent instance
ai_agent = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_ai_agent():
    """Get or create AI agent instance"""
    global ai_agent
    if ai_agent is None:
        try:
            ai_agent = AIAgent()
            logger.info("AI Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI Agent: {str(e)}")
            raise
    return ai_agent

# API Routes

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        agent = get_ai_agent()
        health_status = agent.health_check()
        return jsonify(health_status), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle regular chat conversation
    
    Expected JSON payload:
    {
        "message": "user message",
        "session_id": "optional session id"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Message is required"
            }), 400
        
        message = data['message']
        session_id = data.get('session_id', 'default')
        
        # Get AI agent and process message
        agent = get_ai_agent()
        response = agent.chat(message, session_id)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    """
    Handle PDF upload and processing
    
    Expected: multipart/form-data with 'file' field
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded"
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "error": "No file selected"
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "error": "Only PDF files are allowed"
            }), 400
        
        # Save uploaded file
        temp_path = save_uploaded_file(file)
        
        try:
            # Process PDF with AI agent
            agent = get_ai_agent()
            result = agent.process_pdf(temp_path)
            
            return jsonify(result), 200
            
        finally:
            # Clean up temporary file
            cleanup_temp_file(temp_path)
            
    except Exception as e:
        logger.error(f"Error in upload PDF endpoint: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/chat-pdf', methods=['POST'])
def chat_with_pdf():
    """
    Handle chat with uploaded PDF
    
    Expected JSON payload:
    {
        "question": "user question about PDF"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "error": "Question is required"
            }), 400
        
        question = data['question']
        
        # Get AI agent and process question
        agent = get_ai_agent()
        response = agent.chat_with_pdf(question)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in chat with PDF endpoint: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/summarize-text', methods=['POST'])
def summarize_text():
    """
    Handle text summarization
    
    Expected JSON payload:
    {
        "text": "text to summarize",
        "summary_type": "stuff|map_reduce|refine" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Text is required"
            }), 400
        
        text = data['text']
        summary_type = data.get('summary_type', 'stuff')
        
        # Get AI agent and summarize text
        agent = get_ai_agent()
        response = agent.summarize_text(text, summary_type)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in summarize text endpoint: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/summarize-pdf', methods=['POST'])
def summarize_pdf():
    """
    Handle PDF summarization
    
    Expected: multipart/form-data with 'file' field
    Optional: 'summary_type' field
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded"
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "error": "No file selected"
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "error": "Only PDF files are allowed"
            }), 400
        
        summary_type = request.form.get('summary_type', 'map_reduce')
        
        # Save uploaded file
        temp_path = save_uploaded_file(file)
        
        try:
            # Summarize PDF with AI agent
            agent = get_ai_agent()
            result = agent.summarize_pdf(temp_path, summary_type)
            
            return jsonify(result), 200
            
        finally:
            # Clean up temporary file
            cleanup_temp_file(temp_path)
            
    except Exception as e:
        logger.error(f"Error in summarize PDF endpoint: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/history', methods=['GET'])
def get_conversation_history():
    """Get conversation history"""
    try:
        agent = get_ai_agent()
        history = agent.get_conversation_history()
        
        return jsonify({
            "history": history,
            "status": "success"
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/clear-memory', methods=['POST'])
def clear_conversation_memory():
    """Clear conversation memory"""
    try:
        agent = get_ai_agent()
        agent.clear_memory()
        
        return jsonify({
            "message": "Conversation memory cleared",
            "status": "success"
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        "error": "File too large. Maximum size is 16MB."
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize AI agent on startup
    try:
        get_ai_agent()
        logger.info("AI Agent initialized successfully on startup")
    except Exception as e:
        logger.error(f"Failed to initialize AI Agent on startup: {str(e)}")
        print(f"Warning: AI Agent failed to initialize. Error: {str(e)}")
        print("Please ensure Ollama is installed and running with the llama2 model.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
