# AgentForge - AI Agent Management System

A complete FastAPI-based system for creating, managing, and deploying AI agents with RAG capabilities.

## Features

✅ **Agent Management**
- Create and manage multiple AI agents
- Choose from different AI models (Llama, Mixtral, Gemma)
- Track agent performance and statistics

✅ **Knowledge Management**  
- Upload PDF documents
- Add web URLs as knowledge sources
- Direct text input
- Vector storage with FAISS

✅ **Chat Interface**
- Real-time chat with agents
- Conversation history
- Session management

✅ **Embedding & Integration**
- Generate iframe embed codes
- JavaScript widget integration
- Agent performance tracking

✅ **Analytics**
- Conversation statistics
- Agent performance metrics
- Knowledge source tracking

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
The `.env` file is already configured with your API keys:
- ✅ GROQ_API_KEY (for AI models)
- ✅ LANGSMITH_API_KEY (for tracing)
- ✅ HF_TOKEN (for embeddings)

### 3. Run the Application
```bash
python main.py
```

The application will start on `http://localhost:8000`

### 4. Access the Interface
- **Web UI**: http://localhost:8000/ui
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## File Structure
```
agentforge/
├── main.py              # Backend API server
├── static/
│   └── index.html       # Frontend interface
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
├── agentforge.db       # SQLite database (auto-created)
└── vectorstores/       # FAISS vector storage (auto-created)
```

## Usage

### 1. Create an Agent
- Go to "Create Agent" tab
- Enter agent name and description
- Select AI model (llama3-8b-8192 recommended for speed)
- Click "Create Agent"

### 2. Add Knowledge
- Go to "Knowledge" tab
- Select your agent
- Upload PDF, add URL, or paste text
- Knowledge is automatically processed and indexed

### 3. Chat with Agent
- Go to "Chat" tab
- Select your agent
- Start chatting! Agent will use uploaded knowledge to answer

### 4. Embed Agent
- Go to "Embed" tab
- Select your agent
- Copy iframe or JavaScript code
- Embed on your website

### 5. Monitor Performance
- Go to "Statistics" tab
- View conversation counts, ratings, and performance metrics

## API Endpoints

### Agents
- `POST /agents` - Create new agent
- `GET /agents` - List all agents
- `GET /agents/{id}` - Get agent details
- `DELETE /agents/{id}` - Delete agent

### Knowledge
- `POST /agents/{id}/knowledge` - Upload knowledge
- `GET /agents/{id}/knowledge` - List knowledge sources

### Chat
- `POST /agents/{id}/chat` - Chat with agent
- `GET /agents/{id}/conversations` - Get conversation history

### Analytics
- `GET /agents/{id}/stats` - Get agent statistics
- `GET /agents/{id}/embed` - Get embed codes

## Technical Details

### AI Models Available
- **llama3-8b-8192** - Fast, efficient for most tasks
- **llama3-70b-8192** - More powerful, slower
- **mixtral-8x7b-32768** - Large context window
- **gemma-7b-it** - Google's instruction-tuned model

### Vector Storage
- Uses FAISS for efficient similarity search
- Sentence transformers for embeddings
- Automatic chunking and indexing

### Database Schema
- SQLite for simplicity and portability
- Agents, knowledge sources, conversations, embeddings tables
- Automatic performance tracking

## Production Deployment

### Environment Variables
Make sure to set these in production:
```bash
export GROQ_API_KEY="your-groq-key"
export LANGSMITH_API_KEY="your-langsmith-key" 
export HF_TOKEN="your-huggingface-token"
```

### Run with Gunicorn
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "main.py"]
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **FAISS Installation Issues**
   ```bash
   pip install faiss-cpu --no-cache-dir
   ```

3. **Permission Errors**
   ```bash
   chmod +x main.py
   ```

4. **Port Already in Use**
   - Change port in main.py: `uvicorn.run(app, port=8001)`

### Performance Optimization

1. **For High Traffic**
   - Use PostgreSQL instead of SQLite
   - Implement Redis for caching
   - Use multiple workers

2. **For Large Knowledge Bases**
   - Increase chunk size in text splitter
   - Use more powerful embedding models
   - Implement knowledge source pagination

## License

MIT License - Feel free to use and modify for your projects!

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check application logs for error details