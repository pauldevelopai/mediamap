# ChatGPT Agents Integration Guide

## Overview

This guide explains how to use the ChatGPT Agents integration with your AI agents system. The integration leverages OpenAI's ChatGPT Agents framework using the Assistants API with built-in tools for web search, code execution, and data analysis.

## Features

### 🤖 ChatGPT Agents Capabilities

- **MediaMap Business Intelligence Agent**: Specialized for media industry analysis
- **HealthPIN Clinical Intelligence Agent**: Specialized for healthcare analysis
- **Built-in Tools**: Web search, code interpreter, file search
- **Real-time Analysis**: Continuous learning and insight generation
- **API Integration**: RESTful API for all agent operations

### 🛠️ Available Tools

Each ChatGPT Agent comes with:
- **Web Search**: Real-time information gathering
- **Code Interpreter**: Data analysis and computation
- **File Search**: Document and knowledge base search

## Setup

### 1. Environment Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 2. Installation

The required packages are already included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Verification

Run the test script to verify integration:

```bash
cd backend
python test_chatgpt_agents.py
```

## Usage

### Starting the Agents

```bash
# Start all agents
curl -X POST http://localhost:5000/api/agents/start \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token"
```

### Checking Status

```bash
# Get agent status
curl http://localhost:5000/api/agents/status \
  -H "Authorization: Bearer your_token"
```

### ChatGPT Agent Capabilities

```bash
# Get ChatGPT agent capabilities
curl http://localhost:5000/api/agents/chatgpt/capabilities \
  -H "Authorization: Bearer your_token"
```

### Data Analysis

```bash
# Analyze data with ChatGPT Agent
curl -X POST http://localhost:5000/api/agents/chatgpt/mediamap/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token" \
  -d '{
    "analysis_type": "insights",
    "data": {
      "content": "AI is transforming media workflows",
      "category": "AI_Technology",
      "relevance_score": 0.9
    }
  }'
```

### Getting Recommendations

```bash
# Get ChatGPT recommendations
curl http://localhost:5000/api/agents/chatgpt/mediamap/recommendations?type=recommendations \
  -H "Authorization: Bearer your_token"
```

## API Endpoints

### Core Agent Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents/status` | GET | Get status of all agents |
| `/api/agents/start` | POST | Start all agents |
| `/api/agents/stop` | POST | Stop all agents |
| `/api/agents/{agent_name}/cycle` | POST | Run single learning cycle |
| `/api/agents/{agent_name}/insights` | GET | Get agent insights |
| `/api/agents/{agent_name}/knowledge` | GET | Get agent knowledge |

### ChatGPT Agent Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents/chatgpt/capabilities` | GET | Get ChatGPT agent capabilities |
| `/api/agents/chatgpt/status` | GET | Get ChatGPT integration status |
| `/api/agents/chatgpt/{agent_name}/recommendations` | GET | Get ChatGPT recommendations |
| `/api/agents/chatgpt/{agent_name}/analyze` | POST | Analyze data with ChatGPT |

### Section-Specific Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents/mediamap/insights` | GET | MediaMap insights |
| `/api/agents/mediamap/recommendations` | GET | Business recommendations |
| `/api/agents/mediamap/trends` | GET | Industry trends |
| `/api/agents/healthpin/insights` | GET | HealthPIN insights |
| `/api/agents/healthpin/recommendations` | GET | Clinical recommendations |
| `/api/agents/healthpin/trends` | GET | Clinical trends |

## Agent Types

### MediaMap Business Intelligence Agent

**Purpose**: Media industry business intelligence and strategic analysis

**Specializations**:
- Media business models and revenue strategies
- AI and technology adoption in media
- Audience engagement and analytics
- Content creation and distribution
- Industry trends and market opportunities

**Data Sources**:
- Media industry RSS feeds
- News websites
- Social media trends
- Industry reports

### HealthPIN Clinical Intelligence Agent

**Purpose**: Healthcare clinical intelligence and evidence-based analysis

**Specializations**:
- Clinical care protocols and patient outcomes
- Healthcare technology and AI applications
- Medical research and evidence-based practices
- Patient safety and quality improvement
- Healthcare operations and efficiency

**Data Sources**:
- Medical news RSS feeds
- Research publications
- Healthcare technology feeds
- Clinical guidelines

## Analysis Types

### 1. Insights Analysis
Extract key insights and patterns from collected data.

```json
{
  "analysis_type": "insights",
  "data": {
    "content": "Your data content here",
    "category": "Data category",
    "relevance_score": 0.9
  }
}
```

### 2. Recommendations Analysis
Generate actionable recommendations based on data analysis.

```json
{
  "analysis_type": "recommendations",
  "data": {
    "insights": ["insight1", "insight2"],
    "context": "Business context"
  }
}
```

### 3. Trends Analysis
Identify trends and patterns in the data.

```json
{
  "analysis_type": "trends",
  "data": {
    "historical_data": [...],
    "time_period": "30_days"
  }
}
```

## Response Format

### Successful Analysis Response

```json
{
  "success": true,
  "analysis": "Detailed analysis from ChatGPT Agent...",
  "agent_type": "mediamap",
  "analysis_type": "insights",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Response

```json
{
  "success": false,
  "error": "Error description",
  "agent_type": "mediamap"
}
```

## Configuration

### Agent Configuration

Agents are configured in `agent_manager.py`:

```python
# MediaMap Agent Configuration
mediamap_config = AgentConfig(
    name="MediaMapAgent",
    section="mediamap",
    data_sources=[...],
    learning_interval=30,  # minutes
    max_data_points=1000,
    api_keys={
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "twitter": os.getenv("TWITTER_API_KEY", ""),
        "linkedin": os.getenv("LINKEDIN_API_KEY", "")
    },
    storage_path=os.path.join(self.storage_path, "mediamap")
)
```

### ChatGPT Agent Instructions

Agent instructions can be customized in `openai_agent_integration.py`:

```python
mediamap_assistant = self.client.beta.assistants.create(
    name="MediaMap Business Intelligence Agent",
    instructions="""
    You are a specialized AI agent for media industry business intelligence...
    """,
    model="gpt-4-turbo-preview",
    tools=[
        {"type": "web_search"},
        {"type": "code_interpreter"},
        {"type": "file_search"}
    ]
)
```

## Monitoring and Performance

### Performance Metrics

```bash
# Get performance metrics
curl http://localhost:5000/api/agents/performance \
  -H "Authorization: Bearer your_token"
```

### ChatGPT Integration Status

```bash
# Check ChatGPT integration status
curl http://localhost:5000/api/agents/chatgpt/status \
  -H "Authorization: Bearer your_token"
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Not Found**
   ```
   Error: OpenAI API key is required
   Solution: Set OPENAI_API_KEY environment variable
   ```

2. **ChatGPT Agent Initialization Failed**
   ```
   Error: Failed to initialize ChatGPT Agent
   Solution: Check API key and network connectivity
   ```

3. **Analysis Timeout**
   ```
   Error: Analysis failed with status: timeout
   Solution: Check OpenAI API limits and retry
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Integration

Run the test script to verify everything is working:

```bash
python test_chatgpt_agents.py
```

## Best Practices

### 1. Data Quality
- Ensure data is relevant and well-structured
- Use appropriate relevance scores
- Include meaningful metadata

### 2. Analysis Types
- Use "insights" for pattern discovery
- Use "recommendations" for actionable advice
- Use "trends" for temporal analysis

### 3. Performance
- Monitor API usage and costs
- Use appropriate learning intervals
- Implement error handling and retries

### 4. Security
- Keep API keys secure
- Use environment variables
- Implement proper authentication

## Examples

### Complete Workflow Example

```python
# 1. Start agents
response = requests.post('/api/agents/start')

# 2. Wait for data collection
time.sleep(300)  # 5 minutes

# 3. Get insights
insights = requests.get('/api/agents/mediamap/insights')

# 4. Analyze with ChatGPT
analysis = requests.post('/api/agents/chatgpt/mediamap/analyze', json={
    'analysis_type': 'insights',
    'data': insights.json()['insights']
})

# 5. Get recommendations
recommendations = requests.get('/api/agents/chatgpt/mediamap/recommendations')
```

## Support

For issues or questions:

1. Check the test script output
2. Review the logs for error messages
3. Verify API key configuration
4. Test with simple data first

## Updates

The ChatGPT Agents integration is designed to be extensible. New agent types, analysis methods, and tools can be added by:

1. Creating new agent configurations
2. Adding new analysis types
3. Extending the API endpoints
4. Updating the frontend interfaces



