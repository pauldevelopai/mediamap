# agent.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import schedule
import datetime
import logging
import json
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
from web_tools import web_search, extract_article_content
from database import init_database, store_article, get_sent_articles
from scoring import score_relevance, send_notification
from config import OPENAI_API_KEY, TOPIC, FREQUENCY_HOURS, NUM_ARTICLES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client

def load_orchestrator_prompt(topic, num_articles=5):
    prompt_path = os.path.join(os.path.dirname(__file__), 'orchestrator_prompt.md')
    try:
        with open(prompt_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        # Provide a default prompt if the file is missing
        return f"You are an AI news orchestrator. Find the top {num_articles} news articles about '{topic}' and summarize them for a media organization."

def handle_tool_call(tool_name, parameters):
    if tool_name == "web_search":
        return web_search(parameters["query"], parameters.get("num_results", 10))
    elif tool_name == "extract_article_content":
        return extract_article_content(parameters["url"])
    elif tool_name == "store_article":
        return store_article(parameters["article_data"])
    elif tool_name == "score_relevance":
        return score_relevance(parameters["article_data"], parameters["topic"])
    elif tool_name == "send_notification":
        return send_notification(parameters["articles"])
    else:
        logger.error(f"Unknown tool: {tool_name}")
        return {"error": f"Unknown tool: {tool_name}"}

def run_news_agent(topic, num_articles=5):
    logger.info(f"Starting news agent run for topic: {topic}")
    try:
        init_database()
        prompt = load_orchestrator_prompt(topic, num_articles)
        sent_urls = get_sent_articles(topic)
        conversation = []
        current_message = f"Find the latest news about {topic} for today. Avoid these URLs which we've already processed: {sent_urls[:5]}..."
        max_turns = 20
        current_turn = 0
        while current_turn < max_turns:
            current_turn += 1
            conversation.append({"role": "user", "content": current_message})
            response = client.chat.completions.create(model="gpt-4",
            messages=[{"role": "system", "content": prompt}] + conversation,
            max_tokens=4000,
            temperature=0.2)
            ai_message = response.choices[0].message.content
            conversation.append({"role": "assistant", "content": ai_message})
            # Tool call simulation: look for tool call markers in ai_message
            if "TOOL_CALL:" in ai_message:
                tool_responses = []
                # Example: TOOL_CALL: web_search {"query": "AI news"}
                for line in ai_message.splitlines():
                    if line.startswith("TOOL_CALL:"):
                        try:
                            tool_call = line[len("TOOL_CALL:"):].strip()
                            tool_name, params_json = tool_call.split(" ", 1)
                            parameters = json.loads(params_json)
                            result = handle_tool_call(tool_name, parameters)
                            tool_responses.append({"tool_name": tool_name, "output": json.dumps(result)})
                        except Exception as e:
                            logger.error(f"Error parsing tool call: {e}")
                current_message = "Here are the results of the tool calls:"
                for tool_response in tool_responses:
                    current_message += f"\n\nTool: {tool_response['tool_name']}\nOutput: {tool_response['output']}"
            else:
                if "Top News Articles on" in ai_message:
                    logger.info("News agent completed successfully")
                    break
                else:
                    current_message = "Continue processing. What's the next step?"
        logger.info(f"News agent run completed for topic: {topic}")
    except Exception as e:
        logger.error(f"Error running news agent: {str(e)}")

def schedule_news_agent(topic, frequency_hours=1, num_articles=5):
    run_news_agent(topic, num_articles)
    schedule.every(frequency_hours).hours.do(run_news_agent, topic, num_articles)
    logger.info(f"News agent scheduled to run every {frequency_hours} hour(s) for topic: {topic}")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    schedule_news_agent(TOPIC, FREQUENCY_HOURS, NUM_ARTICLES)