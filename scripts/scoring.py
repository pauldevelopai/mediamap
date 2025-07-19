# scoring.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
from config import OPENAI_API_KEY, EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT

def score_relevance(article_data, topic):
    system_prompt = f"""
    You are an expert at evaluating the relevance of news articles to specific topics.
    Your task is to score how relevant an article is to the topic: "{topic}".
    Provide a score from 0.0 to 1.0 where:
    - 0.0 means completely irrelevant
    - 0.5 means somewhat relevant
    - 1.0 means extremely relevant
    Only return a single number between 0.0 and 1.0.
    """
    article_text = f"""
    Title: {article_data.get('title', '')}
    Content:
    {article_data.get('content', '')[:2000]}
    """
    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": article_text}
        ],
        max_tokens=10,
        temperature=0.0)
        score_text = response.choices[0].message.content.strip()
        try:
            score = float(score_text)
            score = max(0.0, min(1.0, score))
            return score
        except ValueError:
            return 0.5
    except Exception as e:
        print(f"Error scoring relevance: {str(e)}")
        return 0.5

def send_notification(articles):
    """Send email notification with top articles."""
    # Create message
    message = MIMEMultipart("alternative")
    message["Subject"] = f"Top News Articles - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    message["From"] = EMAIL_SENDER
    message["To"] = EMAIL_RECIPIENT

    # Create HTML content
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .article {{ margin-bottom: 25px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .title {{ font-size: 18px; font-weight: bold; }}
            .meta {{ color: #666; font-size: 14px; margin-bottom: 10px; }}
            .summary {{ margin-bottom: 10px; }}
            .key-points {{ margin-left: 20px; }}
            .link {{ display: inline-block; margin-top: 10px; color: #0066cc; }}
        </style>
    </head>
    <body>
        <h1>Top News Articles - {datetime.now().strftime('%Y-%m-%d %H:%M')}</h1>
    """

    for article in articles:
        html += f"""
        <div class="article">
            <div class="title">{article.get('title', 'No Title')}</div>
            <div class="meta">
                Source: {article.get('source', 'Unknown')} | 
                Published: {article.get('publish_date', 'Unknown')}
            </div>
            <div class="summary">{article.get('summary', 'No summary available.')}</div>
            <div>Key Points:</div>
            <ul class="key-points">
        """

        # Add key points
        for point in article.get('key_points', ['No key points available.']):
            html += f"<li>{point}</li>"

        html += f"""
            </ul>
            <a class="link" href="{article.get('url', '#')}">Read Full Article</a>
        </div>
        """

    html += """
    </body>
    </html>
    """

    # Attach HTML content
    part = MIMEText(html, "html")
    message.attach(part)

    # Send email
    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, message.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending notification: {str(e)}")
        return False