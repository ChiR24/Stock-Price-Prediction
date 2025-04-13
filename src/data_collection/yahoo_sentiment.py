"""
Module for collecting sentiment data from Yahoo Finance news.
This module provides functionality to collect and analyze sentiment from Yahoo Finance news articles.
"""
import os
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
import random

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import RAW_DATA_DIR, SOCIAL_MEDIA_KEYWORDS

# Set up logger
logger = setup_logger('yahoo_sentiment')

class YahooSentimentCollector:
    """
    Class for collecting and analyzing sentiment from Yahoo Finance news articles.
    """
    def __init__(self, use_kafka=False):
        """
        Initialize the YahooSentimentCollector.

        Args:
            use_kafka: Whether to use Kafka for data streaming. Defaults to False.
        """
        self.use_kafka = use_kafka

        # Initialize Kafka handler if needed
        if self.use_kafka:
            try:
                from real_time.kafka_handler import KafkaHandler
                from utils.config import KAFKA_SENTIMENT_TOPIC

                self.kafka_handler = KafkaHandler()
                self.kafka_topic = KAFKA_SENTIMENT_TOPIC
                logger.info(f"Initialized Kafka handler for topic: {self.kafka_topic}")
            except Exception as e:
                logger.error(f"Failed to initialize Kafka handler: {e}")
                self.use_kafka = False

        logger.info("YahooSentimentCollector initialized")

    def collect_news(self, ticker, max_news=50):
        """
        Collect news articles for a specific ticker from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol.
            max_news: Maximum number of news articles to collect. Defaults to 50.

        Returns:
            List of news articles.
        """
        try:
            # Get ticker info from yfinance
            stock = yf.Ticker(ticker)

            try:
                # Get news from Yahoo Finance
                news = stock.news

                if not news:
                    logger.warning(f"No news found for {ticker}")
                    return []

                # Limit the number of news articles
                news = news[:min(len(news), max_news)]

                logger.info(f"Found {len(news)} news articles for {ticker}")
                return news
            except Exception as e:
                logger.warning(f"Error getting news from yfinance: {e}. Using simulated data.")
                return []

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

    def analyze_sentiment(self, articles):
        """
        Analyze sentiment of news articles.

        Args:
            articles: List of news articles.

        Returns:
            DataFrame with sentiment analysis results.
        """
        if not articles:
            logger.warning("No articles to analyze")
            return pd.DataFrame()

        # Extract relevant information and analyze sentiment
        results = []

        for article in articles:
            try:
                # Extract article information
                title = article.get("title", "")

                # Get the article content if available
                content = ""
                if "link" in article:
                    try:
                        # Add a delay to avoid rate limiting
                        time.sleep(random.uniform(0.5, 1.5))

                        # Fetch the article content
                        response = requests.get(article["link"], timeout=10)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')

                            # Extract paragraphs
                            paragraphs = soup.find_all('p')
                            content = ' '.join([p.get_text() for p in paragraphs[:10]])  # Limit to first 10 paragraphs
                    except Exception as e:
                        logger.warning(f"Error fetching article content: {e}")

                # Use summary if available
                summary = article.get("summary", "")

                # Combine text for sentiment analysis
                text = f"{title} {summary} {content}"

                # Analyze sentiment using TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                # Determine sentiment label
                if polarity > 0.1:
                    sentiment = "positive"
                elif polarity < -0.1:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

                # Format the published date
                published_at = datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat()

                # Add to results
                results.append({
                    "title": title,
                    "summary": summary,
                    "source": article.get("publisher", ""),
                    "published_at": published_at,
                    "url": article.get("link", ""),
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "sentiment": sentiment
                })

            except Exception as e:
                logger.error(f"Error analyzing article: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Convert published_at to datetime
        if "published_at" in df.columns and not df.empty:
            df["published_at"] = pd.to_datetime(df["published_at"])

        return df

    def collect_sentiment_data(self, ticker, save=True):
        """
        Collect and analyze sentiment data for a specific ticker.

        Args:
            ticker: Stock ticker symbol.
            save: Whether to save the data to disk. Defaults to True.

        Returns:
            DataFrame with sentiment analysis results.
        """
        # Collect news articles
        articles = self.collect_news(ticker)

        # Analyze sentiment
        sentiment_df = self.analyze_sentiment(articles)

        # Save to disk if requested
        if save and not sentiment_df.empty:
            self._save_sentiment_data(ticker, sentiment_df)

        # Send to Kafka if enabled
        if self.use_kafka and not sentiment_df.empty:
            self._send_to_kafka(ticker, sentiment_df)

        # If no data was found, log a warning
        if sentiment_df.empty:
            logger.warning(f"No sentiment data found for {ticker}. Please try again later or check the ticker symbol.")

        return sentiment_df



    def _save_sentiment_data(self, ticker, df):
        """
        Save sentiment data to disk.

        Args:
            ticker: Stock ticker symbol.
            df: DataFrame with sentiment data.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.join(RAW_DATA_DIR, 'sentiment'), exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_sentiment_{timestamp}.csv"
            filepath = os.path.join(RAW_DATA_DIR, 'sentiment', filename)

            # Save to CSV
            df.to_csv(filepath, index=False)
            logger.info(f"Saved sentiment data to {filepath}")

            # Also save a JSON version for easier inspection
            json_filepath = filepath.replace('.csv', '.json')
            df.to_json(json_filepath, orient='records', date_format='iso')
            logger.info(f"Saved sentiment data to {json_filepath}")

            return True

        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")
            return False

    def _send_to_kafka(self, ticker, df):
        """
        Send sentiment data to Kafka.

        Args:
            ticker: Stock ticker symbol.
            df: DataFrame with sentiment data.
        """
        if not self.use_kafka:
            return

        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')

            # Add ticker and timestamp to each record
            for record in records:
                record['ticker'] = ticker
                record['timestamp'] = datetime.now().isoformat()

            # Send each record to Kafka
            for record in records:
                self.kafka_handler.send_message(self.kafka_topic, record)
                # Add a small delay to avoid overwhelming the broker
                time.sleep(0.01)

            logger.info(f"Sent {len(records)} sentiment records to Kafka topic {self.kafka_topic}")

            return True

        except Exception as e:
            logger.error(f"Error sending sentiment data to Kafka: {e}")
            return False
