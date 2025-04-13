"""
Module for collecting news sentiment data for stock market prediction.
This module provides functionality to collect and analyze sentiment from news articles.
"""
import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob
import time

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import RAW_DATA_DIR, SOCIAL_MEDIA_KEYWORDS, NEWS_API_KEY

# Set up logger
logger = setup_logger('news_sentiment')

class NewsSentimentCollector:
    """
    Class for collecting and analyzing sentiment from news articles.
    """
    def __init__(self, api_key=None, use_kafka=False):
        """
        Initialize the NewsSentimentCollector.
        
        Args:
            api_key: API key for News API. Defaults to None (uses the one in config).
            use_kafka: Whether to use Kafka for data streaming. Defaults to False.
        """
        self.api_key = api_key if api_key else os.getenv('NEWS_API_KEY', NEWS_API_KEY)
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
        
        logger.info("NewsSentimentCollector initialized")
    
    def collect_news(self, ticker, days_back=7):
        """
        Collect news articles for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol.
            days_back: Number of days to look back. Defaults to 7.
            
        Returns:
            List of news articles.
        """
        # Get keywords for the ticker
        keywords = SOCIAL_MEDIA_KEYWORDS.get(ticker, [ticker])
        
        # Create query string (OR operator between keywords)
        query = " OR ".join([f'"{keyword}"' for keyword in keywords])
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")
        
        # Prepare API request
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.api_key
        }
        
        try:
            # Make API request
            logger.info(f"Fetching news for {ticker} from {from_date} to {to_date}")
            response = requests.get(url, params=params)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                logger.info(f"Found {len(articles)} news articles for {ticker}")
                return articles
            else:
                logger.error(f"Failed to fetch news: {response.status_code} - {response.text}")
                return []
        
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
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
            # Extract article information
            title = article.get("title", "")
            description = article.get("description", "")
            content = article.get("content", "")
            published_at = article.get("publishedAt", "")
            source = article.get("source", {}).get("name", "")
            url = article.get("url", "")
            
            # Combine text for sentiment analysis
            text = f"{title} {description} {content}"
            
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
            
            # Add to results
            results.append({
                "title": title,
                "description": description,
                "source": source,
                "published_at": published_at,
                "url": url,
                "polarity": polarity,
                "subjectivity": subjectivity,
                "sentiment": sentiment
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Convert published_at to datetime
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"])
        
        return df
    
    def collect_sentiment_data(self, ticker, days_back=7, save=True):
        """
        Collect and analyze sentiment data for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol.
            days_back: Number of days to look back. Defaults to 7.
            save: Whether to save the data to disk. Defaults to True.
            
        Returns:
            DataFrame with sentiment analysis results.
        """
        # Check if API key is available
        if not self.api_key or self.api_key == "YOUR_NEWS_API_KEY_HERE":
            logger.warning(f"News API key not set. Using simulated sentiment data for {ticker}")
            return self._generate_simulated_data(ticker, days_back)
        
        # Collect news articles
        articles = self.collect_news(ticker, days_back)
        
        # Analyze sentiment
        sentiment_df = self.analyze_sentiment(articles)
        
        # Save to disk if requested
        if save and not sentiment_df.empty:
            self._save_sentiment_data(ticker, sentiment_df)
        
        # Send to Kafka if enabled
        if self.use_kafka and not sentiment_df.empty:
            self._send_to_kafka(ticker, sentiment_df)
        
        return sentiment_df
    
    def _generate_simulated_data(self, ticker, days_back=7):
        """
        Generate simulated sentiment data when API key is not available.
        
        Args:
            ticker: Stock ticker symbol.
            days_back: Number of days to look back. Defaults to 7.
            
        Returns:
            DataFrame with simulated sentiment data.
        """
        logger.info(f"Generating simulated sentiment data for {ticker}")
        
        # Generate dates
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(days_back)]
        
        # Generate random sentiment data
        import random
        
        results = []
        for date in dates:
            # Generate 1-5 articles per day
            num_articles = random.randint(1, 5)
            
            for _ in range(num_articles):
                # Generate random sentiment
                polarity = random.uniform(-1.0, 1.0)
                subjectivity = random.uniform(0.0, 1.0)
                
                # Determine sentiment label
                if polarity > 0.1:
                    sentiment = "positive"
                elif polarity < -0.1:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                # Add to results
                results.append({
                    "title": f"Simulated news about {ticker}",
                    "description": f"This is a simulated news article about {ticker}",
                    "source": "Simulation",
                    "published_at": date.isoformat(),
                    "url": "https://example.com",
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "sentiment": sentiment
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Convert published_at to datetime
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"])
        
        # Save to disk
        self._save_sentiment_data(ticker, df, simulated=True)
        
        return df
    
    def _save_sentiment_data(self, ticker, df, simulated=False):
        """
        Save sentiment data to disk.
        
        Args:
            ticker: Stock ticker symbol.
            df: DataFrame with sentiment data.
            simulated: Whether the data is simulated. Defaults to False.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.join(RAW_DATA_DIR, 'sentiment'), exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "simulated_" if simulated else ""
            filename = f"{prefix}{ticker}_sentiment_{timestamp}.csv"
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
