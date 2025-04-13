"""
Module for collecting and analyzing social media sentiment data related to stocks.
"""
import os
import time
import json
from datetime import datetime
import pandas as pd
import tweepy
from textblob import TextBlob
import re
from kafka import KafkaProducer

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import (
    RAW_DATA_DIR, 
    SOCIAL_MEDIA_KEYWORDS,
    TWITTER_API_KEY,
    TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_SECRET,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_SENTIMENT_TOPIC
)

# Set up logger
logger = setup_logger('sentiment_collector')

class SentimentCollector:
    """
    Class to collect and analyze social media sentiment data related to stocks.
    """
    def __init__(self, tickers=None, use_kafka=False):
        """
        Initialize the SentimentCollector.
        
        Args:
            tickers: List of stock tickers to collect sentiment for. Defaults to SOCIAL_MEDIA_KEYWORDS keys.
            use_kafka: Whether to send data to Kafka. Defaults to False.
        """
        self.tickers = tickers if tickers else list(SOCIAL_MEDIA_KEYWORDS.keys())
        self.keywords = {ticker: SOCIAL_MEDIA_KEYWORDS.get(ticker, [ticker]) for ticker in self.tickers}
        self.use_kafka = use_kafka
        
        # Initialize Twitter API
        self.twitter_api = None
        if TWITTER_API_KEY and TWITTER_API_SECRET:
            try:
                # Initialize Twitter API v2
                auth = tweepy.OAuth1UserHandler(
                    TWITTER_API_KEY,
                    TWITTER_API_SECRET,
                    TWITTER_ACCESS_TOKEN,
                    TWITTER_ACCESS_SECRET
                )
                self.twitter_api = tweepy.API(auth)
                logger.info("Twitter API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter API: {e}")
        else:
            logger.warning("Twitter API credentials not found. Twitter data collection disabled.")
        
        # Initialize Kafka producer if needed
        if self.use_kafka:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                logger.info(f"Kafka producer initialized with bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
            except Exception as e:
                logger.error(f"Failed to initialize Kafka producer: {e}")
                self.use_kafka = False
    
    def clean_text(self, text):
        """
        Clean text by removing URLs, special characters, etc.
        
        Args:
            text: Text to clean.
            
        Returns:
            Cleaned text.
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Dictionary with sentiment scores.
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Skip empty text
        if not cleaned_text:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'sentiment': 'neutral'
            }
        
        # Analyze sentiment using TextBlob
        analysis = TextBlob(cleaned_text)
        
        # Determine sentiment label
        polarity = analysis.sentiment.polarity
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': analysis.sentiment.subjectivity,
            'sentiment': sentiment
        }
    
    def collect_twitter_sentiment(self, max_tweets=100):
        """
        Collect and analyze Twitter sentiment for each ticker.
        
        Args:
            max_tweets: Maximum number of tweets to collect per ticker. Defaults to 100.
            
        Returns:
            DataFrame with sentiment data.
        """
        if not self.twitter_api:
            logger.error("Twitter API not initialized. Cannot collect Twitter sentiment.")
            return pd.DataFrame()
        
        all_sentiments = []
        
        for ticker, search_terms in self.keywords.items():
            for term in search_terms:
                try:
                    logger.info(f"Collecting Twitter sentiment for {term} (ticker: {ticker})")
                    
                    # Search for tweets
                    tweets = tweepy.Cursor(
                        self.twitter_api.search_tweets,
                        q=f"{term} -filter:retweets",
                        lang="en",
                        tweet_mode="extended",
                        count=100
                    ).items(max_tweets)
                    
                    for tweet in tweets:
                        # Extract tweet data
                        tweet_data = {
                            'ticker': ticker,
                            'search_term': term,
                            'source': 'twitter',
                            'text': tweet.full_text if hasattr(tweet, 'full_text') else tweet.text,
                            'created_at': tweet.created_at.isoformat(),
                            'user': tweet.user.screen_name,
                            'followers': tweet.user.followers_count,
                            'retweet_count': tweet.retweet_count,
                            'favorite_count': tweet.favorite_count if hasattr(tweet, 'favorite_count') else 0,
                            'collected_at': datetime.now().isoformat()
                        }
                        
                        # Add sentiment analysis
                        sentiment = self.analyze_sentiment(tweet_data['text'])
                        tweet_data.update(sentiment)
                        
                        # Send to Kafka if enabled
                        if self.use_kafka:
                            self.producer.send(KAFKA_SENTIMENT_TOPIC, tweet_data)
                        
                        all_sentiments.append(tweet_data)
                    
                    logger.info(f"Collected {len(all_sentiments)} tweets for {term}")
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error collecting Twitter sentiment for {term}: {e}")
        
        # Convert to DataFrame
        if all_sentiments:
            df = pd.DataFrame(all_sentiments)
            
            # Save to CSV
            os.makedirs(os.path.join(RAW_DATA_DIR, 'sentiment'), exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(RAW_DATA_DIR, 'sentiment', f"twitter_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved Twitter sentiment data to {csv_path}")
            
            return df
        
        return pd.DataFrame()
    
    def run_continuous_collection(self, interval_seconds=300, max_iterations=None):
        """
        Run continuous sentiment data collection at specified intervals.
        
        Args:
            interval_seconds: Time in seconds between collections. Defaults to 300 (5 minutes).
            max_iterations: Maximum number of iterations. Defaults to None (run indefinitely).
        """
        iteration = 0
        
        logger.info(f"Starting continuous sentiment collection every {interval_seconds} seconds")
        
        try:
            while max_iterations is None or iteration < max_iterations:
                logger.info(f"Sentiment collection iteration {iteration + 1}")
                
                # Collect Twitter sentiment
                self.collect_twitter_sentiment(max_tweets=50)  # Reduced for testing
                
                iteration += 1
                
                # Sleep until next collection
                if max_iterations is None or iteration < max_iterations:
                    logger.info(f"Sleeping for {interval_seconds} seconds")
                    time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Sentiment collection stopped by user")
        
        finally:
            if self.use_kafka:
                self.producer.flush()
                logger.info("Kafka producer flushed")


def main():
    """
    Main function to run the sentiment collector.
    """
    # Initialize the collector
    collector = SentimentCollector(use_kafka=False)  # Set to True to use Kafka
    
    # Run continuous collection for a limited time (for testing)
    # For production, you might want to set max_iterations to None
    collector.run_continuous_collection(interval_seconds=300, max_iterations=3)


if __name__ == "__main__":
    main() 