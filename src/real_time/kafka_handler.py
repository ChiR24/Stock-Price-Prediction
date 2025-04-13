"""
Module for handling Kafka operations for real-time data processing.
"""
import os
import json
import time
from datetime import datetime
import threading
import uuid
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_STOCK_TOPIC, KAFKA_SENTIMENT_TOPIC

# Import Kafka simulator for testing without a real Kafka cluster
try:
    from utils.kafka_simulator import kafka_simulator
    KAFKA_SIMULATOR_AVAILABLE = True
except ImportError:
    KAFKA_SIMULATOR_AVAILABLE = False

# Set up logger
logger = setup_logger('kafka_handler')

class KafkaHandler:
    """
    Class to handle Kafka operations for real-time data processing.
    """
    def __init__(self, bootstrap_servers=None):
        """
        Initialize the Kafka handler.

        Args:
            bootstrap_servers: Kafka bootstrap servers. Defaults to the one in config.
        """
        self.bootstrap_servers = bootstrap_servers if bootstrap_servers else KAFKA_BOOTSTRAP_SERVERS
        self.producer = None
        self.consumers = {}
        self.consumer_threads = {}
        self._running = False

    def create_producer(self):
        """
        Create a Kafka producer.

        Returns:
            True if successful, False otherwise.
        """
        # Try to use the Kafka simulator if real Kafka is not available
        if KAFKA_SIMULATOR_AVAILABLE:
            try:
                # First try to create a real Kafka producer
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                logger.info(f"Kafka producer created with bootstrap servers: {self.bootstrap_servers}")
                return True
            except KafkaError as e:
                logger.warning(f"Failed to create real Kafka producer: {e}. Using simulator instead.")
                # Use the simulator as a fallback
                self.producer = "simulator"
                # Start the simulator
                kafka_simulator.start()
                logger.info("Using Kafka simulator as producer")
                return True
        else:
            # Try to create a real Kafka producer
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                logger.info(f"Kafka producer created with bootstrap servers: {self.bootstrap_servers}")
                return True
            except KafkaError as e:
                logger.error(f"Failed to create Kafka producer: {e}")
                return False

    def send_message(self, topic, message):
        """
        Send a message to a Kafka topic.

        Args:
            topic: Kafka topic.
            message: Message to send.

        Returns:
            Future if successful, None otherwise.
        """
        if not self.producer:
            if not self.create_producer():
                return None

        try:
            # Add timestamp if not present
            if isinstance(message, dict) and 'timestamp' not in message:
                message['timestamp'] = datetime.now().isoformat()

            # Send message
            if self.producer == "simulator" and KAFKA_SIMULATOR_AVAILABLE:
                # Use the simulator
                kafka_simulator.produce(topic, message)
                logger.debug(f"Message sent to simulator topic {topic}")
                return True
            else:
                # Use the real Kafka producer
                future = self.producer.send(topic, message)
                logger.debug(f"Message sent to topic {topic}")
                return future
        except Exception as e:
            logger.error(f"Failed to send message to topic {topic}: {e}")
            return None

    def flush_producer(self):
        """
        Flush the Kafka producer.
        """
        if self.producer:
            self.producer.flush()
            logger.info("Kafka producer flushed")

    def close_producer(self):
        """
        Close the Kafka producer.
        """
        if self.producer:
            self.producer.close()
            self.producer = None
            logger.info("Kafka producer closed")

    def create_consumer(self, topic, group_id=None, auto_offset_reset='latest'):
        """
        Create a Kafka consumer.

        Args:
            topic: Kafka topic or list of topics.
            group_id: Consumer group ID. Defaults to None.
            auto_offset_reset: Auto offset reset strategy. Defaults to 'latest'.

        Returns:
            True if successful, False otherwise.
        """
        # Convert single topic to list
        topics = [topic] if isinstance(topic, str) else topic
        topic_key = '_'.join(topics)

        # Try to use the Kafka simulator if real Kafka is not available
        if KAFKA_SIMULATOR_AVAILABLE:
            try:
                # First try to create a real Kafka consumer
                consumer = KafkaConsumer(
                    *topics,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=group_id,
                    auto_offset_reset=auto_offset_reset,
                    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
                )

                # Store consumer by topic(s)
                self.consumers[topic_key] = consumer

                logger.info(f"Kafka consumer created for topics: {topics}")
                return True
            except KafkaError as e:
                logger.warning(f"Failed to create real Kafka consumer: {e}. Using simulator instead.")
                # Use the simulator as a fallback
                self.consumers[topic_key] = "simulator"
                # Start the simulator
                kafka_simulator.start()
                logger.info(f"Using Kafka simulator as consumer for topics: {topics}")
                return True
        else:
            # Try to create a real Kafka consumer
            try:
                consumer = KafkaConsumer(
                    *topics,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=group_id,
                    auto_offset_reset=auto_offset_reset,
                    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
                )

                # Store consumer by topic(s)
                self.consumers[topic_key] = consumer

                logger.info(f"Kafka consumer created for topics: {topics}")
                return True
            except KafkaError as e:
                logger.error(f"Failed to create Kafka consumer for topics {topic}: {e}")
                return False

    def consume_messages(self, topic, callback, group_id=None, auto_offset_reset='latest'):
        """
        Consume messages from a Kafka topic.

        Args:
            topic: Kafka topic or list of topics.
            callback: Callback function to process messages.
            group_id: Consumer group ID. Defaults to None.
            auto_offset_reset: Auto offset reset strategy. Defaults to 'latest'.

        Returns:
            True if successful, False otherwise.
        """
        # Convert single topic to list
        topics = [topic] if isinstance(topic, str) else topic
        topic_key = '_'.join(topics)

        # Check if consumer exists
        if topic_key not in self.consumers:
            if not self.create_consumer(topics, group_id, auto_offset_reset):
                return False

        try:
            consumer = self.consumers[topic_key]

            # Start consumer thread
            self._running = True

            if consumer == "simulator" and KAFKA_SIMULATOR_AVAILABLE:
                # Use the simulator
                consumer_id = f"consumer_{uuid.uuid4().hex[:8]}"
                kafka_simulator.register_consumer(consumer_id, topics, callback)

                # Store thread info for cleanup
                self.consumer_threads[topic_key] = {
                    "consumer_id": consumer_id,
                    "simulator": True
                }

                logger.info(f"Registered simulator consumer {consumer_id} for topics: {topics}")
                return True
            else:
                # Use the real Kafka consumer
                thread = threading.Thread(
                    target=self._consume_thread,
                    args=(consumer, callback),
                    daemon=True
                )
                thread.start()

                # Store thread info for cleanup
                self.consumer_threads[topic_key] = {
                    "thread": thread,
                    "simulator": False
                }

            logger.info(f"Started consuming messages from topics: {topics}")
            return True
        except Exception as e:
            logger.error(f"Failed to start consuming messages from topics {topics}: {e}")
            return False

    def _consume_thread(self, consumer, callback):
        """
        Thread function for consuming messages.

        Args:
            consumer: Kafka consumer.
            callback: Callback function to process messages.
        """
        try:
            for message in consumer:
                if not self._running:
                    break

                try:
                    # Process message using callback
                    callback(message.topic, message.value)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except Exception as e:
            logger.error(f"Error in consumer thread: {e}")
        finally:
            consumer.close()
            logger.info("Consumer closed")

    def stop_consuming(self):
        """
        Stop consuming messages.
        """
        self._running = False

        # Wait for threads to finish
        for topic_key, thread_info in self.consumer_threads.items():
            if thread_info.get("simulator", False) and KAFKA_SIMULATOR_AVAILABLE:
                # Unregister simulator consumer
                consumer_id = thread_info.get("consumer_id")
                if consumer_id:
                    kafka_simulator.unregister_consumer(consumer_id)
                    logger.info(f"Unregistered simulator consumer {consumer_id}")
            else:
                # Wait for real Kafka consumer thread
                thread = thread_info.get("thread")
                if thread:
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        logger.warning(f"Consumer thread for {topic_key} did not terminate")

        # Close consumers
        for topic_key, consumer in self.consumers.items():
            if consumer != "simulator":
                consumer.close()

        self.consumers = {}
        self.consumer_threads = {}

        # Stop the simulator if it was used
        if KAFKA_SIMULATOR_AVAILABLE and (self.producer == "simulator" or "simulator" in self.consumers.values()):
            kafka_simulator.stop()
            logger.info("Stopped Kafka simulator")

        logger.info("All consumers stopped")

    def close(self):
        """
        Close all Kafka connections.
        """
        self.stop_consuming()
        self.close_producer()
        logger.info("Kafka handler closed")


def demo_message_handler(topic, message):
    """
    Demo callback function for handling Kafka messages.

    Args:
        topic: Kafka topic.
        message: Message received.
    """
    print(f"Received message from topic {topic}: {message}")


def main():
    """
    Main function to test the Kafka handler.
    """
    # Initialize Kafka handler
    handler = KafkaHandler()

    # Test producer
    if handler.create_producer():
        # Send test messages
        handler.send_message(KAFKA_STOCK_TOPIC, {
            "ticker": "AAPL",
            "price": 150.25,
            "timestamp": datetime.now().isoformat()
        })

        handler.send_message(KAFKA_SENTIMENT_TOPIC, {
            "ticker": "AAPL",
            "sentiment": "positive",
            "score": 0.75,
            "text": "Apple's new product is amazing!",
            "timestamp": datetime.now().isoformat()
        })

        # Flush and close producer
        handler.flush_producer()

    # Test consumer
    if handler.create_consumer([KAFKA_STOCK_TOPIC, KAFKA_SENTIMENT_TOPIC], "test-group"):
        # Start consuming messages
        handler.consume_messages(
            [KAFKA_STOCK_TOPIC, KAFKA_SENTIMENT_TOPIC],
            demo_message_handler,
            "test-group"
        )

        # Keep consuming for a while
        try:
            print("Consuming messages (Ctrl+C to stop)...")
            time.sleep(30)
        except KeyboardInterrupt:
            print("Interrupted by user")

        # Stop consuming
        handler.stop_consuming()

    # Close handler
    handler.close()


if __name__ == "__main__":
    main()