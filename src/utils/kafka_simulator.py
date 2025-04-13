"""
Kafka simulator for testing without a real Kafka cluster.
This module provides a simple in-memory implementation of Kafka-like functionality.
"""
import threading
import time
import queue
import json
import logging
from datetime import datetime

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('kafka_simulator')

class KafkaSimulator:
    """
    A simple in-memory Kafka simulator for testing.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(KafkaSimulator, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.topics = {}
        self.consumers = {}
        self.running = False
        self.thread = None
        logger.info("KafkaSimulator initialized")
    
    def create_topic(self, topic_name):
        """
        Create a new topic.
        
        Args:
            topic_name: Name of the topic to create.
        """
        if topic_name not in self.topics:
            self.topics[topic_name] = queue.Queue()
            logger.info(f"Created topic: {topic_name}")
    
    def produce(self, topic_name, message):
        """
        Produce a message to a topic.
        
        Args:
            topic_name: Name of the topic to produce to.
            message: Message to produce (will be converted to JSON).
        """
        if topic_name not in self.topics:
            self.create_topic(topic_name)
        
        # Add timestamp to message
        if isinstance(message, dict):
            message['timestamp'] = datetime.now().isoformat()
        
        # Convert message to JSON string
        message_str = json.dumps(message) if isinstance(message, (dict, list)) else str(message)
        
        # Add to queue
        self.topics[topic_name].put(message_str)
        logger.debug(f"Produced message to {topic_name}: {message_str[:100]}...")
    
    def register_consumer(self, consumer_id, topics, callback):
        """
        Register a consumer for one or more topics.
        
        Args:
            consumer_id: Unique identifier for the consumer.
            topics: List of topic names to consume from.
            callback: Function to call with consumed messages.
        """
        if consumer_id in self.consumers:
            logger.warning(f"Consumer {consumer_id} already registered, updating subscription")
        
        # Create any topics that don't exist
        for topic in topics:
            if topic not in self.topics:
                self.create_topic(topic)
        
        # Register consumer
        self.consumers[consumer_id] = {
            'topics': topics,
            'callback': callback,
            'last_processed': {topic: 0 for topic in topics}
        }
        
        logger.info(f"Registered consumer {consumer_id} for topics: {topics}")
    
    def unregister_consumer(self, consumer_id):
        """
        Unregister a consumer.
        
        Args:
            consumer_id: Unique identifier for the consumer to unregister.
        """
        if consumer_id in self.consumers:
            del self.consumers[consumer_id]
            logger.info(f"Unregistered consumer {consumer_id}")
    
    def start(self):
        """
        Start the simulator.
        """
        if self.running:
            logger.warning("Simulator already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Started KafkaSimulator")
    
    def stop(self):
        """
        Stop the simulator.
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info("Stopped KafkaSimulator")
    
    def _run(self):
        """
        Main loop for the simulator.
        """
        while self.running:
            # Process messages for each consumer
            for consumer_id, consumer in self.consumers.items():
                for topic in consumer['topics']:
                    # Check if topic exists
                    if topic not in self.topics:
                        continue
                    
                    # Get queue for topic
                    topic_queue = self.topics[topic]
                    
                    # Process all available messages
                    try:
                        while not topic_queue.empty():
                            message = topic_queue.get(block=False)
                            
                            try:
                                # Parse JSON message
                                message_obj = json.loads(message)
                                
                                # Call callback
                                consumer['callback'](topic, message_obj)
                                
                                # Update last processed
                                consumer['last_processed'][topic] += 1
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse message as JSON: {message}")
                            except Exception as e:
                                logger.error(f"Error processing message: {e}")
                    except queue.Empty:
                        pass
            
            # Sleep to avoid high CPU usage
            time.sleep(0.1)

# Global instance
kafka_simulator = KafkaSimulator()
