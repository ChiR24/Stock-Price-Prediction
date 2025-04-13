"""
Module for handling MongoDB operations for the stock market prediction project.
"""
import os
import json
import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.config import MONGODB_URI, MONGODB_DB

# Set up logger
logger = setup_logger('mongodb_handler')

class MongoDBHandler:
    """
    Class to handle MongoDB operations for the stock market prediction project.
    """
    def __init__(self, uri=None, db_name=None):
        """
        Initialize the MongoDB handler.
        
        Args:
            uri: MongoDB connection URI. Defaults to the one in config.
            db_name: MongoDB database name. Defaults to the one in config.
        """
        self.uri = uri if uri else MONGODB_URI
        self.db_name = db_name if db_name else MONGODB_DB
        self.client = None
        self.db = None
        
        # Connect to MongoDB
        self.connect()
    
    def connect(self):
        """
        Connect to MongoDB.
        
        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            self.client = MongoClient(self.uri)
            # Check if connection is established
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            logger.info(f"Connected to MongoDB: {self.uri}, database: {self.db_name}")
            return True
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def close(self):
        """
        Close MongoDB connection.
        """
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def insert_documents(self, collection_name, documents):
        """
        Insert documents into a collection.
        
        Args:
            collection_name: Name of the collection.
            documents: List of documents or a single document.
            
        Returns:
            List of inserted document IDs or None if failed.
        """
        if not self.db:
            logger.error("MongoDB not connected")
            return None
        
        try:
            collection = self.db[collection_name]
            
            # Convert to list if a single document
            if not isinstance(documents, list):
                documents = [documents]
            
            # Insert documents
            result = collection.insert_many(documents)
            logger.info(f"Inserted {len(result.inserted_ids)} documents into {collection_name}")
            return result.inserted_ids
        except Exception as e:
            logger.error(f"Failed to insert documents into {collection_name}: {e}")
            return None
    
    def find_documents(self, collection_name, query=None, projection=None, sort=None, limit=0):
        """
        Find documents in a collection.
        
        Args:
            collection_name: Name of the collection.
            query: Query filter.
            projection: Fields to include or exclude.
            sort: Sort criteria.
            limit: Maximum number of documents to return.
            
        Returns:
            List of documents or empty list if none found.
        """
        if not self.db:
            logger.error("MongoDB not connected")
            return []
        
        try:
            collection = self.db[collection_name]
            
            # Set default query if None
            if query is None:
                query = {}
            
            # Find documents
            cursor = collection.find(query, projection)
            
            # Apply sort if provided
            if sort:
                cursor = cursor.sort(sort)
            
            # Apply limit if provided
            if limit > 0:
                cursor = cursor.limit(limit)
            
            # Convert cursor to list
            documents = list(cursor)
            logger.info(f"Found {len(documents)} documents in {collection_name}")
            return documents
        except Exception as e:
            logger.error(f"Failed to find documents in {collection_name}: {e}")
            return []
    
    def update_documents(self, collection_name, query, update, upsert=False, many=True):
        """
        Update documents in a collection.
        
        Args:
            collection_name: Name of the collection.
            query: Query filter.
            update: Update operations.
            upsert: Whether to insert if not exists.
            many: Whether to update multiple documents.
            
        Returns:
            Number of documents updated or None if failed.
        """
        if not self.db:
            logger.error("MongoDB not connected")
            return None
        
        try:
            collection = self.db[collection_name]
            
            if many:
                result = collection.update_many(query, update, upsert=upsert)
                modified_count = result.modified_count
            else:
                result = collection.update_one(query, update, upsert=upsert)
                modified_count = result.modified_count
            
            logger.info(f"Updated {modified_count} documents in {collection_name}")
            return modified_count
        except Exception as e:
            logger.error(f"Failed to update documents in {collection_name}: {e}")
            return None
    
    def delete_documents(self, collection_name, query, many=True):
        """
        Delete documents from a collection.
        
        Args:
            collection_name: Name of the collection.
            query: Query filter.
            many: Whether to delete multiple documents.
            
        Returns:
            Number of documents deleted or None if failed.
        """
        if not self.db:
            logger.error("MongoDB not connected")
            return None
        
        try:
            collection = self.db[collection_name]
            
            if many:
                result = collection.delete_many(query)
                deleted_count = result.deleted_count
            else:
                result = collection.delete_one(query)
                deleted_count = result.deleted_count
            
            logger.info(f"Deleted {deleted_count} documents from {collection_name}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete documents from {collection_name}: {e}")
            return None
    
    def create_index(self, collection_name, index_fields, unique=False):
        """
        Create an index on a collection.
        
        Args:
            collection_name: Name of the collection.
            index_fields: List of (field, direction) tuples or a single field name.
            unique: Whether the index should enforce uniqueness.
            
        Returns:
            Name of the created index or None if failed.
        """
        if not self.db:
            logger.error("MongoDB not connected")
            return None
        
        try:
            collection = self.db[collection_name]
            
            # Convert single field to list of tuples
            if isinstance(index_fields, str):
                index_fields = [(index_fields, ASCENDING)]
            elif isinstance(index_fields, list) and all(isinstance(f, str) for f in index_fields):
                index_fields = [(f, ASCENDING) for f in index_fields]
            
            # Create index
            index_name = collection.create_index(index_fields, unique=unique)
            logger.info(f"Created index {index_name} on {collection_name}")
            return index_name
        except Exception as e:
            logger.error(f"Failed to create index on {collection_name}: {e}")
            return None
    
    def store_dataframe(self, collection_name, df, upsert_fields=None):
        """
        Store a pandas DataFrame in MongoDB.
        
        Args:
            collection_name: Name of the collection.
            df: Pandas DataFrame.
            upsert_fields: List of fields to use for upserting (updating existing records).
            
        Returns:
            Number of documents inserted/updated or None if failed.
        """
        if not self.db:
            logger.error("MongoDB not connected")
            return None
        
        try:
            # Convert DataFrame to list of dictionaries
            records = json.loads(df.to_json(orient='records', date_format='iso'))
            
            if upsert_fields and records:
                # Perform upsert for each record
                inserted_count = 0
                updated_count = 0
                
                for record in records:
                    # Create query from upsert fields
                    query = {field: record[field] for field in upsert_fields if field in record}
                    
                    # Skip if query is empty
                    if not query:
                        continue
                    
                    # Upsert record
                    result = self.db[collection_name].update_one(
                        query,
                        {'$set': record},
                        upsert=True
                    )
                    
                    if result.upserted_id:
                        inserted_count += 1
                    elif result.modified_count > 0:
                        updated_count += 1
                
                logger.info(f"Upserted {inserted_count + updated_count} documents "
                           f"({inserted_count} inserted, {updated_count} updated) "
                           f"into {collection_name}")
                return inserted_count + updated_count
            else:
                # Simple insert
                result = self.db[collection_name].insert_many(records)
                logger.info(f"Inserted {len(result.inserted_ids)} documents into {collection_name}")
                return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Failed to store DataFrame in {collection_name}: {e}")
            return None
    
    def load_dataframe(self, collection_name, query=None, projection=None, sort=None):
        """
        Load data from MongoDB into a pandas DataFrame.
        
        Args:
            collection_name: Name of the collection.
            query: Query filter.
            projection: Fields to include or exclude.
            sort: Sort criteria.
            
        Returns:
            Pandas DataFrame or empty DataFrame if no data found.
        """
        if not self.db:
            logger.error("MongoDB not connected")
            return pd.DataFrame()
        
        try:
            # Find documents
            documents = self.find_documents(collection_name, query, projection, sort)
            
            if not documents:
                logger.warning(f"No documents found in {collection_name}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(documents)
            
            # Remove MongoDB _id column
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            logger.info(f"Loaded {len(df)} rows from {collection_name} into DataFrame")
            return df
        except Exception as e:
            logger.error(f"Failed to load DataFrame from {collection_name}: {e}")
            return pd.DataFrame()


def main():
    """
    Main function to test the MongoDB handler.
    """
    # Initialize MongoDB handler
    handler = MongoDBHandler()
    
    # Check if connected
    if not handler.db:
        logger.error("Failed to connect to MongoDB")
        return
    
    # Create some test data
    test_data = [
        {"ticker": "AAPL", "price": 150.25, "timestamp": "2023-07-01T12:00:00"},
        {"ticker": "MSFT", "price": 310.75, "timestamp": "2023-07-01T12:00:00"},
        {"ticker": "GOOG", "price": 125.30, "timestamp": "2023-07-01T12:00:00"}
    ]
    
    # Insert test data
    handler.insert_documents("test_collection", test_data)
    
    # Find documents
    documents = handler.find_documents("test_collection", {"ticker": "AAPL"})
    logger.info(f"Found documents: {documents}")
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    # Store DataFrame
    handler.store_dataframe("test_dataframe", df)
    
    # Load DataFrame
    loaded_df = handler.load_dataframe("test_dataframe")
    logger.info(f"Loaded DataFrame: {loaded_df}")
    
    # Close connection
    handler.close()


if __name__ == "__main__":
    main() 