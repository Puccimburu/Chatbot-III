"""
Database connection and management
MongoDB connection with connection pooling and health monitoring
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import motor.motor_asyncio
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo import monitoring

from config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseEventLogger(monitoring.CommandListener):
    """MongoDB command event listener for monitoring"""
    
    def started(self, event):
        if settings.ENABLE_DETAILED_LOGGING:
            logger.debug(f"MongoDB command started: {event.command_name}")
    
    def succeeded(self, event):
        if settings.ENABLE_DETAILED_LOGGING:
            logger.debug(
                f"MongoDB command {event.command_name} succeeded "
                f"in {event.duration_micros/1000:.2f}ms"
            )
    
    def failed(self, event):
        logger.error(
            f"MongoDB command {event.command_name} failed: {event.failure}"
        )


class DatabaseManager:
    """MongoDB database manager with connection pooling and health monitoring"""
    
    def __init__(self):
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.database: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None
        self.is_connected: bool = False
        self._connection_attempts: int = 0
        self._last_health_check: Optional[datetime] = None
        
    async def connect(self) -> bool:
        """Establish database connection with retry logic"""
        
        try:
            logger.info(f"🔗 Connecting to MongoDB: {settings.DATABASE_NAME}")
            
            # Add event monitoring
            if settings.ENABLE_DETAILED_LOGGING:
                monitoring.register(DatabaseEventLogger())
            
            # Create MongoDB client with connection pooling
            self.client = motor.motor_asyncio.AsyncIOMotorClient(
                settings.MONGODB_URL,
                maxPoolSize=settings.MAX_POOL_SIZE,
                minPoolSize=settings.MIN_POOL_SIZE,
                serverSelectionTimeoutMS=settings.CONNECTION_TIMEOUT * 1000,
                connectTimeoutMS=settings.CONNECTION_TIMEOUT * 1000,
                socketTimeoutMS=settings.CONNECTION_TIMEOUT * 1000,
                retryWrites=True,
                retryReads=True
            )
            
            # Get database reference
            self.database = self.client[settings.DATABASE_NAME]
            
            # Test connection
            await self._test_connection()
            
            self.is_connected = True
            self._connection_attempts = 0
            self._last_health_check = datetime.utcnow()
            
            logger.info("✅ MongoDB connection established successfully")
            
            # Log connection info
            await self._log_connection_info()
            
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self._connection_attempts += 1
            logger.error(f"❌ MongoDB connection failed (attempt {self._connection_attempts}): {str(e)}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Unexpected error connecting to MongoDB: {str(e)}")
            return False
    
    async def _test_connection(self):
        """Test database connection"""
        try:
            # Ping the database
            await self.client.admin.command('ping')
            
            # Test basic database operations
            await self.database.command('dbStats')
            
        except Exception as e:
            raise ConnectionFailure(f"Database connection test failed: {str(e)}")
    
    async def _log_connection_info(self):
        """Log database connection information"""
        try:
            # Get server info
            server_info = await self.client.server_info()
            
            # Get database stats
            db_stats = await self.database.command('dbStats')
            
            # Get collection count
            collections = await self.database.list_collection_names()
            
            logger.info(f"📊 Database Info:")
            logger.info(f"   Server Version: {server_info.get('version', 'unknown')}")
            logger.info(f"   Database: {settings.DATABASE_NAME}")
            logger.info(f"   Collections: {len(collections)}")
            logger.info(f"   Data Size: {db_stats.get('dataSize', 0) / (1024*1024):.2f} MB")
            
            if settings.DEBUG:
                logger.debug(f"   Available Collections: {', '.join(collections[:10])}")
                if len(collections) > 10:
                    logger.debug(f"   ... and {len(collections) - 10} more")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not retrieve database info: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            start_time = datetime.utcnow()
            
            # Ping test
            await self.client.admin.command('ping')
            
            # Get server status
            server_status = await self.client.admin.command('serverStatus')
            
            ping_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._last_health_check = datetime.utcnow()
            
            return {
                "status": "healthy",
                "ping_ms": round(ping_time, 2),
                "server_version": server_status.get('version', 'unknown'),
                "uptime_seconds": server_status.get('uptime', 0),
                "connections": server_status.get('connections', {}),
                "last_check": self._last_health_check.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Database health check failed: {str(e)}")
            self.is_connected = False
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    async def get_collections(self) -> List[str]:
        """Get list of all collections in the database"""
        try:
            if not self.is_connected:
                await self.connect()
            
            collections = await self.database.list_collection_names()
            logger.debug(f"📋 Found {len(collections)} collections")
            
            return collections
            
        except Exception as e:
            logger.error(f"❌ Failed to get collections: {str(e)}")
            return []
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a specific collection"""
        try:
            if not self.is_connected:
                await self.connect()
            
            collection = self.database[collection_name]
            
            # Get collection stats
            stats = await self.database.command('collStats', collection_name)
            
            # Get document count (more accurate)
            doc_count = await collection.count_documents({})
            
            # Get index information
            indexes = await collection.list_indexes().to_list(length=None)
            
            return {
                "name": collection_name,
                "document_count": doc_count,
                "size_bytes": stats.get('size', 0),
                "avg_obj_size": stats.get('avgObjSize', 0),
                "storage_size": stats.get('storageSize', 0),
                "indexes": len(indexes),
                "index_names": [idx.get('name') for idx in indexes]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats for collection {collection_name}: {str(e)}")
            return {
                "name": collection_name,
                "error": str(e)
            }
    
    async def sample_documents(self, collection_name: str, sample_size: int = 100) -> List[Dict[str, Any]]:
        """Get sample documents from a collection"""
        try:
            if not self.is_connected:
                await self.connect()
            
            collection = self.database[collection_name]
            
            # Use aggregation pipeline for random sampling
            pipeline = [
                {"$sample": {"size": min(sample_size, settings.SCHEMA_SAMPLE_SIZE)}}
            ]
            
            documents = []
            async for doc in collection.aggregate(pipeline):
                # Convert ObjectId to string for JSON serialization
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                documents.append(doc)
            
            logger.debug(f"📄 Sampled {len(documents)} documents from {collection_name}")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Failed to sample documents from {collection_name}: {str(e)}")
            return []
    
    async def execute_aggregation(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute aggregation pipeline on a collection"""
        try:
            if not self.is_connected:
                await self.connect()
            
            collection = self.database[collection_name]
            
            # Add limit to prevent large result sets
            if not any('$limit' in stage for stage in pipeline):
                pipeline.append({"$limit": settings.MAX_QUERY_RESULTS})
            
            results = []
            async for doc in collection.aggregate(pipeline):
                # Convert ObjectId to string
                if '_id' in doc and hasattr(doc['_id'], '__str__'):
                    doc['_id'] = str(doc['_id'])
                results.append(doc)
            
            logger.debug(f"📊 Aggregation on {collection_name} returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Aggregation failed on {collection_name}: {str(e)}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("🔒 MongoDB connection closed")
    
    def get_collection(self, collection_name: str):
        """Get a collection object"""
        if not self.is_connected or not self.database:
            raise ConnectionError("Database not connected")
        
        return self.database[collection_name]


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_database_manager() -> Optional[DatabaseManager]:
    """Get global database manager instance"""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        
        # Attempt connection with retries
        max_retries = 3
        for attempt in range(max_retries):
            if await _db_manager.connect():
                break
            
            if attempt < max_retries - 1:
                logger.warning(f"⏳ Retrying database connection in 5 seconds... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(5)
        else:
            logger.error("❌ Failed to establish database connection after all retries")
            return None
    
    return _db_manager


async def close_database_connection():
    """Close global database connection"""
    global _db_manager
    
    if _db_manager:
        await _db_manager.close()
        _db_manager = None