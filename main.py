"""
Multi-tenant Postprocessor Server
Handles multiple clients with isolated MongoDB databases.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import (
    get_client_config,
    get_mongodb_database_name,
    get_client_id_from_request
)

import json
from typing import Optional, List
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import ObjectId
from openai import OpenAI
from loguru import logger
import threading

# Load environment variables
load_dotenv()

# Configure logger
logger.add("postprocessor.log", rotation="10 MB", level="INFO")

# FastAPI app
app = FastAPI(title="Multi-Tenant Postprocessor", version="2.0.0")

# Add CORS middleware (configurable via environment)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Client-ID", "X-Request-ID"],  # Expose custom headers
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Tenant Context Middleware
@app.middleware("http")
async def tenant_context_middleware(request: Request, call_next):
    """
    Middleware to extract and store tenant/client context in request state.
    Logs which tenant is accessing which resource for monitoring and debugging.
    """
    # Extract client_id from query params or headers
    client_id = request.query_params.get("client_id") or request.headers.get("X-Client-ID")
    
    # Store in request state for easy access throughout request lifecycle
    request.state.client_id = client_id
    request.state.has_client_context = bool(client_id)
    
    # Log tenant context (optional - can be disabled in production)
    if client_id:
        logger.debug(f"[Tenant: {client_id}] {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Add client_id to response headers for tracking
    if client_id:
        response.headers["X-Client-ID"] = client_id
    
    return response

# Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI must be set in .env file")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize MongoDB client with connection pooling
# Single client instance shared across all requests and all client databases
try:
    mongodb_client = MongoClient(
        MONGODB_URI,
        maxPoolSize=50,  # Maximum 50 connections in pool
        minPoolSize=10,  # Minimum 10 connections always ready
        maxIdleTimeMS=45000,  # Close idle connections after 45s
        serverSelectionTimeoutMS=5000,  # Timeout for server selection
        connectTimeoutMS=10000,  # Timeout for initial connection
        socketTimeoutMS=45000,  # Timeout for socket operations
        retryWrites=True,  # Retry writes on network errors
        retryReads=True,  # Retry reads on network errors
    )
    # Test connection
    mongodb_client.admin.command('ping')
    logger.info("✅ MongoDB connection pool initialized successfully")
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    logger.error(f"❌ Failed to connect to MongoDB: {e}")
    raise ValueError(f"MongoDB connection failed: {str(e)}")
except Exception as e:
    logger.error(f"❌ Error connecting to MongoDB: {e}")
    raise ValueError(f"MongoDB error: {str(e)}")


def normalize_phone_number(phone_number: str) -> str:
    """Normalize phone number to 10 digits"""
    digits_only = ''.join(filter(str.isdigit, phone_number))
    
    if digits_only.startswith("91") and len(digits_only) > 10:
        phone_number_clean = digits_only[2:]
    elif len(digits_only) > 10:
        phone_number_clean = digits_only[-10:]
    elif len(digits_only) < 10:
        phone_number_clean = digits_only.zfill(10)
    else:
        phone_number_clean = digits_only
    
    return phone_number_clean


def normalize_conversation_tags(conversation: str) -> str:
    """Normalize conversation tags to strict 'User:' and 'Agent:' format"""
    import re
    
    lines = conversation.split('\n')
    normalized_lines = []
    
    for line in lines:
        agent_pattern = r'^(?:Natalie\s*\(Agent\)|Agent\s*\([^)]*\)|Natalie|Agent)\s*:\s*(.*)$'
        user_pattern = r'^User\s*:\s*(.*)$'
        
        agent_match = re.match(agent_pattern, line, re.IGNORECASE)
        if agent_match:
            normalized_lines.append(f"Agent: {agent_match.group(1)}")
        else:
            user_match = re.match(user_pattern, line, re.IGNORECASE)
            if user_match:
                normalized_lines.append(f"User: {user_match.group(1)}")
            else:
                normalized_lines.append(line)
    
    return '\n'.join(normalized_lines)


def detect_languages(conversation: str) -> List[str]:
    """Detect languages used in the conversation using OpenAI"""
    try:
        prompt = f"""Analyze the following conversation and identify ALL languages used.
The conversation may contain multiple languages mixed together.
Return the result as a JSON object with a "languages" field containing an array:
{{"languages": ["english", "tamil"]}}

Conversation:
{conversation}

JSON:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert language detection assistant. Return a JSON object with a 'languages' array containing all detected languages in lowercase."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        languages = result.get("languages", [])
        languages = [lang.lower().strip() for lang in languages if lang and lang.strip()]
        
        if not languages:
            languages = ["english"]
        
        languages = sorted(list(set(languages)))
        logger.info(f"Detected languages: {languages}")
        return languages
        
    except Exception as e:
        logger.error(f"Error detecting languages: {e}")
        return ["english"]


def extract_phone_number(conversation: str) -> Optional[str]:
    """Extract phone number from conversation using OpenAI"""
    try:
        prompt = f"""Analyze the following conversation and extract the phone number mentioned in it.
Return ONLY the phone number in digits (no spaces, no dashes, no plus signs).
If no phone number is found, return "NOT_FOUND".

Conversation:
{conversation}

Phone number:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a phone number extraction assistant. Extract phone numbers from conversations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=20
        )
        
        phone_number = response.choices[0].message.content.strip()
        
        if phone_number == "NOT_FOUND" or not phone_number:
            logger.warning("No phone number found in conversation")
            return None
        
        normalized = normalize_phone_number(phone_number)
        
        if len(normalized) != 10:
            logger.warning(f"Invalid phone number format extracted: {phone_number}")
            return None
        
        return normalized
        
    except Exception as e:
        logger.error(f"Error extracting phone number: {e}")
        return None


def extract_email(conversation: str) -> Optional[str]:
    """Extract email address from conversation using OpenAI"""
    try:
        prompt = f"""Analyze the following conversation and extract the email address mentioned in it.
Return ONLY the email address in lowercase.
If no email address is found, return "NOT_FOUND".

Conversation:
{conversation}

Email address:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an email extraction assistant. Extract email addresses from conversations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        email = response.choices[0].message.content.strip()
        
        if email == "NOT_FOUND" or not email:
            logger.warning("No email address found in conversation")
            return None
        
        email = email.lower().strip()
        
        if "@" not in email or "." not in email.split("@")[1]:
            logger.warning(f"Invalid email format extracted: {email}")
            return None
        
        return email
        
    except Exception as e:
        logger.error(f"Error extracting email: {e}")
        return None


def create_user_from_conversation(client_id: str, conversation: str, phone_number: Optional[str] = None, email: Optional[str] = None) -> Optional[dict]:
    """Create user record from conversation using OpenAI. Name and email are optional (nullable)."""
    try:
        prompt = f"""Analyze the following conversation and extract the user's information.
Extract the following information:
1. Name: The person's full name
2. Email: The person's email address

Return the information in JSON format:
{{
    "name": "Full Name",
    "email": "email@example.com"
}}

If any information is not found, use null for that field.

Conversation:
{conversation}

JSON:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. Extract user information from conversations and return valid JSON. Use null for missing fields."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        user_data = json.loads(response.choices[0].message.content)
        
        # Extract name - can be null
        extracted_name = user_data.get("name")
        if extracted_name and isinstance(extracted_name, str):
            extracted_name = extracted_name.strip() or None
        else:
            extracted_name = None
        
        # Extract email - can be null
        extracted_email = email or user_data.get("email")
        if extracted_email and isinstance(extracted_email, str):
            extracted_email = extracted_email.lower().strip() or None
        else:
            extracted_email = None
        
        # Build user document with nullable fields
        user_doc = {
            "name": extracted_name,
            "email": extracted_email
        }
        
        if phone_number:
            phone_db_format = f"91{phone_number}"
            user_doc["phone_number"] = phone_db_format
        
        # Get database for this client (using shared MongoDB client pool)
        db_name = get_mongodb_database_name(client_id)
        db = mongodb_client[db_name]
        users_collection = db["users"]
        
        # Check if user already exists
        if phone_number:
            phone_db_format = f"91{phone_number}"
            existing = users_collection.find_one({"phone_number": phone_db_format})
            if existing:
                logger.info(f"User already exists with phone: {phone_db_format} (client: {client_id})")
                return existing
        
        if extracted_email:
            existing = users_collection.find_one({"email": extracted_email})
            if existing:
                logger.info(f"User already exists with email: {extracted_email} (client: {client_id})")
                return existing
        
        # Insert new user
        result = users_collection.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        
        identifier_parts = []
        if extracted_name:
            identifier_parts.append(f"name: {extracted_name}")
        if phone_number:
            identifier_parts.append(f"phone: {phone_db_format}")
        if extracted_email:
            identifier_parts.append(f"email: {extracted_email}")
        identifier = ", ".join(identifier_parts) if identifier_parts else "anonymous"
        
        logger.info(f"Created new user: {identifier} (client: {client_id})")
        
        return user_doc
        
    except Exception as e:
        logger.error(f"Error creating user from conversation for client {client_id}: {e}")
        return None


def generate_analytics(client_id: str, conversation: str, conversation_history_id: str) -> Optional[dict]:
    """Generate basic call analytics from conversation using OpenAI"""
    try:
        # Calculate conversation duration (approximate based on word count)
        words = conversation.split()
        estimated_duration = len(words) / 150  # Average speaking rate ~150 wpm
        
        prompt = f"""Analyze the following conversation between a user and an agent. Extract comprehensive call analytics.

Conversation:
{conversation}

Provide detailed analytics in JSON format with these EXACT fields:
{{
    "call_quality_score": <0-100 integer, overall quality of the conversation>,
    "engagement_score": <0-100 integer, how engaged both parties were>,
    "caller_talk_percentage": <0-100 integer, percentage of conversation by caller/user>,
    "client_talk_percentage": <0-100 integer, percentage of conversation by agent/client>,
    "client_sentiment_index": <-1.0 to 1.0 float, agent's sentiment: -1=negative, 0=neutral, 1=positive>,
    "agent_sentiment_index": <-1.0 to 1.0 float, user's sentiment: -1=negative, 0=neutral, 1=positive>,
    "question_rate_per_minute": <float, number of questions asked per minute>,
    "clarification_score": <0-100 integer, how well questions were clarified>,
    "objection_handling_effectiveness": <0-100 integer, if applicable, else 100>,
    "clarity_index_wpm": <integer, estimated words per minute - aim for 130-160>,
    "filler_word_percentage": <0-100 float, percentage of filler words like um, uh, like>,
    "interruptions_count": <integer, number of interruptions>,
    "sentiment_bias": <-1.0 to 1.0 float, difference between agent and client sentiment>,
    "interest_moments": [<array of strings, key moments where user showed high interest>],
    "next_step_compliance": <boolean, were next steps clearly defined and agreed upon>,
    "topics_discussed": [
        {{
            "heading": "<topic heading>",
            "content": "<brief description of what was discussed>"
        }}
    ],
    "action_items": [<array of strings, concrete action items from the conversation>],
    "call_summary": "<2-3 sentence summary of the entire conversation>"
}}

Return ONLY valid JSON, no markdown or extra text."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a call analytics expert. Analyze conversations and extract detailed metrics. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        analytics_data = json.loads(response.choices[0].message.content)
        
        # Build the analytics document with the new schema
        analytics_doc = {
            "conversationHistory_id": conversation_history_id,
            "analysis_timestamp": datetime.now(),
            "metrics": {
                "call_quality_score": int(analytics_data.get("call_quality_score", 50)),
                "engagement_score": int(analytics_data.get("engagement_score", 50)),
                "caller_talk_percentage": int(analytics_data.get("caller_talk_percentage", 50)),
                "client_talk_percentage": int(analytics_data.get("client_talk_percentage", 50)),
                "client_sentiment_index": float(analytics_data.get("client_sentiment_index", 0.0)),
                "agent_sentiment_index": float(analytics_data.get("agent_sentiment_index", 0.0)),
                "question_rate_per_minute": float(analytics_data.get("question_rate_per_minute", 0.0)),
                "clarification_score": int(analytics_data.get("clarification_score", 50)),
                "objection_handling_effectiveness": int(analytics_data.get("objection_handling_effectiveness", 100)),
                "clarity_index_wpm": int(analytics_data.get("clarity_index_wpm", 140)),
                "filler_word_percentage": float(analytics_data.get("filler_word_percentage", 0.0)),
                "interruptions_count": int(analytics_data.get("interruptions_count", 0)),
                "sentiment_bias": float(analytics_data.get("sentiment_bias", 0.0)),
                "interest_moments": analytics_data.get("interest_moments", []),
                "next_step_compliance": bool(analytics_data.get("next_step_compliance", False)),
                "call_duration_minutes": round(estimated_duration, 1),
                "topics_discussed": analytics_data.get("topics_discussed", []),
                "action_items": analytics_data.get("action_items", []),
                "call_summary": analytics_data.get("call_summary", "")
            }
        }

        logger.info("Analytics generated successfully")
        return analytics_doc

    except Exception as e:
        logger.error(f"Error generating analytics for client {client_id}: {e}", exc_info=True)
        return None


# Pydantic models
class ConversationRequest(BaseModel):
    conversation: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "multi-tenant-postprocessor"}


@app.post("/process")
async def process_conversation(request: Request, conversation_request: ConversationRequest):
    """
    Process conversation data for a specific client:
    Phase 1: Extract phone number, check/create user
    Phase 2: Generate/update analytics
    Phase 3: Store conversation history
    """
    try:
        client_id = get_client_id_from_request(request)
        conversation = conversation_request.conversation
        
        if not conversation or not conversation.strip():
            raise HTTPException(status_code=400, detail="Conversation data is required")
        
        logger.info(f"Starting conversation processing for client: {client_id}...")
        
        # Phase 1: Extract phone number or email, detect languages
        logger.info("Phase 1: Extracting phone number and detecting languages...")
        phone_number = extract_phone_number(conversation)
        email = None
        
        if not phone_number:
            logger.info("No phone number found, attempting to extract email...")
            email = extract_email(conversation)
            
            if email:
                logger.info(f"Extracted email: {email}")
            else:
                logger.info("No phone number or email found - conversation will be stored without user association")
        else:
            logger.info(f"Extracted phone number: {phone_number}")
        
        languages_used = detect_languages(conversation)
        logger.info(f"Detected languages: {languages_used}")
        
        # Get database for this client (using shared MongoDB client pool)
        db_name = get_mongodb_database_name(client_id)
        db = mongodb_client[db_name]
        users_collection = db["users"]
        analytics_collection = db["callAnalytics"]  # Changed from userAnalytics to callAnalytics
        
        user = None
        user_id = None
        
        # Only try to find/create user if we have phone or email
        if phone_number or email:
            # Try to find user by phone number first
            if phone_number:
                phone_db_format = f"91{phone_number}"
                user = users_collection.find_one({"phone_number": phone_db_format})
            
            # If not found by phone, try to find by email
            if not user and email:
                email_normalized = email.lower().strip()
                user = users_collection.find_one({"email": email_normalized})
            
            if not user:
                # User doesn't exist - create new user (name and email are now optional)
                logger.info("User not found, creating new user...")
                user = create_user_from_conversation(client_id, conversation, phone_number=phone_number, email=email)
                
                if not user:
                    logger.warning("Failed to create user, conversation will be stored without user association")
                else:
                    logger.info(f"User created: {user.get('name', 'Anonymous')}")
            else:
                logger.info(f"User found: {user.get('name', 'Anonymous')}")
            
            if user:
                user_id = user.get("_id")
        else:
            logger.info("No user identification found - storing conversation as anonymous")
        
        # Phase 3: Store conversation history FIRST (always, even without user)
        logger.info("Phase 3: Storing conversation history...")
        conversation_history_collection = db["conversationHistory"]
        
        normalized_conversation = normalize_conversation_tags(conversation)
        logger.info("Normalized conversation tags to strict 'User:' and 'Agent:' format")
        
        conversation_history_doc = {
            "user_id": ObjectId(user_id) if user_id else None,  # Store null if no user
            "conversation": normalized_conversation,
            "timestamp": datetime.now(),
            "languages_used": languages_used,
            "anonymous": not bool(user_id)  # Flag to indicate anonymous conversation
        }
        
        logger.info(f"Creating new conversation history document (anonymous={not bool(user_id)})...")
        result = conversation_history_collection.insert_one(conversation_history_doc)
        conversation_history_doc["_id"] = result.inserted_id
        conversation_history_id = str(result.inserted_id)
        logger.info(f"Conversation history created successfully with ID: {conversation_history_id}")
        
        # Phase 2: Generate analytics (always, for every conversation)
        analytics_doc = None
        logger.info("Phase 2: Generating call analytics...")
        analytics_doc = generate_analytics(client_id, conversation, conversation_history_id)
        
        if not analytics_doc:
            logger.warning("Failed to generate analytics, skipping analytics storage")
        else:
            logger.info("Creating new call analytics...")
            result = analytics_collection.insert_one(analytics_doc)
            analytics_doc["_id"] = result.inserted_id
            logger.info(f"Call analytics created successfully with ID: {result.inserted_id}")
        
        # Prepare response
        response = {
            "status": "success",
            "message": "Conversation processed successfully",
            "client_id": client_id,
            "conversation_history_id": conversation_history_id,
            "user": {
                "_id": str(user.get("_id")) if user else None,
                "name": user.get("name") if user else None,
                "email": user.get("email") if user else None,
                "phone_number": user.get("phone_number") if user else None
            } if user else None,
            "analytics": {
                "_id": str(analytics_doc.get("_id")) if analytics_doc else None,
                "conversationHistory_id": analytics_doc.get("conversationHistory_id") if analytics_doc else None,
                "analysis_timestamp": str(analytics_doc.get("analysis_timestamp")) if analytics_doc else None,
                "metrics": analytics_doc.get("metrics") if analytics_doc else None
            } if analytics_doc else None,
            "conversation_history": {
                "_id": str(conversation_history_doc.get("_id")),
                "user_id": str(conversation_history_doc.get("user_id")) if conversation_history_doc.get("user_id") else None,
                "conversation": conversation_history_doc.get("conversation"),
                "anonymous": conversation_history_doc.get("anonymous", False)
            }
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import threading
    port = int(os.getenv("PORT", "8003"))
    uvicorn.run(app, host="0.0.0.0", port=port)

