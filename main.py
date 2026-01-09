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
    """Create user record from conversation using OpenAI"""
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

If any information is not found, use empty string "" for that field.

Conversation:
{conversation}

JSON:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. Extract user information from conversations and return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        user_data = json.loads(response.choices[0].message.content)
        
        extracted_email = email or user_data.get("email", "")
        if extracted_email:
            extracted_email = extracted_email.lower().strip()
        
        user_doc = {
            "name": user_data.get("name", ""),
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
        
        identifier = f"phone: {phone_db_format}" if phone_number else f"email: {extracted_email}"
        logger.info(f"Created new user: {user_doc['name']} ({identifier}, client: {client_id})")
        
        return user_doc
        
    except Exception as e:
        logger.error(f"Error creating user from conversation for client {client_id}: {e}")
        return None


def generate_analytics(client_id: str, conversation: str, user_id: ObjectId) -> Optional[dict]:
    """Generate analytics from conversation using OpenAI"""
    try:
        prompt = f"""Analyze the following conversation and extract analytics information about the user.
Extract the following information:
1. course_interest: The engineering branch or course they're interested in
2. city: The city they're from or located in
3. budget: Their budget range for education
4. hostel_needed: Boolean value (true or false) - whether they need hostel accommodation
5. intent_level: One of "TOFU" (Top of Funnel - early interest), "MOFU" (Middle of Funnel - considering), or "BOFU" (Bottom of Funnel - ready to enroll)

Return the information in JSON format:
{{
    "course_interest": "course name in lowercase",
    "city": "city name",
    "budget": "budget range as mentioned",
    "hostel_needed": true or false,
    "intent_level": "TOFU" or "MOFU" or "BOFU"
}}

Conversation:
{conversation}

JSON:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an analytics extraction assistant. Extract user analytics from conversations and return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        analytics_data = json.loads(response.choices[0].message.content)
        
        hostel_needed = analytics_data.get("hostel_needed", False)
        if isinstance(hostel_needed, str):
            hostel_needed = hostel_needed.lower() in ["true", "yes", "1", "needed", "required"]
        
        course_interest = analytics_data.get("course_interest") or ""
        course_interest = course_interest.lower() if isinstance(course_interest, str) else ""
        
        city = analytics_data.get("city") or ""
        
        intent_level = analytics_data.get("intent_level") or "TOFU"
        intent_level = intent_level.upper() if isinstance(intent_level, str) else "TOFU"
        
        analytics_doc = {
            "user_id": user_id,
            "course_interest": course_interest,
            "city": city,
            "budget": analytics_data.get("budget", ""),
            "hostel_needed": bool(hostel_needed),
            "intent_level": intent_level
        }
        
        return analytics_doc
        
    except Exception as e:
        logger.error(f"Error generating analytics for client {client_id}: {e}")
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
        analytics_collection = db["userAnalytics"]
        
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
                # User doesn't exist - create new user
                logger.info("User not found, creating new user...")
                user = create_user_from_conversation(client_id, conversation, phone_number=phone_number, email=email)
                
                if not user:
                    logger.warning("Failed to create user, conversation will be stored without user association")
                else:
                    logger.info(f"User created: {user.get('name', 'Unknown')}")
            else:
                logger.info(f"User found: {user.get('name', 'Unknown')}")
            
            if user:
                user_id = user.get("_id")
        else:
            logger.info("No user identification found - storing conversation as anonymous")
        
        # Phase 2: Generate/update analytics (only if user exists)
        analytics_doc = None
        if user_id:
            logger.info("Phase 2: Generating analytics...")
            analytics_doc = generate_analytics(client_id, conversation, user_id)
            
            if not analytics_doc:
                logger.warning("Failed to generate analytics, skipping analytics storage")
            else:
                # Check if analytics already exists for this user
                existing_analytics = analytics_collection.find_one({"user_id": ObjectId(user_id)})
                
                if existing_analytics:
                    logger.info("Updating existing analytics...")
                    analytics_collection.update_one(
                        {"user_id": ObjectId(user_id)},
                        {"$set": analytics_doc}
                    )
                    analytics_doc["_id"] = existing_analytics.get("_id")
                    logger.info("Analytics updated successfully")
                else:
                    logger.info("Creating new analytics...")
                    result = analytics_collection.insert_one(analytics_doc)
                    analytics_doc["_id"] = result.inserted_id
                    logger.info("Analytics created successfully")
        else:
            logger.info("Phase 2: Skipping analytics (no user identified)")
        
        # Phase 3: Store conversation history (always, even without user)
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
        logger.info(f"Conversation history created successfully with ID: {result.inserted_id}")
        
        # Prepare response
        response = {
            "status": "success",
            "message": "Conversation processed successfully",
            "client_id": client_id,
            "user": {
                "_id": str(user.get("_id")) if user else None,
                "name": user.get("name") if user else None,
                "email": user.get("email") if user else None,
                "phone_number": user.get("phone_number") if user else None
            } if user else None,
            "analytics": {
                "_id": str(analytics_doc.get("_id")) if analytics_doc else None,
                "user_id": str(analytics_doc.get("user_id")) if analytics_doc else None,
                "course_interest": analytics_doc.get("course_interest") if analytics_doc else None,
                "city": analytics_doc.get("city") if analytics_doc else None,
                "budget": analytics_doc.get("budget") if analytics_doc else None,
                "hostel_needed": analytics_doc.get("hostel_needed") if analytics_doc else None,
                "intent_level": analytics_doc.get("intent_level") if analytics_doc else None
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

