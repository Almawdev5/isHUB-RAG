from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import PortfolioItem
from .serializers import QuerySerializer, PortfolioItemSerializer, UploadPDFSerializer, AddWebContentSerializer, AddExistingPDFSerializer
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import PyPDF2
import logging
import os
import base64
from django.core.files.base import ContentFile

# Configure loggerlogger = logging.getLogger(name)

class QueryView(APIView):
    """Handles user queries by retrieving relevant portfolio items and generating responses via the Groq API."""

    def post(self, request):
        logger.info("Processing query request")
        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"Invalid query data: {serializer.errors}")
            return Response(
                {"error": "Invalid query data", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        query = serializer.validated_data['query']
        logger.debug(f"Received query: {query}")

        # Generate query embedding
        try:
            logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(query).tolist()
            logger.debug("Query embedding generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            return Response(
                {"error": "Failed to generate query embedding", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Query ChromaDB
        try:
            logger.info(f"Connecting to ChromaDB at {settings.CHROMA_DB_PATH}")
            client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
            collection = client.get_or_create_collection("portfolio")
            results = collection.query(query_embeddings=[query_embedding], n_results=5)
            vector_ids = results['ids'][0]
            logger.info(f"Retrieved vector IDs: {vector_ids}")
            items = PortfolioItem.objects.filter(vector_id__in=vector_ids)
            context = [item.content for item in items]
            logger.debug(f"Context retrieved: {context[:100]}...")
        except Exception as e:
            logger.error(f"ChromaDB query failed: {str(e)}", exc_info=True)
            return Response(
                {"error": "Failed to retrieve portfolio items", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Query Groq API
        try:
            logger.info(f"Sending request to Groq API with key: {settings.GROQ_API_KEY[:4]}...{settings.GROQ_API_KEY[-4:]}")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a portfolio assistant. Provide accurate responses based solely on the provided portfolio context."
                        },
                        {"role": "user", "content": f"Query: {query}\nContext: {context}"}
                    ],
                    "max_tokens": 500
                },
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                timeout=10
            )
            llm_response = response.json()
            logger.debug(f"Groq API response: {llm_response}")
            if response.status_code != 200:
                logger.error(f"Groq API error: {llm_response.get('error', 'Unknown error')} (Status: {response.status_code})")
                return Response(
                    {"error": "Groq API request failed", "details": llm_response.get('error', 'Unknown error')},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            if 'choices' not in llm_response or not llm_response['choices']:
                logger.error(f"Invalid Groq API response: {llm_response}")
                return Response(
                    {"error": "Invalid response from Groq API", "details": str(llm_response)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            response_text = llm_response['choices'][0]['message']['content']
            logger.info("Groq API response received successfully")
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {str(e)}", exc_info=True)
            return Response(
                {"error": "Groq API communication error", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        item_serializer = PortfolioItemSerializer(items, many=True)
        logger.info("Query processed successfully")
        return Response(
            {
                "response": response_text,
                "items": item_serializer.data
            },
            status=status.HTTP_200_OK
        )

class UploadPDFView(APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def post(self, request):
        logger.info("Processing PDF upload request")
        logger.info(f"Incoming Content-Type: {request.content_type or 'None'}")

        if request.content_type == 'application/json':
            logger.info("Processing JSON-based PDF upload")
            file_data = request.data.get('file')
            title = request.data.get('title', '')
            metadata = request.data.get('metadata', {})

            if not file_data:
                logger.error("File data is missing in JSON request")
                return Response(
                    {"error": "File data is required in JSON request"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Validate and decode base64 file data
            try:
                if not file_data.startswith("data:") or ";base64," not in file_data:
                    logger.error(f"Invalid file data received: {file_data[:100]}...")
                    raise ValueError(
                        "Invalid base64 format. Expected 'data:<mime-type>;base64,<data>'."
                    )