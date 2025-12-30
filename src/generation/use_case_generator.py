"""
Use case and test case generation using LLM
Generates structured JSON output grounded in retrieved context
"""

import json
import re
from typing import List, Dict, Any, Optional
import anthropic
from openai import AzureOpenAI

from src.ingestion.chunking_strategy import Chunk
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class UseCaseGenerator:
    """Generate use cases and test cases from retrieved context"""
    
    def __init__(self, config):
        """Initialize generator"""
        self.config = config
        
        # Initialize LLM client
        if config.llm_provider == 'anthropic':
            if not config.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            self.generate_func = self._generate_anthropic
        elif config.llm_provider == 'openai':
            # Use Azure OpenAI if endpoint is configured
            if hasattr(config, 'azure_openai_endpoint') and config.azure_openai_endpoint:
                if not config.azure_openai_api_key:
                    raise ValueError("AZURE_OPENAI_API_KEY not set")
                self.client = AzureOpenAI(
                    api_key=config.azure_openai_api_key,
                    api_version="2025-01-01-preview",
                    azure_endpoint="https://intern-testing-resource.cognitiveservices.azure.com"
                )
                logger.info(f"Using Azure OpenAI with deployment: {config.azure_deployment_name}")
            else:
                if not config.openai_api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                from openai import OpenAI
                self.client = OpenAI(api_key=config.openai_api_key)
            self.generate_func = self._generate_openai
        else:
            raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
        
        logger.info(f"Initialized generator with {config.llm_provider}/{config.llm_model}")
    
    def generate_use_cases(self, query: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Generate use cases based on query and retrieved context
        
        Args:
            query: User query
            chunks: Retrieved context chunks
            
        Returns:
            Dictionary containing use cases, assumptions, and missing info
        """
        logger.info(f"Generating use cases for: {query}")
        
        # Prepare context
        context = self._prepare_context(chunks)
        
        # Generate prompt
        prompt = self._create_prompt(query, context)
        
        # Generate response
        try:
            response_text = self.generate_func(prompt)
            
            # Parse response
            result = self._parse_response(response_text)
            
            logger.info(f"Generated {len(result.get('use_cases', []))} use cases")
            
            return result
        
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
    
    def _prepare_context(self, chunks: List[Chunk]) -> str:
        """Prepare context from chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get('filename', 'unknown')
            score = chunk.metadata.get('score', 0)
            
            context_part = f"""
[CONTEXT {i}] (Source: {source}, Relevance: {score:.3f})
{chunk.content}
"""
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create generation prompt"""
        return f"""You are an expert QA engineer creating comprehensive test cases and use cases.

Your task is to generate test cases based on the user's query and the provided context.

CRITICAL RULES:
1. ONLY use information from the provided context - do not invent features or behaviors
2. If information is insufficient, explicitly state assumptions and missing information
3. Generate structured test cases in VALID JSON format
4. Include positive, negative, and boundary test cases where applicable
5. Be specific and actionable

CONTEXT:
{context}

USER QUERY: {query}

Generate a comprehensive set of use cases/test cases. Your response must be in this EXACT JSON format:

{{
  "use_cases": [
    {{
      "id": "UC001",
      "title": "Clear, descriptive title",
      "goal": "What this test case verifies",
      "preconditions": ["List of preconditions"],
      "test_data": {{
        "field1": "value1",
        "field2": "value2"
      }},
      "steps": [
        "Step 1: Action to take",
        "Step 2: Next action"
      ],
      "expected_results": [
        "Expected outcome 1",
        "Expected outcome 2"
      ],
      "test_type": "positive|negative|boundary",
      "priority": "high|medium|low",
      "tags": ["tag1", "tag2"]
    }}
  ],
  "assumptions": [
    "List any assumptions made due to missing information"
  ],
  "missing_info": [
    "List information that would improve these test cases"
  ]
}}

Generate at least 3-5 test cases covering different scenarios (positive, negative, boundary cases).
Ensure your response is VALID JSON - no markdown formatting, no extra text outside the JSON.
"""
    
    def _generate_anthropic(self, prompt: str) -> str:
        """Generate using Anthropic Claude"""
        response = self.client.messages.create(
            model=self.config.llm_model,
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI or Azure OpenAI"""
        # For Azure OpenAI, use the deployment name directly as the model
        # For standard OpenAI, use the configured model
        if hasattr(self.config, 'azure_openai_endpoint') and self.config.azure_openai_endpoint:
            # Azure OpenAI - use deployment name as model parameter
            model_name = "intern-gpt-4o-mini"
        else:
            model_name = self.config.llm_model
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert QA engineer creating test cases."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        
        return response.choices[0].message.content
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        # Remove markdown code blocks if present
        response_text = response_text.strip()
        
        # Remove markdown json formatting
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        try:
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate structure
            if 'use_cases' not in result:
                logger.warning("Response missing 'use_cases' key")
                result['use_cases'] = []
            
            if not isinstance(result['use_cases'], list):
                logger.warning("'use_cases' is not a list")
                result['use_cases'] = []
            
            # Ensure assumptions and missing_info exist
            if 'assumptions' not in result:
                result['assumptions'] = []
            if 'missing_info' not in result:
                result['missing_info'] = []
            
            # Validate each use case structure
            validated_use_cases = []
            for uc in result['use_cases']:
                if self._validate_use_case(uc):
                    validated_use_cases.append(uc)
                else:
                    logger.warning(f"Invalid use case structure: {uc.get('id', 'unknown')}")
            
            result['use_cases'] = validated_use_cases
            
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result
                except:
                    pass
            
            # Return empty structure
            return {
                'use_cases': [],
                'assumptions': ['Failed to parse LLM response'],
                'missing_info': ['Could not generate structured output'],
                'error': str(e)
            }
    
    def _validate_use_case(self, use_case: Dict) -> bool:
        """Validate use case structure"""
        required_fields = ['title', 'steps', 'expected_results']
        
        for field in required_fields:
            if field not in use_case:
                return False
        
        # Ensure lists are actually lists
        if not isinstance(use_case.get('steps', []), list):
            return False
        if not isinstance(use_case.get('expected_results', []), list):
            return False
        
        return True
