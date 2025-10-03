import json
import re
import os
import time
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Google Gemini SDK for LLM integration
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class LegalCaseAnalysis(BaseModel):
    """Structured Pydantic model for legal case analysis"""
    key_principles: List[str] = Field(default_factory=list, description="Key legal principles applicable to this case")
    patterns: List[str] = Field(default_factory=list, description="Common legal patterns identified in similar cases")
    precedents: List[Dict[str, str]] = Field(default_factory=list, description="Relevant legal precedents with case name and outcome")
    recommendations: List[str] = Field(default_factory=list, description="Strategic legal recommendations")
    risk_factors: List[str] = Field(default_factory=list, description="Potential legal risks")
    strong_arguments: List[str] = Field(default_factory=list, description="Strong arguments to present to judges")
    likely_outcome: str = Field(default="", description="Predicted outcome based on precedents")

class LegalAdvisor:
    def __init__(self):
        # Initialize Gemini API
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize the Gemini model
        # Using gemini-1.5-flash for faster responses or gemini-1.5-pro for more complex analysis
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from response text that might contain markdown or extra text"""
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON object boundaries
        if '{' in text and '}' in text:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            return text[start_idx:end_idx]
        return text
        
    def analyze_case(self, case_type: str, opposition_demand: str, additional_details: str) -> Dict:
        """Analyze the case using Gemini LLM without external data"""
        try:
            # Start timer to measure response time
            start_time = time.time()
            
            # Create the prompt for Gemini
            prompt = f"""You are a highly knowledgeable legal AI assistant specialized in Indian law, with extensive experience advising lawyers on legal strategy. You have detailed knowledge of Indian legal precedents, statutes, and court judgments.

CURRENT CASE INFORMATION:
Case Type: {case_type}
Opposition's Demand: {opposition_demand}
Additional Details: {additional_details}

Based on your comprehensive knowledge of Indian law and similar precedents, provide a detailed analysis addressing the following aspects:

1. Identify the key legal principles that apply to this case under Indian law.
2. Analyze common patterns seen in similar cases in Indian courts.
3. List the most relevant legal precedents (including specific case citations if possible) and their outcomes. For each precedent, include:
   - The case name (e.g., "Mohiri Bibi v. Dharmodas Ghose" or other relevant Indian cases)
   - A brief description of the outcome and its relevance to the current case
4. Provide strategic recommendations for handling this case, with specific tactics.
5. Identify potential risk factors and challenges the lawyer should prepare for.
6. Present strong arguments that would be persuasive to Indian judges in this matter.
7. Predict the likely outcome based on precedent analysis and current legal climate in India.

Your response MUST be in valid JSON format that matches this structure exactly:
{{
  "key_principles": ["principle 1", "principle 2", "principle 3"],
  "patterns": ["pattern 1", "pattern 2", "pattern 3"],
  "precedents": [
    {{"case": "Case Name 1", "outcome": "Detailed outcome and relevance"}},
    {{"case": "Case Name 2", "outcome": "Detailed outcome and relevance"}}
  ],
  "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
  "risk_factors": ["risk 1", "risk 2", "risk 3"],
  "strong_arguments": ["argument 1", "argument 2", "argument 3"],
  "likely_outcome": "Detailed prediction of the case outcome"
}}

IMPORTANT: 
- Respond with ONLY the JSON object, no additional text or markdown formatting
- Include at least 3-5 items per field where appropriate
- All precedents must be actual Indian legal cases
- Ensure all JSON keys match exactly as shown above

DO NOT include any explanatory text before or after the JSON."""
            
            # Make the API call to Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                )
            )
            
            # Get text from the response
            raw_response = response.text
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Parse JSON response
            try:
                # Clean the response text for JSON parsing
                clean_text = self._extract_json_from_response(raw_response.strip())
                
                # Parse JSON
                analysis_dict = json.loads(clean_text)
                
                # Create Pydantic model instance
                analysis = LegalCaseAnalysis(**analysis_dict)
                
                # Convert to dictionary for JSON response
                response_data = analysis.dict()
                response_data["response_time"] = f"{response_time:.2f} seconds"
                
                return response_data
            
            except json.JSONDecodeError as json_err:
                # If JSON parsing fails, return a structured error
                return {
                    "error": f"Error parsing JSON response: {str(json_err)}",
                    "key_principles": ["Unable to parse model response - invalid JSON format"],
                    "patterns": [],
                    "precedents": [],
                    "recommendations": ["Please try again with more specific details about your case"],
                    "risk_factors": ["Technical error in response parsing"],
                    "strong_arguments": [],
                    "likely_outcome": "Unable to determine due to parsing error",
                    "response_time": f"{response_time:.2f} seconds"
                }
            except Exception as parse_err:
                return {
                    "error": f"Error processing response: {str(parse_err)}",
                    "key_principles": [f"Error in parsing model response: {str(parse_err)}"],
                    "patterns": [],
                    "precedents": [],
                    "recommendations": ["Please try again with more specific details about your case"],
                    "risk_factors": [],
                    "strong_arguments": [],
                    "likely_outcome": "Unable to determine due to processing error",
                    "response_time": f"{response_time:.2f} seconds"
                }
        
        except Exception as e:
            return {
                "error": f"Error during analysis: {str(e)}",
                "key_principles": [f"Error in analysis: {str(e)}"],
                "patterns": [],
                "precedents": [],
                "recommendations": ["Please try again later or contact support"],
                "risk_factors": ["System error occurred"],
                "strong_arguments": [],
                "likely_outcome": "Unable to determine due to system error",
                "response_time": "N/A"
            }

# Initialize the legal advisor globally
try:
    legal_advisor = LegalAdvisor()
except ValueError as e:
    print(f"Error initializing Legal Advisor: {e}")
    legal_advisor = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if legal_advisor is None:
        return jsonify({
            "status": "error",
            "message": "Legal Advisor not initialized. Check GEMINI_API_KEY."
        }), 500
    return jsonify({
        "status": "healthy",
        "message": "Legal Advisor API is running"
    }), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_case():
    """Main endpoint for analyzing legal cases"""
    try:
        # Check if legal advisor is initialized
        if legal_advisor is None:
            return jsonify({
                "error": "Service not available",
                "message": "Legal Advisor service is not properly initialized. Please check API key configuration."
            }), 503
        
        data = request.json
        
        # Validate required fields
        if not data.get('case_type') or not data.get('opposition_demand'):
            return jsonify({
                "error": "Missing required fields",
                "message": "Please provide both case_type and opposition_demand"
            }), 400
            
        # Get data from request
        case_type = data.get('case_type')
        opposition_demand = data.get('opposition_demand')
        additional_details = data.get('additional_details', '')
        
        # Process the case analysis
        analysis = legal_advisor.analyze_case(case_type, opposition_demand, additional_details)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Server error while processing your request"
        }), 500

@app.route('/api/example', methods=['GET'])
def get_example():
    """Endpoint to get example case data"""
    return jsonify({
        "case_type": "Property Dispute - Ancestral Property Inheritance",
        "opposition_demand": "Claiming exclusive ownership rights over ancestral property based on a will",
        "additional_details": "Client is one of three siblings. Father passed away leaving ancestral property. Eldest brother claims father verbally expressed wish to give him entire property. No registered will exists. Property has been jointly maintained for 15 years."
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred on the server"
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)