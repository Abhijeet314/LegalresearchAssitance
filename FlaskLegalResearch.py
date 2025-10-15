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
        
        # Initialize the Gemini model with proper configuration
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
        )
        
    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from response text that might contain markdown or extra text"""
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        
        # Find JSON object boundaries
        if '{' in text and '}' in text:
            start_idx = text.find('{')
            # Find the matching closing brace
            brace_count = 0
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start_idx:i+1]
        return text
        
    def analyze_case(self, case_type: str, opposition_demand: str, additional_details: str) -> Dict:
        """Analyze the case using Gemini LLM without external data"""
        try:
            # Start timer to measure response time
            start_time = time.time()
            
            # Create the prompt for Gemini - more structured for JSON output
            prompt = f"""You are a legal AI assistant specialized in Indian law. Analyze the following case and provide a JSON response.

CASE DETAILS:
- Case Type: {case_type}
- Opposition's Demand: {opposition_demand}
- Additional Details: {additional_details}

Provide a comprehensive legal analysis in the following JSON structure:

{{
  "key_principles": [
    "List 3-5 key legal principles under Indian law that apply to this case"
  ],
  "patterns": [
    "List 3-5 common patterns seen in similar cases in Indian courts"
  ],
  "precedents": [
    {{
      "case": "Full case name (e.g., State of Maharashtra v. Som Nath Thapa)",
      "outcome": "Detailed outcome and how it applies to this case"
    }}
  ],
  "recommendations": [
    "List 3-5 strategic recommendations with specific tactics"
  ],
  "risk_factors": [
    "List 3-5 potential risks and challenges to prepare for"
  ],
  "strong_arguments": [
    "List 3-5 persuasive arguments for Indian judges"
  ],
  "likely_outcome": "Detailed prediction based on precedents and current legal climate in India"
}}

Ensure all precedents are actual Indian legal cases. Provide detailed, actionable insights."""
            
            # Make the API call to Gemini with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    
                    # Check if response was blocked
                    if not response.text:
                        if hasattr(response, 'prompt_feedback'):
                            return {
                                "error": "Response blocked by safety filters",
                                "key_principles": ["The request was blocked due to safety settings"],
                                "patterns": [],
                                "precedents": [],
                                "recommendations": ["Please rephrase your case details and try again"],
                                "risk_factors": [],
                                "strong_arguments": [],
                                "likely_outcome": "Unable to analyze due to content filtering",
                                "response_time": f"{time.time() - start_time:.2f} seconds"
                            }
                    
                    # Get text from the response
                    raw_response = response.text
                    break
                    
                except Exception as api_err:
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    else:
                        raise api_err
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Parse JSON response
            try:
                # Clean the response text for JSON parsing
                clean_text = self._extract_json_from_response(raw_response.strip())
                
                # Parse JSON
                analysis_dict = json.loads(clean_text)
                
                # Validate and ensure all required fields exist
                required_fields = {
                    "key_principles": [],
                    "patterns": [],
                    "precedents": [],
                    "recommendations": [],
                    "risk_factors": [],
                    "strong_arguments": [],
                    "likely_outcome": ""
                }
                
                for field, default in required_fields.items():
                    if field not in analysis_dict:
                        analysis_dict[field] = default
                
                # Create Pydantic model instance for validation
                analysis = LegalCaseAnalysis(**analysis_dict)
                
                # Convert to dictionary for JSON response
                response_data = analysis.model_dump() if hasattr(analysis, 'model_dump') else analysis.dict()
                response_data["response_time"] = f"{response_time:.2f} seconds"
                
                return response_data
            
            except json.JSONDecodeError as json_err:
                # If JSON parsing fails, return a structured error with raw response for debugging
                print(f"JSON Decode Error: {json_err}")
                print(f"Raw response: {raw_response[:500]}")  # Print first 500 chars for debugging
                
                return {
                    "error": f"Error parsing JSON response: {str(json_err)}",
                    "key_principles": ["Unable to parse model response - invalid JSON format"],
                    "patterns": ["The AI model returned an improperly formatted response"],
                    "precedents": [],
                    "recommendations": ["Please try again with more specific details about your case"],
                    "risk_factors": ["Technical error in response parsing"],
                    "strong_arguments": [],
                    "likely_outcome": "Unable to determine due to parsing error",
                    "response_time": f"{response_time:.2f} seconds",
                    "debug_info": f"Response preview: {raw_response[:200]}..."
                }
            except Exception as parse_err:
                print(f"Parse Error: {parse_err}")
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
            print(f"Analysis Error: {e}")
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
    print("Legal Advisor initialized successfully")
except ValueError as e:
    print(f"Error initializing Legal Advisor: {e}")
    legal_advisor = None
except Exception as e:
    print(f"Unexpected error initializing Legal Advisor: {e}")
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
        case_type = data.get('case_type', '').strip()
        opposition_demand = data.get('opposition_demand', '').strip()
        additional_details = data.get('additional_details', '').strip()
        
        # Validate non-empty after strip
        if not case_type or not opposition_demand:
            return jsonify({
                "error": "Invalid input",
                "message": "case_type and opposition_demand cannot be empty"
            }), 400
        
        # Process the case analysis
        analysis = legal_advisor.analyze_case(case_type, opposition_demand, additional_details)
        
        return jsonify(analysis)
        
    except Exception as e:
        print(f"Request Error: {e}")
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
    print("Starting Legal Advisor API...")
    app.run(debug=True, host='0.0.0.0', port=5000)