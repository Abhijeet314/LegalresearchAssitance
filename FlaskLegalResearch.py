import together
import json
import re
import os
import time
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, Field
from dotenv import load_dotenv

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
        # Initialize Together AI client
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        if not self.together_api_key:
            raise ValueError("API key not found. Please set the TOGETHER_API_KEY environment variable.")
            
        # Initialize the Together client
        self.client = together.Together(api_key=self.together_api_key)
        
    def analyze_case(self, case_type: str, opposition_demand: str, additional_details: str) -> Dict:
        """Analyze the case using the LLM without external data"""
        try:
            # Start timer to measure response time
            start_time = time.time()
            
            # Create the prompt directly here for simplicity
            prompt = f"""You are a highly knowledgeable legal AI assistant specialized in Indian law, with extensive experience advising lawyers on legal strategy. You have detailed knowledge of Indian legal precedents, statutes, and court judgments up to your training data.

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
"key_principles": ["principle 1", "principle 2", ...],
"patterns": ["pattern 1", "pattern 2", ...],
"precedents": [
    {{"case": "Case Name 1", "outcome": "Detailed outcome and relevance"}},
    {{"case": "Case Name 2", "outcome": "Detailed outcome and relevance"}},
    ...
],
"recommendations": ["recommendation 1", "recommendation 2", ...],
"risk_factors": ["risk 1", "risk 2", ...],
"strong_arguments": ["argument 1", "argument 2", ...],
"likely_outcome": "Detailed prediction of the case outcome"
}}

IMPORTANT: The JSON response must be valid JSON with no leading or trailing whitespace around keys. Do not include newlines before key names. The keys must be exactly as shown above. Include at least 3-5 items per field where appropriate.

DO NOT include any text outside of the JSON structure. The entire response should be valid JSON."""
            
            # Make the API call directly
            response = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-classifier",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8192
            )
            
            # Get text from the response
            raw_response = response.choices[0].message.content
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Parse JSON response
            try:
                # Clean the response text for JSON parsing
                clean_text = raw_response.strip()
                
                # Handle markdown code blocks if present
                if "```json" in clean_text:
                    match = re.search(r"```json\n(.*?)\n```", clean_text, re.DOTALL)
                    if match:
                        clean_text = match.group(1)
                elif "```" in clean_text:
                    match = re.search(r"```\n(.*?)\n```", clean_text, re.DOTALL)
                    if match:
                        clean_text = match.group(1)
                
                # Parse JSON
                analysis_dict = json.loads(clean_text)
                
                # Create Pydantic model instance
                analysis = LegalCaseAnalysis(**analysis_dict)
                
                # Convert to dictionary for JSON response
                response_data = analysis.dict()
                response_data["response_time"] = f"{response_time:.2f} seconds"
                
                return response_data
            
            except Exception as json_err:
                return {
                    "error": f"Error parsing JSON response: {json_err}",
                    "key_principles": ["Error in parsing model response"],
                    "recommendations": ["Please try again with more specific details about your case"],
                    "response_time": f"{response_time:.2f} seconds"
                }
        
        except Exception as e:
            return {
                "error": f"Error during analysis: {str(e)}",
                "key_principles": [f"Error in analysis: {str(e)}"],
                "recommendations": ["Please try again later"],
                "response_time": "N/A"
            }

# Initialize the legal advisor globally
legal_advisor = LegalAdvisor()

@app.route('/api/analyze', methods=['POST'])
def analyze_case():
    try:
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
    return jsonify({
        "case_type": "Property Dispute - Ancestral Property Inheritance",
        "opposition_demand": "Claiming exclusive ownership rights over ancestral property based on a will",
        "additional_details": "Client is one of three siblings. Father passed away leaving ancestral property. Eldest brother claims father verbally expressed wish to give him entire property. No registered will exists. Property has been jointly maintained for 15 years."
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)