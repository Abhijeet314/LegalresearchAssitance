from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS to allow requests from your frontend
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],  # In production, specify your frontend domain
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
print(f"API Key Status: {'Found' if GEMINI_API_KEY else 'NOT FOUND'}")
print(f"API Key (first 10 chars): {GEMINI_API_KEY[:10] if GEMINI_API_KEY else 'None'}...")

model = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try different model names in order of preference
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro'
        ]
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                # Test the model with a simple prompt
                test_response = model.generate_content("Say 'ready'")
                print(f"✓ Gemini model '{model_name}' initialized successfully")
                break
            except Exception as model_error:
                print(f"✗ Model '{model_name}' failed: {str(model_error)}")
                continue
        
        if not model:
            print("✗ All model attempts failed")
            
    except Exception as e:
        print(f"✗ Error initializing Gemini: {str(e)}")
        model = None
else:
    print("✗ GEMINI_API_KEY not found in environment variables")

# Sample data for fallback
SAMPLE_ANALYSIS = {
    "key_principles": [
        "Property rights under Indian law are governed by the Transfer of Property Act, 1882",
        "Adverse possession requires continuous, open, and hostile occupation for 12 years",
        "Title documents must be verified for clear ownership rights"
    ],
    "patterns": [
        "Property disputes often involve documentation issues",
        "Boundary disputes require survey evidence",
        "Family property disputes need clear succession documentation"
    ],
    "precedents": [
        {
            "case": "Ram Singh vs. State of Punjab (2019)",
            "outcome": "Court ruled in favor of documented ownership over adverse possession claims"
        },
        {
            "case": "Maharashtra State vs. Property Developers (2021)",
            "outcome": "Established precedent for property registration requirements"
        }
    ],
    "recommendations": [
        "Gather all original property documents and verify their authenticity",
        "Conduct a thorough title search to identify any encumbrances",
        "Obtain a current survey report to establish exact boundaries",
        "File for injunctive relief to prevent further encroachment"
    ],
    "risk_factors": [
        "Weak documentation may undermine ownership claims",
        "Delay in legal action could strengthen adverse possession claims",
        "Conflicting family interests may complicate the case"
    ],
    "strong_arguments": [
        "Clear chain of title through registered documents",
        "Continuous payment of property taxes demonstrates ownership",
        "No evidence of 12-year continuous hostile possession by opposition"
    ],
    "likely_outcome": "Favorable outcome expected if proper documentation is presented and legal action is taken promptly"
}

EXAMPLE_DATA = {
    "case_type": "Property Dispute",
    "opposition_demand": "Claiming ownership of ancestral property through adverse possession",
    "additional_details": "The opposing party claims they have been in continuous possession of the property for 15 years and are seeking ownership rights. Our client has original property documents and has been paying taxes regularly."
}

def create_legal_analysis_prompt(case_type, opposition_demand, additional_details):
    """Create a structured prompt for legal analysis"""
    prompt = f"""You are an expert legal research assistant specializing in Indian law. Analyze the following case and provide comprehensive legal insights.

**Case Details:**
- Case Type: {case_type}
- Opposition's Demand/Claim: {opposition_demand}
- Additional Details: {additional_details if additional_details else "No additional details provided"}

**Provide a detailed analysis covering:**

1. KEY LEGAL PRINCIPLES (3-5 relevant legal principles from Indian law)
2. COMMON PATTERNS (3-4 patterns seen in similar cases)
3. RELEVANT PRECEDENTS (2-3 actual or representative Indian case precedents)
4. STRATEGIC RECOMMENDATIONS (4-6 actionable steps)
5. RISK FACTORS (3-4 potential risks to consider)
6. STRONG ARGUMENTS (3-4 strong legal arguments for the client)
7. LIKELY OUTCOME (brief assessment with reasoning)

**IMPORTANT**: Return your response as a valid JSON object with this exact structure:
{{
  "key_principles": ["principle1", "principle2", ...],
  "patterns": ["pattern1", "pattern2", ...],
  "precedents": [
    {{"case": "Case Name vs. Party (Year)", "outcome": "Brief outcome description"}},
    ...
  ],
  "recommendations": ["recommendation1", "recommendation2", ...],
  "risk_factors": ["risk1", "risk2", ...],
  "strong_arguments": ["argument1", "argument2", ...],
  "likely_outcome": "Detailed outcome assessment"
}}

Ensure the JSON is properly formatted and all fields are populated with substantive legal analysis based on Indian law."""
    
    return prompt

def parse_ai_response(response_text):
    """Parse AI response and structure it properly"""
    import json
    import re
    
    try:
        # Clean the response text
        cleaned = response_text.strip()
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', cleaned, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = cleaned
        
        # Parse JSON
        parsed = json.loads(json_str)
        
        # Validate required fields
        required_fields = ['key_principles', 'patterns', 'precedents', 
                          'recommendations', 'risk_factors', 'strong_arguments', 
                          'likely_outcome']
        
        for field in required_fields:
            if field not in parsed:
                print(f"Warning: Missing field '{field}' in AI response")
                parsed[field] = [] if field != 'likely_outcome' else 'Analysis unavailable'
        
        print("✓ Successfully parsed AI response")
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error: {str(e)}")
        print(f"Response text: {response_text[:500]}...")
        return None
    except Exception as e:
        print(f"✗ Error parsing response: {str(e)}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_available": model is not None,
        "api_key_configured": GEMINI_API_KEY is not None
    }), 200

@app.route('/api/example', methods=['GET'])
def get_example():
    """Return example case data"""
    return jsonify(EXAMPLE_DATA), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_case():
    """Analyze legal case and return insights"""
    try:
        start_time = time.time()
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        case_type = data.get('case_type', '').strip()
        opposition_demand = data.get('opposition_demand', '').strip()
        additional_details = data.get('additional_details', '').strip()
        
        # Validate required fields
        if not case_type or not opposition_demand:
            return jsonify({
                "error": "Both case_type and opposition_demand are required"
            }), 400
        
        print(f"\n{'='*50}")
        print(f"Analyzing case: {case_type}")
        print(f"Model available: {model is not None}")
        
        # If model is available, use AI analysis
        if model:
            try:
                print("Generating AI analysis...")
                prompt = create_legal_analysis_prompt(
                    case_type, 
                    opposition_demand, 
                    additional_details
                )
                
                # Generate response with error handling
                response = model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 40,
                        'max_output_tokens': 2048,
                    }
                )
                
                print(f"AI response received (length: {len(response.text)} chars)")
                
                # Parse the response
                analysis = parse_ai_response(response.text)
                
                if analysis:
                    print("✓ Using AI-generated analysis")
                    # Add metadata
                    analysis["source"] = "AI Generated"
                    analysis["model"] = "Gemini 1.5 Flash"
                else:
                    print("✗ Failed to parse AI response, using sample data")
                    analysis = SAMPLE_ANALYSIS.copy()
                    analysis["source"] = "Sample Data (Parse Error)"
                    analysis["likely_outcome"] = f"AI analysis failed to parse. Based on the {case_type.lower()} case, strategic legal approach recommended."
                
            except Exception as e:
                print(f"✗ AI generation error: {str(e)}")
                # Fallback to sample data with customization
                analysis = SAMPLE_ANALYSIS.copy()
                analysis["source"] = "Sample Data (API Error)"
                analysis["likely_outcome"] = f"Based on the {case_type.lower()} case details provided, analysis suggests proceeding with strategic legal approach to address the opposition's demand."
                analysis["error_details"] = str(e)
        else:
            print("✗ Model not available, using sample data")
            # Use sample data if no model
            analysis = SAMPLE_ANALYSIS.copy()
            analysis["source"] = "Sample Data (No API Key)"
            analysis["likely_outcome"] = f"Based on the {case_type.lower()} case details provided, analysis suggests proceeding with strategic legal approach to address the opposition's demand."
        
        # Calculate response time
        response_time = round(time.time() - start_time, 2)
        analysis["response_time"] = f"{response_time} seconds"
        
        print(f"Analysis complete in {response_time}s")
        print(f"Source: {analysis.get('source', 'Unknown')}")
        print(f"{'='*50}\n")
        
        return jsonify(analysis), 200
        
    except Exception as e:
        print(f"✗ Error in analyze_case: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/api/test', methods=['GET'])
def test():
    """Simple test endpoint"""
    return jsonify({
        "message": "Legal Research API is running",
        "model_status": "Available" if model else "Not Available",
        "api_key_status": "Configured" if GEMINI_API_KEY else "Not Configured",
        "endpoints": {
            "health": "/api/health",
            "example": "/api/example",
            "analyze": "/api/analyze (POST)",
            "test": "/api/test"
        }
    }), 200

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"\n{'='*50}")
    print(f"Starting Legal Research API on port {port}")
    print(f"Model Status: {'✓ Ready' if model else '✗ Not Available'}")
    print(f"API Key Status: {'✓ Configured' if GEMINI_API_KEY else '✗ Not Configured'}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, debug=True)