from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import json
import re
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

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_API_KEY = GEMINI_API_KEY.strip().strip('"').strip("'")
print(f"API Key Status: {'Found' if GEMINI_API_KEY else 'NOT FOUND'}")
print(f"API Key (first 10 chars): {GEMINI_API_KEY[:10] if GEMINI_API_KEY else 'None'}...")

genai = None
model = None  # This will hold the initialized model object
SELECTED_MODEL_NAME = None

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        # Candidate model names to attempt. 'flash' is fast and cost-effective.
        preferred_candidates = [
            'gemini-1.5-flash-latest',
            'gemini-1.5-pro-latest',
            'gemini-1.0-pro'
        ]

        # Try each candidate until one initializes successfully
        for model_name in preferred_candidates:
            try:
                # Test the model by creating an instance
                temp_model = genai.GenerativeModel(model_name)
                # Test with a simple generation call
                # *** CORRECTION: Used 'contents' instead of 'prompt' ***
                test_resp = temp_model.generate_content("Say 'ready'")
                print(f"✓ Gemini model '{model_name}' is usable.")
                SELECTED_MODEL_NAME = model_name
                model = temp_model  # Store the initialized model object
                break
            except Exception as gen_err:
                print(f"✗ Model '{model_name}' not usable: {gen_err}")
                continue

        if not model:
            print("✗ All model attempts failed. Check your API key and model access permissions.")

    except ImportError:
        print("✗ The 'google.generativeai' package is not installed. Please run 'pip install google-generativeai'")
    except Exception as e:
        print(f"✗ Error initializing Gemini SDK: {str(e)}")
else:
    print("✗ GEMINI_API_KEY not found in environment variables. The API will use sample data.")

# --- Sample Data for Fallback ---
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

# --- Helper Functions ---
def create_legal_analysis_prompt(case_type, opposition_demand, additional_details):
    """Create a structured prompt for legal analysis"""
    return f"""You are an expert legal research assistant specializing in Indian law. Analyze the following case and provide comprehensive legal insights.

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

def parse_ai_response(response_text):
    """Parse AI response and extract the JSON object."""
    try:
        # Clean the response text by removing markdown code block fences
        cleaned_text = re.sub(r'```(?:json)?\s*', '', response_text)
        cleaned_text = re.sub(r'```\s*$', '', cleaned_text).strip()
        
        # Parse the cleaned text as JSON
        parsed = json.loads(cleaned_text)
        
        # Basic validation
        required_fields = ['key_principles', 'patterns', 'precedents', 
                           'recommendations', 'risk_factors', 'strong_arguments', 'likely_outcome']
        for field in required_fields:
            if field not in parsed:
                print(f"Warning: Missing field '{field}' in parsed AI response.")
        
        print("✓ Successfully parsed AI response")
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error: {e}")
        print(f"--- AI Response Start ---\n{response_text}\n--- AI Response End ---")
        return None
    except Exception as e:
        print(f"✗ An unexpected error occurred during parsing: {e}")
        return None

# --- API Endpoints ---
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_available": model is not None,
        "selected_model": SELECTED_MODEL_NAME,
        "api_key_configured": bool(GEMINI_API_KEY)
    }), 200

@app.route('/api/example', methods=['GET'])
def get_example():
    """Return example case data"""
    return jsonify(EXAMPLE_DATA), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_case():
    """Analyze legal case and return insights"""
    start_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        case_type = data.get('case_type', '').strip()
        opposition_demand = data.get('opposition_demand', '').strip()
        additional_details = data.get('additional_details', '').strip()
        
        if not case_type or not opposition_demand:
            return jsonify({"error": "Both 'case_type' and 'opposition_demand' are required"}), 400
        
        print(f"\n{'='*50}\nAnalyzing case: {case_type}")
        
        analysis = None
        source = "Unknown"

        # If a model was successfully initialized, use it
        if model:
            try:
                print(f"Generating AI analysis using '{SELECTED_MODEL_NAME}'...")
                prompt = create_legal_analysis_prompt(case_type, opposition_demand, additional_details)
                
                # *** CORRECTION: Use the initialized 'model' object ***
                # *** CORRECTION: Pass prompt to 'contents' ***
                # *** IMPROVEMENT: Request JSON output format ***
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.7
                    }
                )
                
                print(f"AI response received (length: {len(response.text)} chars)")
                analysis = parse_ai_response(response.text)
                
                if analysis:
                    source = "AI Generated"
                    analysis["model"] = SELECTED_MODEL_NAME
                else:
                    source = "Sample Data (Parse Error)"
                    
            except Exception as e:
                print(f"✗ AI generation error: {str(e)}")
                source = "Sample Data (API Error)"
        else:
            print("✗ Model not available, falling back to sample data.")
            source = "Sample Data (No Model)"

        # If analysis failed or model wasn't available, use fallback data
        if not analysis:
            analysis = SAMPLE_ANALYSIS.copy()
            analysis["likely_outcome"] = f"Based on the '{case_type.lower()}' case, a strategic legal approach is recommended. (This is sample data as AI generation failed)."

        analysis["source"] = source
        response_time = round(time.time() - start_time, 2)
        analysis["response_time"] = f"{response_time} seconds"
        
        print(f"Analysis complete in {response_time}s. Source: {source}")
        print(f"{'='*50}\n")
        
        return jsonify(analysis), 200
        
    except Exception as e:
        print(f"✗ Critical error in /api/analyze: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# --- Boilerplate and Main Execution ---
@app.route('/')
def index():
    return "Legal Research API Backend is running. Use the /api endpoints."

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"\n{'='*50}")
    print(f"Starting Legal Research API on http://localhost:{port}")
    print(f"Model Status: {'✓ Ready (' + SELECTED_MODEL_NAME + ')' if model else '✗ Not Available'}")
    print(f"API Key Status: {'✓ Configured' if GEMINI_API_KEY else '✗ Not Configured'}")
    print(f"{'='*50}\n")
    # debug=True is great for development, but should be False in production
    app.run(host='0.0.0.0', port=port, debug=True)