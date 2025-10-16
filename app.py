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

genai = None
model = None  # This will hold the initialized model object, not just a boolean
SELECTED_MODEL_NAME = None

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        # *** IMPROVEMENT: Simpler and more robust model selection ***
        # Try preferred, modern model names first.
        preferred_candidates = [
            'gemini-1.5-flash-latest',
            'gemini-1.5-pro-latest',
            'gemini-1.0-pro'
        ]

        for model_name in preferred_candidates:
            try:
                # Attempt to create a model instance
                temp_model = genai.GenerativeModel(model_name)
                # *** CORRECTION: Use 'contents' argument (pass prompt directly) ***
                temp_model.generate_content("test") # A quick check to see if it works
                
                print(f"✓ Successfully initialized Gemini model: '{model_name}'")
                SELECTED_MODEL_NAME = model_name
                model = temp_model  # Store the actual model object
                break # Stop after finding a working model
            except Exception as e:
                print(f"✗ Could not initialize model '{model_name}': {e}")
                continue
        
        if not model:
            print("✗ All model initialization attempts failed. Check API key and access permissions.")

    except ImportError:
        print("✗ The 'google.generativeai' package is not installed. Please run 'pip install google-generativeai'")
    except Exception as e:
        print(f"✗ Error during Gemini SDK setup: {str(e)}")
else:
    print("✗ GEMINI_API_KEY not found in environment. API will use sample data.")

# --- Sample Data for Fallback ---
SAMPLE_ANALYSIS = {
    "key_principles": ["Principle 1 from sample data.", "Principle 2 from sample data."],
    "patterns": ["Pattern 1 from sample data."],
    "precedents": [{"case": "Sample Case vs. State", "outcome": "Sample outcome."}],
    "recommendations": ["Recommendation 1 from sample data."],
    "risk_factors": ["Risk 1 from sample data."],
    "strong_arguments": ["Strong argument from sample data."],
    "likely_outcome": "This is a sample outcome as the AI model could not be reached."
}

EXAMPLE_DATA = {
    "case_type": "Property Dispute",
    "opposition_demand": "Claiming ownership of ancestral property through adverse possession.",
    "additional_details": "The opposing party claims they have been in continuous possession of the property for 15 years."
}

# --- Helper Functions ---
def create_legal_analysis_prompt(case_type, opposition_demand, additional_details):
    return f"""Analyze the following Indian law case. Return a valid JSON object.
Case Type: {case_type}
Opposition's Claim: {opposition_demand}
Details: {additional_details}
JSON Structure: {{
  "key_principles": ["principle1", ...], "patterns": ["pattern1", ...],
  "precedents": [{{"case": "Name (Year)", "outcome": "..."}}, ...],
  "recommendations": ["recommendation1", ...], "risk_factors": ["risk1", ...],
  "strong_arguments": ["argument1", ...], "likely_outcome": "assessment"
}}"""

def parse_ai_response(response_text):
    """Safely parses JSON from the AI's response text."""
    try:
        # Clean the response by removing markdown fences
        cleaned_text = re.sub(r'```(?:json)?\s*', '', response_text).strip()
        parsed = json.loads(cleaned_text)
        print("✓ Successfully parsed AI response.")
        return parsed
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error: {e}")
        print(f"--- Raw AI Response ---\n{response_text}\n----------------------")
        return None

# --- API Endpoints ---
@app.route('/api/analyze', methods=['POST'])
def analyze_case():
    start_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        case_type = data.get('case_type', '').strip()
        opposition_demand = data.get('opposition_demand', '').strip()
        
        if not case_type or not opposition_demand:
            return jsonify({"error": "Both 'case_type' and 'opposition_demand' are required"}), 400
        
        analysis = None
        source = "Unknown"

        if model:
            try:
                print(f"\n{'='*20}\nGenerating AI analysis with '{SELECTED_MODEL_NAME}'...")
                prompt = create_legal_analysis_prompt(
                    case_type, 
                    opposition_demand, 
                    data.get('additional_details', '')
                )
                
                # *** CORRECTION: Call generate_content on the model object and pass the prompt directly ***
                # *** IMPROVEMENT: Request JSON output explicitly for better reliability ***
                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                
                analysis = parse_ai_response(response.text)
                if analysis:
                    source = f"AI Generated ({SELECTED_MODEL_NAME})"
                else:
                    source = "Sample Data (AI Response Parse Error)"
                
            except Exception as e:
                print(f"✗ AI generation error: {str(e)}")
                source = "Sample Data (API Call Error)"
        else:
            print("✗ Model not available, using sample data.")
            source = "Sample Data (Model Not Initialized)"

        # If AI analysis failed at any stage, use the fallback data
        if not analysis:
            analysis = SAMPLE_ANALYSIS.copy()
            analysis["likely_outcome"] = f"Based on the '{case_type.lower()}' case, a strategic legal approach is recommended. ({source})"

        response_time = round(time.time() - start_time, 2)
        analysis["source"] = source
        analysis["response_time"] = f"{response_time} seconds"
        
        print(f"Analysis complete in {response_time}s. Source: {source}")
        return jsonify(analysis), 200
        
    except Exception as e:
        print(f"✗ Critical error in analyze_case: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# (Other endpoints like /api/health, /api/example, etc. can remain the same)
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_available": model is not None,
        "selected_model": SELECTED_MODEL_NAME,
        "api_key_configured": bool(GEMINI_API_KEY)
    }), 200

# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"\n{'='*50}")
    print(f"Starting Legal Research API on http://localhost:{port}")
    print(f"Model Status: {'✓ Ready (' + SELECTED_MODEL_NAME + ')' if model else '✗ Not Available'}")
    print(f"API Key Status: {'✓ Configured' if GEMINI_API_KEY else '✗ Not Configured'}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, debug=True)