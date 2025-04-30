from flask_cors import CORS
from flask import Flask, request, jsonify
import pandas as pd
import openai
import os
import json

# Set up the app
app = Flask(__name__)

# Securely pull API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your dataset once at startup
CSV_PATH = "attache_cleaned_places.csv"
df = pd.read_csv(CSV_PATH)

@app.route("/query", methods=["POST"])
def query():
    user_input = request.json.get("prompt", "")

    # Construct the GPT prompt
    gpt_prompt = f"""
You are a travel concierge for a curated app called Attaché. 
You ONLY interpret user requests into structured filters to search a private database.
Return valid JSON with:
- city
- category (e.g. Food, Drink, Stay, See, Tip)
- tags (e.g. Romantic, Trendy, Coffee)
- duration_hours
- preferences

USER REQUEST:
{user_input}
    """

    try:
        # Call GPT to interpret the prompt
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": gpt_prompt}],
            temperature=0.3
        )

        # Parse JSON output from GPT
        content = response["choices"][0]["message"]["content"]
        filters = json.loads(content)

        # Filter the dataset
        results = df.copy()

        if "city" in filters:
            results = results[results["City"].str.contains(filters["city"], case=False, na=False)]

        if "category" in filters and isinstance(filters["category"], list):
            results = results[results["Category"].isin(filters["category"])]

        if "tags" in filters and filters["tags"]:
            results = results[results["Tags"].apply(
                lambda x: any(tag.lower() in str(x).lower() for tag in filters["tags"])
            )]

        return jsonify({"results": results.to_dict(orient="records")})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run only if not being imported
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
