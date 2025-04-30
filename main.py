from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import openai
import os
import json
import traceback  # ✅ Move this import to the top level

# Set up the app
app = Flask(__name__)
CORS(app)

# Securely load OpenAI key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your dataset once at startup
CSV_PATH = "attache_cleaned_places.csv"
df = pd.read_csv(CSV_PATH)

@app.route("/query", methods=["POST"])
def query():
    user_input = request.json.get("prompt", "")

    # Build prompt for OpenAI
    gpt_prompt = f"""
You are a travel concierge for a curated app called Attaché.
You ONLY interpret user requests into structured filters to search a private database.
Return valid JSON with the following fields only:
- city: string
- category: list of strings (e.g., Food, Drink, Stay, See, Tip)
- tags: list of strings (e.g., Romantic, Trendy, Coffee)
- duration_hours: number (optional)
- preferences: string (optional)

USER REQUEST:
{user_input}
    """

    try:
        # Call OpenAI to interpret the user prompt
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": gpt_prompt}],
            temperature=0.3
        )

        content = response["choices"][0]["message"]["content"]
        filters = json.loads(content)

        # Logging to Render console
        print("----- GPT FILTERS -----")
        print(json.dumps(filters, indent=2))
        print("----- CSV TOTAL ROWS -----")
        print(f"{len(df)} total rows before filtering")

        # Start filtering
        results = df.copy()

        # City
        if "city" in filters:
            results = results[results["City"].str.contains(filters["city"], case=False, na=False)]

        # Category
        if "category" in filters and isinstance(filters["category"], list):
            results = results[results["Category"].isin(filters["category"])]

        # Tags (with fallback)
        if "tags" in filters and filters["tags"]:
            tag_mask = results["Tags"].apply(
                lambda x: any(tag.lower() in str(x).lower() for tag in filters["tags"])
            )
            tag_filtered = results[tag_mask]

            if not tag_filtered.empty:
                results = tag_filtered  # Only apply if it yields something

        # Log final result count
        print(f"Returning {len(results)} results after filtering.")

        return jsonify({"results": results.to_dict(orient="records")})

    except Exception as e:  # ✅ Correctly dedented
        print("ERROR during query:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Run only if executed directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
