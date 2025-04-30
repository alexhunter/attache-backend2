from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import openai
import os
import json
import traceback

# Set up the app
app = Flask(__name__)
CORS(app)

# Create OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load dataset once on startup
CSV_PATH = "attache_cleaned_places.csv"
df = pd.read_csv(CSV_PATH)
print("----- DISTINCT CITIES IN CSV -----", flush=True)
print(df["City"].dropna().unique(), flush=True)

@app.route("/query", methods=["POST"])
def query():
    user_input = request.json.get("prompt", "")
    print(f"🟢 Received query: {user_input}", flush=True)

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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": gpt_prompt}],
            temperature=0.3
        )

        content = response.choices[0].message.content
        filters = json.loads(content)

        print("----- GPT FILTERS -----", flush=True)
        print(json.dumps(filters, indent=2), flush=True)
        print(f"----- CSV TOTAL ROWS: {len(df)} -----", flush=True)

        # Always start with full dataset
        results = df.copy()

        # City filter
        if "city" in filters:
            results = results[results["City"].str.contains(filters["city"], case=False, na=False)]

        # Category filter
        if "category" in filters and isinstance(filters["category"], list):
            results = results[results["Category"].isin(filters["category"])]

        # Tag filter (optional fallback)
        if "tags" in filters and filters["tags"]:
            tag_mask = results["Tags"].apply(
                lambda x: any(tag.lower() in str(x).lower() for tag in filters["tags"])
            )
            tag_filtered = results[tag_mask]
            if not tag_filtered.empty:
                results = tag_filtered
            else:
                print("No tag matches — falling back to city/category only", flush=True)

        if results.empty:
            print("No matches found — fallback to city-only results", flush=True)
            results = df[df["City"].str.contains(filters["city"], case=False, na=False)]

        print(f"✅ Returning {len(results)} results after filtering.", flush=True)
        return jsonify({"results": results.to_dict(orient="records")})

    except Exception as e:
        print("❌ ERROR during query:", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Run the app locally if needed
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
