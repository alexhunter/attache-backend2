from flask import Flask, request, jsonify
import pandas as pd
import openai
import json

# Config
openai.api_key = "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"
CSV_PATH = "attache_cleaned_places.csv"

# Load data
df = pd.read_csv(CSV_PATH)

# App setup
app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    user_input = request.json.get("prompt", "")

    # Create prompt for GPT
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

    # Call OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": gpt_prompt}],
        temperature=0.3
    )

    try:
        parsed = json.loads(response["choices"][0]["message"]["content"])
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse GPT output"}), 500

    # Apply filters to dataset
    filtered = df[
        df["City"].str.contains(parsed["city"], case=False, na=False)
    ]

    if "category" in parsed:
        filtered = filtered[filtered["Category"].isin(parsed["category"])]

    if "tags" in parsed and parsed["tags"]:
        tag_mask = filtered["Tags"].apply(lambda x: any(tag in str(x) for tag in parsed["tags"]))
        filtered = filtered[tag_mask]

    results = filtered.to_dict(orient="records")

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=81)
