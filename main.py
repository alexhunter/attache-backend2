
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import openai
import os
import json
import requests

# === Setup ===
app = Flask(__name__)
CORS(app)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Airtable config
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
BASE_ID = "app5AeI5uilErzEbw"
TABLE_NAME = "Places"
AIRTABLE_URL = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}"
}

# === Airtable Pull ===
def load_airtable_data():
    records = []
    offset = None
    while True:
        params = {}
        if offset:
            params["offset"] = offset
        response = requests.get(AIRTABLE_URL, headers=HEADERS, params=params)
        data = response.json()

        # Debug: print errors
        if "records" not in data:
            print("❌ Airtable API ERROR:")
            print(json.dumps(data, indent=2))
            raise Exception("Airtable API request failed")

        records.extend(data["records"])
        offset = data.get("offset")
        if not offset:
            break
    return pd.DataFrame([r["fields"] for r in records])

# === Filtering Helpers ===
def matches_filters(row, filters):
    tags = str(row.get("Tags", "")).lower().split(", ")
    types = str(row.get("Type", "")).lower().split(", ")
    match = False

    if "tags" in filters and filters["tags"]:
        tag_match = any(tag.lower() in tags for tag in filters["tags"])
        match |= tag_match

    if "type" in filters and filters["type"]:
        type_match = any(t.lower() in types for t in filters["type"])
        match |= type_match

    return match

# === Routes ===
@app.route("/query", methods=["POST"])
def query():
    user_input = request.json.get("prompt", "")

    gpt_prompt = f"""
You are a travel concierge for a curated app called Attaché.
You ONLY interpret user requests into structured filters to search a private database.
Return valid JSON with the following fields only:
- city: string
- category: list of strings (e.g., Food, Drink, Stay, See, Tip)
- tags: list of strings (e.g., Romantic, Trendy, Coffee)
- type: list of strings (e.g., Bakery, Bar, Café)
- duration_hours: number (optional)
- preferences: string (optional)

USER REQUEST:
{user_input}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": gpt_prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content
        filters = json.loads(content)

        print("🟢 Received query:", user_input, flush=True)
        print("----- GPT FILTERS -----")
        print(json.dumps(filters, indent=2), flush=True)

        df = load_airtable_data()
        print("----- Airtable rows:", len(df), "-----", flush=True)

        results = df.copy()

        # Filter by City
        if "city" in filters:
            results = results[results["City"].str.contains(filters["city"], case=False, na=False)]

        # Filter by Category
        if "category" in filters and isinstance(filters["category"], list):
            results = results[results["Category"].isin(filters["category"])]

        # Filter by tags or type
        if not results.empty and ("tags" in filters or "type" in filters):
            filtered = results[results.apply(lambda row: matches_filters(row, filters), axis=1)]
            if not filtered.empty:
                results = filtered
            else:
                print("No tag/type matches — fallback to city/category only", flush=True)

        print(f"✅ Returning {len(results)} results after filtering.", flush=True)

        sanitised = results.where(pd.notnull(results), None)
        return jsonify({"results": sanitised.to_dict(orient="records")})

    except Exception as e:
        import traceback
        print("❌ ERROR during query:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# === Main Entry Point ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
