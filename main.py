
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import openai
import os
import json
import requests

# === Setup ===
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
openai.api_key = os.getenv("OPENAI_API_KEY")

# Airtable config
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
BASE_ID = "app0NvSPOVHFrDuM9"
TABLE_NAME = "Places"
AIRTABLE_URL = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}

# City aliases map
CITY_ALIASES = {
    "nyc": "New York",
    "la": "Los Angeles",
    "sf": "San Francisco",
    "nola": "New Orleans",
    "cdmx": "Mexico City",
    "bcn": "Barcelona",
    "ldn": "London",
    "berlin": "Berlin",
    "vienna": "Wien",
    "tel aviv": "Tel Aviv-Yafo",
    "lisbon": "Lisboa",
    "bcn": "Barcelona",
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

        if "records" not in data:
            print("❌ Airtable API ERROR:")
            print("RAW RESPONSE:", response.text)
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

        # Apply city alias if needed
        city = filters.get("city", "").lower()
        filters["city"] = CITY_ALIASES.get(city, filters.get("city"))

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

        import numpy as np

        print(f"✅ Returning {len(results)} results after filtering.", flush=True)

        # Step 1: Replace all NaNs with None
        sanitised = results.replace({np.nan: None})

        # Step 2: Force conversion to JSON-safe values
        cleaned = json.loads(json.dumps(sanitised.to_dict(orient="records")))

        # Step 3: Return JSON-safe payload
        return jsonify({"results": cleaned})

    except Exception as e:
        import traceback
        print("❌ ERROR during query:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# === Main Entry Point ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
