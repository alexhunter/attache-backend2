
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import openai
import os
import json
import requests
import numpy as np
import unicodedata

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
    "berlin": "Berlin"
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

# === Normalize text for accent-insensitive matching ===
def normalise(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower().strip()

# === Filtering Helpers with Debug Output ===
def matches_filters(row, filters):
    tags = [normalise(t) for t in str(row.get("Tags", "")).split(",")]
    types = [normalise(t) for t in str(row.get("Type", "")).split(",")]
    match = False

    if "tags" in filters and filters["tags"]:
        filter_tags = [normalise(t) for t in filters.get("tags", [])]
        tag_match = any(tag in tags for tag in filter_tags)
        if tag_match:
            print(f"✅ TAG MATCH: {row.get('Name')} — matched tags: {filter_tags}")
        match |= tag_match

    if "type" in filters and filters["type"]:
        filter_types = [normalise(t) for t in filters.get("type", [])]
        type_match = any(t in types for t in filter_types)
        if type_match:
            print(f"✅ TYPE MATCH: {row.get('Name')} — matched types: {filter_types}")
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

        print(f"✅ Returning {len(results)} results after filtering.", flush=True)

        # Sanitize JSON output
        sanitised = results.replace({np.nan: None})
        cleaned = json.loads(json.dumps(sanitised.to_dict(orient="records")))
        return jsonify({"results": cleaned})

    except Exception as e:
        import traceback
        print("❌ ERROR during query:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# === Main Entry Point ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
