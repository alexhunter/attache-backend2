import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from urllib.parse import quote

app = Flask(__name__)
CORS(app)

# Airtable config
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN", "patnWjj7xlvsmIyRU")
BASE_ID = "app0NvSPOVHFrDuM9"
TABLE_NAME = "Places"
AIRTABLE_URL = f"https://api.airtable.com/v0/{BASE_ID}/{quote(TABLE_NAME)}"
HEADERS = {"Authorization": f"Bearer {AIRTABLE_TOKEN}"}

# Known aliases for city matching
CITY_ALIASES = {
    "nyc": "New York",
    "la": "Los Angeles",
    "sf": "San Francisco",
    "nola": "New Orleans",
    "cdmx": "Mexico City",
    "bcn": "Barcelona",
    "ldn": "London",
    "berlin": "Berlin",
    "naples": "napoli",
    "rome": "roma",
    "vienna": "wien",
    "valencia": "valència",
    "tel aviv": "tel aviv-yafo",
    "milan": "milano",
    "florence": "firenze",
    "lisbon": "lisboa",
    "mexico city": "ciudad de méxico",
    "san sebastián": "donostia",
    "brussels": "bruxelles",
    "geneva": "genève",
    "montreal": "montréal",
    "reykjavik": "reykjavík",
    "delhi": "new delhi",
    "boulogne": "boulogne-sur-mer",
    "athens": "athina",
    "cabo": "san josé del cabo",
    "gothenburg": "göteborg",
    "rhodes": "rodos"
}

# Known recommenders and their master Instagram URLs
RECOMMENDER_MAP = {
    "Eric Wareheim": "https://www.instagram.com/ericwareheim",
    "Alice Gao": "https://www.instagram.com/alice_gao"
}

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return text.strip().lower()

def clean_list(raw):
    if not isinstance(raw, str):
        return []
    cleaned = raw.strip("[] ").replace("'", "").replace('"', "")
    return [x.strip().lower() for x in cleaned.split(",") if x.strip()]

def load_airtable_data():
    records = []
    offset = None

    while True:
        params = {"offset": offset} if offset else {}
        response = requests.get(AIRTABLE_URL, headers=HEADERS, params=params)
        data = response.json()
        if "records" not in data:
            print("❌ Airtable API ERROR:")
            print(json.dumps(data, indent=2))
            raise Exception("Airtable API request failed")
        records.extend(data["records"])
        offset = data.get("offset")
        if not offset:
            break

    rows = []
    for record in records:
        fields = record.get("fields", {})
        fields["id"] = record["id"]
        rows.append(fields)
    return pd.DataFrame(rows)

def format_recommender_reference(name_field, link_field):
    if isinstance(name_field, str) and name_field.strip():
        name = name_field.strip()
        link = link_field or RECOMMENDER_MAP.get(name)
        if link:
            return f'This one was recommended by friend of Attaché <a href="{link}" target="_blank">{name}</a>.'
        else:
            return f"This one was recommended by friend of Attaché {name}."
    return None

@app.route("/query", methods=["POST"])
def query():
    try:
        body = request.get_json()
        print("🟢 Received query:", body.get("query", ""))
        print("----- GPT FILTERS -----")
        print(json.dumps(body, indent=2))

        filters = {
            "city": normalize_text(body.get("city", "")),
            "category": [normalize_text(x) for x in body.get("category", [])],
            "tags": [normalize_text(x) for x in body.get("tags", [])],
            "type": [normalize_text(x) for x in body.get("type", [])],
        }

        if filters["city"] in CITY_ALIASES:
            filters["city"] = CITY_ALIASES[filters["city"]]

        df = load_airtable_data()
        print(f"----- Airtable rows: {len(df)} -----")

        results = []
        for _, row in df.iterrows():
            city_match = normalize_text(row.get("City")) == filters["city"]
            if not city_match:
                continue

            tag_match = any(t in clean_list(row.get("Tags")) for t in filters["tags"]) if filters["tags"] else True
            type_match = any(t in clean_list(row.get("Type")) for t in filters["type"]) if filters["type"] else True
            category_match = any(c in clean_list(row.get("Category")) for c in filters["category"]) if filters["category"] else True

            match = tag_match and type_match and category_match
            if match:
                results.append(row.to_dict())

        print(f"✅ Returning {len(results)} results after filtering.")

        for r in results:
            r["Recommended Reference"] = format_recommender_reference(
                r.get("Recommended By Name", ""), r.get("Recommended By Link", "")
            )

        return jsonify({"results": results})
    except Exception as e:
        print(f"❌ ERROR during query: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/places", methods=["GET"])
def get_places():
    try:
        category = normalize_text(request.args.get("category", ""))
        city = normalize_text(request.args.get("city", ""))

        if city in CITY_ALIASES:
            city = CITY_ALIASES[city]

        df = load_airtable_data()
        print(f"🔍 Filtering Airtable for /api/places — Category: {category}, City: {city}")

        filtered = df[
            df["Category"].apply(lambda c: category in clean_list(c)) &
            df["City"].apply(lambda c: normalize_text(c) == city)
        ]

        results = []
        for _, row in filtered.iterrows():
            results.append({
                "name": row.get("Name"),
                "address": row.get("Formatted Address"),
                "image": row.get("Image", [None])[0] if isinstance(row.get("Image"), list) else None,
                "rating": row.get("Rating"),
                "reviewCount": row.get("Number of Reviews"),
                "type": row.get("Type"),
                "price": row.get("Price"),
                "tags": clean_list(row.get("Tags")),
                "note": row.get("Attaché Note"),
                "recommendedBy": row.get("Recommended By Name"),
                "mapsUrl": row.get("Google Maps URL")
            })

        return jsonify({"results": results})
    except Exception as e:
        print(f"❌ ERROR in /api/places: {e}")
        return jsonify({"error": "Failed to fetch data"}), 500

@app.route("/")
def index():
    return "API is live."

if __name__ == "__main__":
    app.run(debug=True)
