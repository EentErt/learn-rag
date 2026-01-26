import os
from dotenv import load_dotenv
from google import genai
import json

def load_gemini():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") 
    client = genai.Client(api_key=api_key)
    return client

def spell_check(text):
    client = load_gemini()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{text}"

If no errors, return the original query.
Corrected:""")
    return response.text

def rewrite(text):
    client = load_gemini()
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{text}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def expand_query(text):
    client = load_gemini()
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{text}"
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def llm_score(doc, query):
    client = load_gemini()
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def llm_score_batch(doc_list, query):
    client = load_gemini()
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{str(doc_list)}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. The list should contain only numbers, and no text. For example:

[75, 12, 34, 2, 1]
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return json.loads(response.text.strip("```").strip("json"))

def llm_evaluate(results, query):
    client = load_gemini()
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return json.loads(response.text.strip("```").strip("json"))