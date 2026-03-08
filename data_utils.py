import re

# Cleaning strategy:
# - Keep main lexical content (words, digits, basic punctuation) so the embedding model
#   can still distinguish topics like 'space shuttle' vs 'hockey game'.
# - Drop email-style noise (headers/footers/quotes via sklearn) and control chars,
#   which add length but little semantic value for downstream clustering/search.
def clean_text(text: str) -> str:
    """
    Basic cleaning for 20 Newsgroups posts.
    - Lowercase
    - Remove non-printable/control chars
    - Collapse multiple whitespace into single spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove control characters
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Keep common punctuation, letters, numbers
    text = re.sub(r"[^a-z0-9.,!?;:'\"()\\[\\]/_-]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\\s+", " ", text).strip()
    return text
