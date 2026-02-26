import json
from collections import defaultdict, Counter

# 1. Load the bigram frequency JSON
# If you `npm install english-bigrams`, you can copy the exported object
# into a file like `english_bigrams.json`, or fetch a JSON version directly.
with open("english_bigrams.json", "r") as f:
    bigrams = json.load(f)  # e.g. {"es": 60365, "in": 51816, ...}[page:1]

# 2. Build mapping: prev_letter -> Counter of next letters
next_by_prev = defaultdict(Counter)

for pair, count in bigrams.items():
    if len(pair) != 2:
        continue  # skip anything unexpected
    prev, nxt = pair[0], pair[1]
    # (Optional) restrict to a–z lowercase
    if not (prev.isalpha() and nxt.isalpha()):
        continue
    prev = prev.lower()
    nxt = nxt.lower()
    next_by_prev[prev][nxt] += count

# 3a. Dictionary: prev -> list of next letters sorted by frequency
sorted_next_letters = {}
for prev, counter in next_by_prev.items():
    letters_sorted = [ch for ch, _ in counter.most_common()]
    sorted_next_letters[prev] = letters_sorted

# 3b. (Optional) Dictionary: prev -> list of (next, probability) pairs
prob_next_letters = {}
for prev, counter in next_by_prev.items():
    total = sum(counter.values())
    prob_list = [(ch, cnt / total) for ch, cnt in counter.most_common()]
    prob_next_letters[prev] = prob_list

# 4. Example usage
print("Top 5 next letters after 't':", sorted_next_letters.get("t", [])[:5])
print("Top 5 with probabilities after 't':", prob_next_letters.get("t", [])[:5])
