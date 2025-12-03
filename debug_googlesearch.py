from googlesearch import search
import logging

logging.basicConfig(level=logging.INFO)

print("Testing googlesearch-python directly...")

try:
    # Test 1: Simple search (returns strings)
    print("\n--- Test 1: Simple Search ---")
    results = search("Open Droids Jetson", num_results=3)
    count = 0
    for r in results:
        print(f"Result: {r}")
        count += 1
        if count >= 3: break
    if count == 0: print("No results found.")

    # Test 2: Advanced search (returns objects)
    print("\n--- Test 2: Advanced Search ---")
    results = search("Open Droids Jetson", num_results=3, advanced=True)
    count = 0
    for r in results:
        print(f"Title: {r.title}")
        print(f"Desc: {r.description}")
        print(f"URL: {r.url}")
        print("-")
        count += 1
        if count >= 3: break
    if count == 0: print("No results found.")

except Exception as e:
    print(f"ERROR: {e}")
