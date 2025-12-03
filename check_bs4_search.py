import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)

def search_ddg_html(query, max_results=3):
    print(f"Searching DuckDuckGo HTML for: {query}")
    url = "https://html.duckduckgo.com/html/"
    payload = {'q': query}
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    
    try:
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # DDG HTML results are usually in div class="result"
        # Title in a.result__a
        # Snippet in a.result__snippet
        
        for result in soup.find_all('div', class_='result'):
            if len(results) >= max_results:
                break
                
            title_tag = result.find('a', class_='result__a')
            snippet_tag = result.find('a', class_='result__snippet')
            
            if title_tag and snippet_tag:
                title = title_tag.get_text(strip=True)
                snippet = snippet_tag.get_text(strip=True)
                link = title_tag['href']
                
                results.append({
                    'title': title,
                    'body': snippet,
                    'link': link
                })
                
        return results
        
    except Exception as e:
        print(f"Error scraping DDG: {e}")
        return []

if __name__ == "__main__":
    results = search_ddg_html("Open Droids Jetson")
    if results:
        print("\n--- Success! Found Results ---")
        for r in results:
            print(f"Title: {r['title']}")
            print(f"Body: {r['body']}")
            print("-")
    else:
        print("No results found via HTML scraping.")
