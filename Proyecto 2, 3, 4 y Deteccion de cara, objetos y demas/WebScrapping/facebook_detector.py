# scrape_fb_group_playwright_fixed.py
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from bs4 import BeautifulSoup
import json
import time

FACEBOOK_EMAIL = ""  # usa cuenta secundaria
FACEBOOK_PASSWORD = ""
GROUP_URL = "https://www.facebook.com/groups/2406893716312968"
OUTPUT = "posts_paro_tec.jsonl"
SCROLL_PAUSES = 3   # segundos entre scrolls
MAX_SCROLLS = 100   # cuántos scrolls hacer

def parse_posts_from_html(html):
    soup = BeautifulSoup(html, "lxml")
    posts = []
    for article in soup.find_all("div", {"role": "article"}):
        text_el = article.find("div", string=True)
        text = text_el.get_text(" ", strip=True) if text_el else ""
        if text.strip():
            posts.append({"text": text})
    return posts

def main():
    seen = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # LOGIN
        print("[INFO] Iniciando sesión...")
        page.goto("https://www.facebook.com/login", timeout=60000)
        page.fill('input[name="email"]', FACEBOOK_EMAIL)
        page.fill('input[name="pass"]', FACEBOOK_PASSWORD)
        page.click('button[name="login"]')
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Ir al grupo
        print(f"[INFO] Entrando al grupo: {GROUP_URL}")
        page.goto(GROUP_URL, timeout=60000)
        page.wait_for_load_state("networkidle")
        time.sleep(5)

        collected = []
        for i in range(MAX_SCROLLS):
            print(f"[SCROLL {i+1}/{MAX_SCROLLS}] bajando página...")
            # Desplazar hacia abajo
            page.evaluate("window.scrollBy(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSES)

            try:
                # Esperar hasta que carguen los artículos
                page.wait_for_selector('div[role="article"]', timeout=15000)
            except PWTimeout:
                print(f"[WARN] Timeout esperando posts (scroll {i+1}), continuando...")
                continue
            except Exception as e:
                print(f"[ERROR] No se pudo obtener artículos: {e}")
                continue

            # Intentar obtener HTML de la página sin que truene
            try:
                html = page.content()
            except Exception as e:
                print(f"[WARN] No se pudo leer el HTML (scroll {i+1}): {e}")
                time.sleep(2)
                continue

            posts = parse_posts_from_html(html)
            new = 0
            with open(OUTPUT, "a", encoding="utf-8") as f:
                for post in posts:
                    key = post.get("text")[:120]
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(json.dumps(post, ensure_ascii=False) + "\n")
                    new += 1
            print(f"[INFO] Scroll {i+1} -> {new} posts nuevos")

        browser.close()
    print("✅ Terminado.")

if __name__ == "__main__":
    main()
