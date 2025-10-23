import os, time, requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode, urljoin

BASE = "https://www.antweb.org"
OUTDIR = "ant_images"
os.makedirs(OUTDIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.antweb.org",
}

session = requests.Session()
session.headers.update(headers)

def get_specimens(subfamily, page=1):
    url = f"{BASE}/images.do?{urlencode({'subfamily': subfamily, 'rank': 'subfamily', 'page': page})}"
    r = session.get(url, timeout=30)
    if r.status_code == 403:
        print(f"Bloqueado en {url}")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    return [urljoin(BASE, a["href"]) for a in soup.select("a[href*='specimen.do']")]

def get_images(specimen_url):
    r = session.get(specimen_url, timeout=30)
    if r.status_code == 403:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    return [urljoin(BASE, i["src"]) for i in soup.select("img[src*='/images/']")]

def download(url, folder):
    fname = os.path.join(folder, url.split("/")[-1])
    if os.path.exists(fname): return
    r = session.get(url, stream=True, timeout=60)
    if r.ok:
        with open(fname, "wb") as f:
            for chunk in r.iter_content(1024*32):
                f.write(chunk)

SUBFAMILIES = ["formicinae", "myrmicinae"]

for sub in SUBFAMILIES:
    folder = os.path.join(OUTDIR, sub)
    os.makedirs(folder, exist_ok=True)
    print(f"Descargando {sub}...")
    for page in range(1, 3):  # prueba con pocas páginas primero
        specimens = get_specimens(sub, page)
        print(f" - Página {page}, {len(specimens)} especímenes")
        for s in specimens:
            for img in get_images(s):
                download(img, folder)
                time.sleep(1)
