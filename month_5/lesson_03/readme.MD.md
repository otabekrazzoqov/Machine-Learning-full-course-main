# 🕷️ Web Scraping Project

A complete web scraping project using **Selenium** and **BeautifulSoup** in Python — covering element selection, pagination, data extraction, and saving results.

---

## 📦 Installation

```bash
pip install selenium beautifulsoup4 requests pandas openpyxl
```

Also install **ChromeDriver** matching your Chrome version:
- Download from: https://chromedriver.chromium.org/downloads
- Or use: `pip install webdriver-manager`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `Selenium` | Browser automation, dynamic pages (JavaScript) |
| `BeautifulSoup` | Parsing static HTML content |
| `Pandas` | Saving and managing scraped data |
| `Requests` | Fetching raw HTML for BeautifulSoup |

---

## 🔍 When to Use What

| Scenario | Use |
|---|---|
| Page loads with JavaScript | ✅ Selenium |
| Simple static HTML page | ✅ BeautifulSoup |
| Need to click buttons or paginate | ✅ Selenium |
| Fast lightweight scraping | ✅ BeautifulSoup |
| Login, forms, dropdowns | ✅ Selenium |

---

## 🤖 SELENIUM

### Setup

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

driver = webdriver.Chrome(options=chrome_options)
driver.get("https://your-website.com")
```

---

### 🎯 All Methods to Get Elements

#### 1. 🔑 By ID
```python
element = driver.find_element(By.ID, "search")
```

#### 2. 🏷️ By Name
```python
element = driver.find_element(By.NAME, "username")
```

#### 3. 🎨 By Class Name
```python
# Single class only — no spaces allowed
element = driver.find_element(By.CLASS_NAME, "product-price")

# Get ALL elements with this class
elements = driver.find_elements(By.CLASS_NAME, "product-price")
for el in elements:
    print(el.text)
```

#### 4. 🏷️ By Tag Name
```python
links  = driver.find_elements(By.TAG_NAME, "a")
images = driver.find_elements(By.TAG_NAME, "img")
for img in images:
    print(img.get_attribute("src"))
```

#### 5. 🎯 By CSS Selector
```python
el = driver.find_element(By.CSS_SELECTOR, "#price")           # by ID
el = driver.find_element(By.CSS_SELECTOR, ".product-title")   # by class
el = driver.find_element(By.CSS_SELECTOR, "span.a-price")     # tag + class
el = driver.find_element(By.CSS_SELECTOR, "input[type='text']")  # by attribute
el = driver.find_element(By.CSS_SELECTOR, ".class1.class2")   # multiple classes
el = driver.find_element(By.CSS_SELECTOR, "div.container > span.price")  # child
el = driver.find_element(By.CSS_SELECTOR, "ul li:nth-child(3)")  # by index
```

#### 6. 📍 By XPath
```python
el = driver.find_element(By.XPATH, "//*[@id='price']")
el = driver.find_element(By.XPATH, "//span[@class='a-price']")
el = driver.find_element(By.XPATH, "//button[text()='Add to Cart']")
el = driver.find_element(By.XPATH, "//span[contains(text(), 'UZS')]")
el = driver.find_element(By.XPATH, "//*[contains(@class, 'price')]")
el = driver.find_element(By.XPATH, "(//span[@class='price'])[2]")
el = driver.find_element(By.XPATH, "//input[@placeholder='Search']")
```

#### 7. 🔗 By Link Text
```python
element = driver.find_element(By.LINK_TEXT, "Sign In")
element.click()
```

#### 8. 🔗 By Partial Link Text
```python
element = driver.find_element(By.PARTIAL_LINK_TEXT, "Sign")
element.click()
```

---

### 📊 All Methods Comparison

| Method | Best For | Example |
|---|---|---|
| `By.ID` | Unique elements | `id="price"` |
| `By.NAME` | Form inputs | `name="email"` |
| `By.CLASS_NAME` | Styled elements | `class="btn"` |
| `By.TAG_NAME` | All of a type | all `<a>` tags |
| `By.CSS_SELECTOR` | Complex selectors | `.card > .price` |
| `By.XPATH` | Any scenario, text search | `//span[text()='Buy']` |
| `By.LINK_TEXT` | Exact link text | `"Sign In"` |
| `By.PARTIAL_LINK_TEXT` | Partial link text | `"Sign"` |

---

### 🔄 find_element vs find_elements

```python
# find_element → ONE element, raises error if not found
element = driver.find_element(By.CLASS_NAME, "price")

# find_elements → LIST of all matches, returns [] if not found
elements = driver.find_elements(By.CLASS_NAME, "price")
for el in elements:
    print(el.text)
```

---

### ⚡ Always Use WebDriverWait

```python
wait = WebDriverWait(driver, 10)
el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".price")))
print(el.text)
```

---

### 📄 Pagination — Loop Through All Pages

```python
import time

all_data = []

while True:
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "item_car")))

    cards = driver.find_elements(By.CLASS_NAME, "item_car")
    for card in cards:
        try:
            name  = card.find_element(By.CLASS_NAME, "title").text
            price = card.find_element(By.CLASS_NAME, "card_right_info").text
            all_data.append({"name": name, "price": price})
        except:
            continue

    next_btn = driver.find_element(By.CLASS_NAME, "btn-next")
    if next_btn.get_attribute("disabled"):
        break

    next_btn.click()
    time.sleep(2)
```

---

### 💾 Stop & Save Safely

```python
import pandas as pd

def save_data(data, filename="output.csv"):
    pd.DataFrame(data).to_csv(filename, index=False)
    print(f"✅ Saved {len(data)} records to {filename}")

try:
    # your scraping loop here
    pass

except KeyboardInterrupt:
    print("⛔ Stopped by user!")

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    save_data(all_data, "output_final.csv")  # always runs
    driver.quit()
```

---

## 🍲 BEAUTIFULSOUP

### Setup

```python
import requests
from bs4 import BeautifulSoup

url = "https://your-website.com"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
```

---

### 🎯 All Methods to Get Elements

#### 1. By Tag
```python
title    = soup.find("h1")                  # first match
all_tags = soup.find_all("a")              # all matches
for link in all_tags:
    print(link.text, link.get("href"))
```

#### 2. By Class
```python
price = soup.find("span", class_="price")
cards = soup.find_all("div", class_="product-card")
for card in cards:
    print(card.text)
```

#### 3. By ID
```python
element = soup.find("div", id="main-content")
print(element.text)
```

#### 4. By Attribute
```python
el = soup.find("input", {"placeholder": "Search"})
el = soup.find("a", {"href": "/products"})
```

#### 5. By CSS Selector
```python
el  = soup.select_one("div.container > span.price")
els = soup.select("ul.product-list li")
for el in els:
    print(el.text)
```

#### 6. Get Attributes
```python
link = soup.find("a")
print(link.get("href"))

img = soup.find("img")
print(img.get("src"))
print(img.get("alt"))
```

#### 7. Navigate the Tree
```python
parent = soup.find("div", class_="card")
title  = parent.find("h2")         # child element
grand  = parent.parent             # go up
```

---

### 🌐 BeautifulSoup with Pagination

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

all_data = []
base_url = "https://your-website.com/page/"
headers  = {"User-Agent": "Mozilla/5.0"}

for page_num in range(1, 101):
    url      = f"{base_url}{page_num}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"❌ Failed at page {page_num}")
        break

    soup  = BeautifulSoup(response.text, "html.parser")
    items = soup.find_all("div", class_="product-card")

    for item in items:
        try:
            name  = item.find("h2",   class_="title").text.strip()
            price = item.find("span", class_="price").text.strip()
            all_data.append({"name": name, "price": price})
        except:
            continue

    print(f"📄 Page {page_num} — Total: {len(all_data)}")

pd.DataFrame(all_data).to_csv("output.csv", index=False)
print("✅ Done!")
```

---

## 🌐 Best Practice Websites

| Website | Difficulty | Best For |
|---|---|---|
| `books.toscrape.com` | 🟢 Easy | Pagination, prices |
| `quotes.toscrape.com` | 🟢 Easy | Tags, authors |
| `the-internet.herokuapp.com` | 🟡 Medium | Forms, tables |
| `news.ycombinator.com` | 🟡 Medium | News aggregator |
| `wikipedia.org` | 🟡 Medium | Tables, text |
| `amazon.com` | 🔴 Hard | Anti-bot handling |

---

## 📁 Project Structure

```
web-scraping-project/
│
├── selenium/
│   ├── scraper.py
│   └── pagination.py
│
├── beautifulsoup/
│   ├── scraper.py
│   └── parser.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ requirements.txt

```
selenium
beautifulsoup4
requests
pandas
openpyxl
webdriver-manager
```

```bash
pip install -r requirements.txt
```

---

## 👤 Author

**Otabek Razzoqov** · [GitHub](https://github.com/otabekrazzoqov)

---

## 📜 License

MIT License — open source & free to use.
