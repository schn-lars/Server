from fastapi import APIRouter, Request
import datetime
from fastapi.responses import JSONResponse, FileResponse
import os

utils_api_router = APIRouter(
    prefix="/api/utils"
)
SERPAPI_API_KEY = os.getenv("SERP_API_KEY", "5432")

@utils_api_router.get("/help")
async def get_help(
        language: str = "ENG"
    ):
    try:
        return FileResponse(path="./Mr__Intenso__How_To.pdf" if language == "ENG" else "Mr__Intenso__Hilfe.pdf", status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@utils_api_router.post("/log")
async def log(request: Request):
    body = await request.body()
    with open("log.txt", "a") as f:
        date = datetime.now().strftime("%d.%m.%y %H:%M:%S")
        f.write(f"{date}|{body.decode('utf-8')}" + "\n")
    return JSONResponse(content={}, status_code=200)

# SERP-API
@utils_api_router.get("/apikey")
async def get_apikey():
    try:
        return JSONResponse(content={"apikey": SERPAPI_API_KEY}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@utils_api_router.get("/test")
async def get_test():
    response = {
        "products_page_token": "8P7KinicTZBJsqNGFEXDO_ECrLJEIyAq...",
        "serpapi_products_link": "https://serpapi.com/search.json?...",
        "serpapi_exact_matches_link": "https://serpapi.com/search.json?...",
        "object": "laptop",
        "visual_matches": [
            {
                "image": "http://c.tutti.ch/big/0614252591.jpg",
                "image_height": 430,
                "image_width": 768,
                "link": "https://www.tutti.ch/de/vi/st-gallen/computer-zubehoer/computer/macbook-pro-13-2019-tb-qc-i5-2-4ghz-16gb-512gb/71370093",
                "position": 1,
                "source": "tutti.ch",
                "source_icon": "https://serpapi.com/searches/6842e3a603b1360912090d62/images/c5e251cd5315d035c8d1e308f76c597a9517f041014bd3d2d7b4df3979a003be.png",
                "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOhoNO0hQ...",
                "thumbnail_height": 168,
                "thumbnail_width": 300,
                "title": "MacBook Pro 13\" 2019 TB QC i5 2,4GHz 16GB 512GB im Kanton St. Gallen - tutti.ch"
            },
            {
                "image": "https://preview.redd.it/my-macbook-pro-comes-up-with-an-error-when-i-try-to-connect-v0-e8pcf8fdnqne1.jpeg?width=640&crop=smart&auto=webp&s=3325683f1c1bbbf0dd726efa05215eccae0bc029",
                "image_height": 640,
                "image_width": 853,
                "link": "https://www.reddit.com/r/applehelp/comments/1j7j844/...",
                "position": 2,
                "source": "Reddit",
                "source_icon": "https://serpapi.com/searches/6842e3a603b1360912090d62/images/c5e251cd5315d035d46dc2fbee449e59e3f78dc5977f90f9fe12be5c3ea459e0.png",
                "thumbnail": "https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcR1iKSbi7-nNrgaAvKcVS0VSbWjgb3bFHs_K2cOVFHGyoindGgL",
                "thumbnail_height": 194,
                "thumbnail_width": 259,
                "title": "My MacBook Pro comes up with an error when I try to connect to WiFi : r/applehelp"
            },
            {
                "image": "https://external-preview.redd.it/flashing-pink-before-turning-off-v0-NzY1eWtqczhwcjViMf3lzKkurh-TXBuL8fkUC-9sMG-2nsMiIvElkNOfel9D.png?width=640&crop=smart&format=pjpg&auto=webp&s=ae3dff2f8133da7619a672d472361d3d5c3b0451",
                "image_height": 640,
                "image_width": 1137,
                "link": "https://www.reddit.com/r/macbookrepair/comments/148csy0/...",
                "position": 3,
                "source": "Reddit",
                "source_icon": "https://serpapi.com/searches/6842e3a603b1360912090d62/images/c5e251cd5315d0354aee44d98390809bdf8d0c1325c590c9903220c414a0b0b9.png",
                "thumbnail": "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQRJ7kmVaDtbffyz04nMHLLS_Nd1hDi2JBT2gGlfp-txz0ADsqn",
                "thumbnail_height": 168,
                "thumbnail_width": 299,
                "title": "Flashing Pink Before Turning Off : r/macbookrepair"
            },
            {
                "condition": "Used",
                "image": "https://i.ebayimg.com/images/g/AnQAAOSw1OllqaL4/s-l1200.jpg",
                "image_height": 900,
                "image_width": 1200,
                "link": "https://www.ebay.com/itm/315105573570",
                "position": 18,
                "price": {
                    "currency": "$",
                    "extracted_value": 300,
                    "value": "$300*"
                },
                "rating": "3.9",
                "reviews": 1920,
                "source": "eBay",
                "source_icon": "https://encrypted-tbn0.gstatic.com/favicon-tbn?q=tbn:ANd9GcT5NXsW5qpQoNKtvnEC0sNL88H54opWmBBYIh2gQ3U_SGUU-yc8xV_BfeECVq4HYfwroQsx3k4lpMvjDByZM4KvONyK63j7aI6RPQrDFwRa9lo",
                "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxi-QaJcOVLnezkhk4rSCD4gz52Ze42__klJQTCEa26j3gdiK-",
                "thumbnail_height": 194,
                "thumbnail_width": 259,
                "title": "Apple Macbook Pro 13” (256GB SSD, Intel Core i5 Dual-Core, 3.1GHz, 8GB RAM) 2017 | eBay"
            },
            {
                "image": "https://pisces.bbystatic.com/image2/BestBuy_US/ugc/photos/thumbnail/7500865a3d202ecc6ae59d9f10262d9d.jpg;maxHeight=256;maxWidth=256?format=webp",
                "image_height": 192,
                "image_width": 256,
                "in_stock": 0,
                "link": "https://www.bestbuy.com/site/...",
                "position": 56,
                "price": {
                    "currency": "$",
                    "extracted_value": 670,
                    "value": "$670*"
                },
                "rating": "4.8",
                "reviews": 23286,
                "source": "Best Buy",
                "source_icon": "https://encrypted-tbn1.gstatic.com/favicon-tbn?q=tbn:ANd9GcQRElpIn0_ngPFmOJ5ZdW65M7G4nuI9MycoEeMitUDh35QpZN3krTU2tf03QRaenlaJn4hTIdv1gBd0yWlFCTuep64IeIo7wzb0RKajGmDwhoSbHbc",
                "thumbnail": "https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcR3yQPwHtzZZNcFKmzEaEEeqUMtupmwG8MCsCtB74BkrkWRyBE2",
                "thumbnail_height": 192,
                "thumbnail_width": 256,
                "title": "Geek Squad Certified Refurbished MacBook Air 13.6\" Laptop Apple M2 chip 8GB Memory 256GB SSD Midnight GSRF MLY33LL/A - Best Buy"
            }
        ],
        "search_metadata": {
            "created_at": "2025-06-06 12:48:38 UTC",
            "google_lens_url": "https://lens.google.com/uploadbyurl?url=...",
            "id": "6842e3a603b1360912090d62",
            "json_endpoint": "https://serpapi.com/searches/3943d6e05970c134/6842e3a603b1360912090d62.json",
            "processed_at": "2025-06-06 12:48:38 UTC",
            "raw_html_file": "https://serpapi.com/searches/3943d6e05970c134/6842e3a603b1360912090d62.html",
            "status": "Success",
            "total_time_taken": "6.91"
        },
        "visual_matches_page_token": "NmO4anicTZBJsqNGFEXDO_...",
        "related_content": [
            {
                "link" : "https://www.google.com/search?sca_esv=87b41ab4477c98ab&lns_surface=26&hl=en&q=Apple+MacBook+Air+13-inch+Apple+M1+Chip+7-core+GPU+16GB+256GB+Space+Grey&kgmid=/g/11mw8j71m4&sa=X&ved=2ahUKEwi244u06tyNAxUhiP0HHbARA28Q9_gLKAB6BQjtAhAB",
                "query" : "Apple MacBook Air 13-inch Apple M1 Chip 7-core GPU 16GB 256GB Space Grey",
                "serpapi_link" : "https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&hl=en&kgmid=%2Fg%2F11mw8j71m4&q=Apple+MacBook+Air+13-inch+Apple+M1+Chip+7-core+GPU+16GB+256GB+Space+Grey",
                "thumbnail" : "https://serpapi.com/searches/6842e3a603b1360912090d62/images/5e7b1f0a649ea9f9a3cff9b018b4fe534b8f6aa3a86a3971fd702cd7f178ddc0.jpeg"
            },
            {
                "link" : "https://www.google.com/search?sca_esv=87b41ab4477c98ab&lns_surface=26&hl=en&q=Apple+MacBook+Pro&kgmid=/m/09tzfp&sa=X&ved=2ahUKEwi244u06tyNAxUhiP0HHbARA28Q9_gLKAF6BQjtAhAD",
                "query" : "Apple MacBook Pro",
                "serpapi_link" : "https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&hl=en&kgmid=%2Fm%2F09tzfp&q=Apple+MacBook+Pro",
                "thumbnail" : "https://serpapi.com/searches/6842e3a603b1360912090d62/images/5e7b1f0a649ea9f9a3cff9b018b4fe53baad5b23c6502719ec11b3ffd6b4d134.jpeg",
            },
            {
                "link" : "https://www.google.com/search?sca_esv=87b41ab4477c98ab&lns_surface=26&hl=en&q=MacBook+Pro&kgmid=/g/11t7nmnv1m&sa=X&ved=2ahUKEwi244u06tyNAxUhiP0HHbARA28Q9_gLKAJ6BQjtAhAF",
                "query" : "MacBook Pro",
                "serpapi_link" : "https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&hl=en&kgmid=%2Fg%2F11t7nmnv1m&q=MacBook+Pro",
                "thumbnail" : "https://serpapi.com/searches/6842e3a603b1360912090d62/images/5e7b1f0a649ea9f9a3cff9b018b4fe5317cb68559d45e152a1458cdcbd521002.jpeg"
            }
        ]
    }
    return response