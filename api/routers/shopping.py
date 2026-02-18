from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

shopping_api_router = APIRouter(
    prefix="/api/shopping"
)


@shopping_api_router.post("")
async def get_shopping_items(
        picture: UploadFile = File(...)
    ):
    try:
        print("Starting with shopping")
        response = {
            "shopping_results": [
            {
              "position": 1,
              "title": "Apple - MacBook Pro 14' Laptop - M3 chip - 8GB Memory - 10-core GPU - 512GB SSD ...",
              "link": "https://www.bestbuy.com/site/apple-macbook-pro-14-laptop-m3-chip-8gb-memory-10-core-gpu-512gb-ssd-latest-model-space-gray/6534641.p?skuId=6534641&utm_source=feed",
              "product_link": "https://www.google.com/shopping/product/1?gl=us&prds=pid:6210135998181032295",
              "product_id": "6210135998181032295",
              "serpapi_product_api": "https://serpapi.com/search.json?device=desktop&engine=google_product&gl=us&google_domain=google.com&hl=en&product_id=6210135998181032295",
              "source": "Best Buy",
              "source_icon": "https://encrypted-tbn0.gstatic.com/favicon-tbn?q=tbn%3AANd9GcRJLrYt8ApvztGsW8TSy6-5HL7LwDNwH2emYmRabMUepMDXWE3LqD_Jltucg6NfE5z5MV57q9G1n_VVMyiUtZCVGXOuFlVA6g",
              "price": "$1,449.00",
              "extracted_price": 1449.0,
              "old_price": "$1,599.00",
              "extracted_old_price": 1599.0,
              "rating": 4.8,
              "reviews": 295,
              "extensions": [
                "Mac OS",
                "Octa Core",
                "USB-C",
                "SALE"
              ],
              "badge": "Top Quality Store",
              "thumbnail": "https://encrypted-tbn3.gstatic.com/shopping?q=tbn:ANd9GcQSBzh7sbe1Ya52-vjINxImUABAV6GkZiP1hRjLqtzbuzjrkMqCeOv_fePvs-2-wblGqi2V_Qlr&usqp=CAE",
              "serpapi_thumbnail": "https://serpapi.com/images/url/tyWlMHicu9mdUVJSUGylr5-al1xUWVCSmqJbkpRnrJdeXJJYkpmsl5yfq1-ckV9QkJmXbl9oC5SzcvRLsXRPDgx2qsowL05KNYxMNDXSLcvy9KvwzA11dHIMM3PPjsoMMMwIyvIpLKlKKq3KKsr2LXRO9S-LT0sNKCvWNdItT8pxL8w0CosPzClSKy0uLLB1dnQFAHuuMu4",
              "tag": "SALE",
              "delivery": "Free delivery by Feb 13 & Free 15-day returns",
              "store_rating": 4.6,
              "store_reviews": 358
            },
            {
              "position": 2,
              "title": "Apple - MacBook Pro 14' Laptop - M3 chip - 8GB Memory - 10-core GPU - 512GB SSD ...",
              "link": "https://www.bestbuy.com/site/apple-macbook-pro-14-laptop-m3-chip-8gb-memory-10-core-gpu-512gb-ssd-latest-model-silver/6534640.p?skuId=6534640&utm_source=feed",
              "product_link": "https://www.google.com/shopping/product/1?gl=us&prds=pid:3059131534171723531",
              "product_id": "3059131534171723531",
              "serpapi_product_api": "https://serpapi.com/search.json?device=desktop&engine=google_product&gl=us&google_domain=google.com&hl=en&product_id=3059131534171723531",
              "source": "Best Buy",
              "source_icon": "https://encrypted-tbn0.gstatic.com/favicon-tbn?q=tbn%3AANd9GcRJLrYt8ApvztGsW8TSy6-5HL7LwDNwH2emYmRabMUepMDXWE3LqD_Jltucg6NfE5z5MV57q9G1n_VVMyiUtZCVGXOuFlVA6g",
              "price": "$1,449.00",
              "extracted_price": 1449.0,
              "old_price": "$1,599.00",
              "extracted_old_price": 1599.0,
              "rating": 4.8,
              "reviews": 69,
              "extensions": [
                "Mac OS",
                "Octa Core",
                "USB-C",
                "SALE"
              ],
              "badge": "Top Quality Store",
              "thumbnail": "https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcTopcSSvTfIe-JheQn8HrcAkgHN5yH_toz8fnJudGBf1v3N9cy1i-1tlPqYzeyISyBVdMnnEYI&usqp=CAE",
              "serpapi_thumbnail": "https://serpapi.com/images/url/gtpsuXicDcndCoIwFADgt-lOZUSggoSG-ANJoQReRZ1tKtbZdEdhPkJv2NvUd_t9Pz2RNqHnCYTZahLcoScytzP0oAFcUG_P9ErrAbvjFP0vjCseZNAoDXW9NrIQTtmLK_r5DPHY5dXB5ndSmy-xXHiWSLbuqwAsGxxGr8vUbsIWtU1u_IyYtsVuMZOOTnH6A17VMok",
              "tag": "SALE",
              "delivery": "Free delivery by Feb 13 & Free 15-day returns",
              "store_rating": 4.6,
              "store_reviews": 358
            },
            {
              "position": 3,
              "title": "Apple - MacBook Pro 14' Laptop - M3 Pro chip - 18GB Memory - 18-core GPU - 1TB ...",
              "link": "https://www.bestbuy.com/site/apple-macbook-pro-14-laptop-m3-pro-chip-18gb-memory-18-core-gpu-1tb-ssd-latest-model-space-black/6534624.p?skuId=6534624&utm_source=feed",
              "product_link": "https://www.google.com/shopping/product/1?gl=us&prds=pid:3713303853686963414",
              "product_id": "3713303853686963414",
              "serpapi_product_api": "https://serpapi.com/search.json?device=desktop&engine=google_product&gl=us&google_domain=google.com&hl=en&product_id=3713303853686963414",
              "source": "Best Buy",
              "source_icon": "https://encrypted-tbn0.gstatic.com/favicon-tbn?q=tbn%3AANd9GcRJLrYt8ApvztGsW8TSy6-5HL7LwDNwH2emYmRabMUepMDXWE3LqD_Jltucg6NfE5z5MV57q9G1n_VVMyiUtZCVGXOuFlVA6g",
              "price": "$2,199.00",
              "extracted_price": 2199.0,
              "old_price": "$2,399.00",
              "extracted_old_price": 2399.0,
              "rating": 4.9,
              "reviews": 3813,
              "extensions": [
                "Mac OS",
                "USB-C",
                "HDMI",
                "120Hz",
                "SALE"
              ],
              "badge": "Top Quality Store",
              "thumbnail": "https://encrypted-tbn0.gstatic.com/shopping?q=tbn:ANd9GcSfcqIJ2kTmhAF2t_Pxs9HR2aPI6599y2wNrkwdi4vgnHSdUYuLGg7uBrFily2cibeqpBampas&usqp=CAE",
              "serpapi_thumbnail": "https://serpapi.com/images/url/191f4HicDclLDoIwFADA27gTTOMnJSEGjHyMIUR04YqUFkuDlFf6ELmCN_Q2Otv5fhpEsJ7r1poPM2AtlljplSMtMlTc4X3n2qYHUFrujf8_L8gEjXnx4CY9kfbaNUFEsMzfliYXwvJ0u6F0JlM2tJNQ65fUSSFu9_Ecy90YDpF6zoSrqjYQsg6YXYzWgH8Ijj9AhTKY",
              "tag": "SALE",
              "delivery": "Free delivery by Feb 13 & Free 15-day returns",
              "store_rating": 4.6,
              "store_reviews": 358
            }
          ]
        }

        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        print(f"Error in get_shopping_items: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)