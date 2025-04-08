import requests

apis = [
    "https://ifconfig.me/ip",
    "https://icanhazip.com/",
    "https://api.ipify.org",
    "http://whatismyip.akamai.com"
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

for api in apis:  
    try:
        response = requests.get(api, headers=headers, timeout=5)
        if response.status_code == 200:
            print(f"[成功] {api} 返回 IP: {response.text.strip()}")
            break
    except Exception as e:
        print(f"[失败] {api} 错误: {str(e)}")
else:
    print("所有 API 均失效！")