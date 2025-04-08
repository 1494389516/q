import urllib.request
import socket
import re
urls = ["www.qq.com","www.baidu.com","www.sina.com"]
def getHtml(url):
    try:
        with urllib.request.urlopen(f"http://{url}") as response:
            html = response.read()
        return html
    except Exception as e:
        return f"Error fetching HTML: {e}"

def get_ip_address(url):
    try:
        ip_address = socket.gethostbyname(url)
        return ip_address
    except socket.gaierror:
        return "无法获取IP地址"

# 示例用法
if __name__ == "__main__":
    for url in urls:
        ip_address = get_ip_address(url)
        print(f"{url} 的 IP 地址是: {ip_address}")