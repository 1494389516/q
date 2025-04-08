import nmap
import requests
from bs4 import BeautifulSoup
import argparse
def get_web_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def extract_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

url = "http://example.com"
html_content = get_web_page(url)
if html_content:
    links = extract_links(html_content)
    print("Found links:", links)


def scan_ports(target, scan_type='syn'):
    try:
        nm = nmap.PortScanner()
        arguments = '-sS' if scan_type == 'syn' else '-sT'
        
        print(f"正在扫描 {target} ({scan_type}扫描模式)...")
        nm.scan(target, arguments=arguments)
        
        results = []
        for host in nm.all_hosts():
            for proto in nm[host].all_protocols():
                ports = nm[host][proto].keys()
                for port in ports:
                    service = nm[host][proto][port]['name']
                    results.append(f"{host} 的 {proto.upper()}端口 {port} 开放 - {service}")
        return results
    except nmap.PortScannerError as e:
        print(f"扫描错误: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='网络工具集')
    parser.add_argument('-t', '--target', required=True, help='扫描目标IP或域名')
    parser.add_argument('-s', '--scan-type', choices=['syn', 'connect'], default='syn', 
                       help='扫描类型: syn(TCP SYN) 或 connect(全连接)')
    args = parser.parse_args()
    
    # 执行端口扫描
    scan_results = scan_ports(args.target, args.scan_type)
    if scan_results:
        print("\n扫描结果：")
        for item in scan_results:
            print(f"[+] {item}")