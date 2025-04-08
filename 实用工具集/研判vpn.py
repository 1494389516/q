import requests

def get_public_ip():
    # 使用公开的服务来获取公网IP地址
    ip_lookup_services = [
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://icanhazip.com/"
    ]
    
    for service in ip_lookup_services:
        try:
            response = requests.get(service)
            if response.status_code == 200:
                print("Using service:", service)
                return response.text.strip()
        except requests.RequestException as e:
            print(f"Failed to get IP from {service}: {e}")
    
    print("Failed to retrieve public IP address.")
    return None

if __name__ == "__main__":
    public_ip = get_public_ip()
    if public_ip:
        print(f"Your public IP address is: {public_ip}")