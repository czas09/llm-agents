def get_ip_by_domain_name(domain_name: str):
    """
    根据用户提供的域名查询对应的IP
    """
    if domain_name == "www.zyw.com": 
        observation = "29.246.168.188"
        return {"ip": observation}
    else: 
        return None