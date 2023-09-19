def get_location_by_ip_address(ip_address: str):
    """
    根据用户提供的IP查询对应的归属地
    """
    if ip_address == "29.246.168.188": 
        observation = "Nanjing"
        return {"location": observation}
    else: 
        return None