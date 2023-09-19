def get_pm25_forecast(location: str):
    """
    根据用户提供的地点获取PM2.5预报
    """
    if "Nanjing" in location:
        observation = "Good"
        return observation
    else: 
        return None