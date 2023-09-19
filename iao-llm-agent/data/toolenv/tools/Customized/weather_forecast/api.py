def get_weather_forecast(location: str):
    """
    根据用户提供的地点获取天气预报
    """
    if "Nanjing" in location: 
        observation = "Sunny"
        return observation
    else: 
        return None