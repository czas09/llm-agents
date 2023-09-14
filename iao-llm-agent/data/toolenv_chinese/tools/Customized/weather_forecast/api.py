def get_weather_forecast(location: str):
    """
    根据用户提供的地点获取天气预报
    """
    if "南京" in location: 
        observation = "晴天"
        return observation
    else: 
        return None