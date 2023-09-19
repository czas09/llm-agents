def get_coordinates_by_location(location: str):
    """
    根据用户提供的归属地查询对应的经纬度
    """
    if location == "Nanjing": 
        observation = "32° North, 118° West"
        return observation
    else: 
        return None