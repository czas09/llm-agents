def get_coordinates_by_location(location: str):
    """
    根据用户提供的归属地查询对应的经纬度
    """
    if "南京" in location: 
        observation = "北纬32°，西经118°"
        return observation
    else: 
        return None