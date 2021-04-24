import screeninfo

def get_monitor_dict():
    monitor_list = screeninfo.get_monitors()
    monitor_dict = {}
    for item in monitor_list:
        monitor_dict[item.name] = item
    return monitor_dict
