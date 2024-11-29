import json
import random
from datetime import datetime, timedelta
from utils.file_process import jsonl_reader

current_date = datetime.now()

def cal_date(n):
    # 计算 n 天后的日期
    future_date = current_date + timedelta(days=n)

    # 输出 年、月、日
    year = future_date.year
    month = future_date.month
    day = future_date.day
    return f"{year}年{month}月{day}日({n}天后)"


def pre_process(person_info):
    cur_loc = person_info["current_location"][0] + ("的" + person_info["current_location"][1] if len(person_info["current_location"])==2 and person_info["current_location"][1] != "" else "")
    schedules = person_info["schedule"]
    # 已经有的工作安排
    driver_schedules = []
    if schedules:
        for i, schedule in enumerate(schedules):
            departure_date = cal_date(schedule["date"])
            arrival_date = cal_date(schedule["date"] + schedule["during"])
            departure = schedule["send_city"][0] + ("的" + schedule["send_city"][1] if len(schedule["send_city"])==2 and person_info["current_location"][1] != "" else "")
            arrival = schedule["dest_city"][0] + ("的" + schedule["dest_city"][1] if len(schedule["dest_city"])==2 and person_info["current_location"][1] != "" else "")
            schedule_single = f"货运出发日期：{departure_date}; 货运到达日期: {arrival_date}; 出发地点: {departure}; 到达地点: {arrival}"
            driver_schedules.append(f"已签订的运单{i+1}\n" + schedule_single)
    driver_schedules = "\n".join(driver_schedules)
    car_type = person_info["car"]["car_type"]
    car_length = person_info["car"]["car_length"]

    routes = []
    familiar_routes = person_info["familiar_route"]
    for item in familiar_routes:
        routes.append(item[0][0] + item[0][1]  + "-" + item[1][0] + item[1][1])
    routes = "\n".join(routes)
    return "\n\n".join(["姓名：秦天柱", f"当前所在地: {cur_loc}", f"{driver_schedules}", f"货车类型: {car_type}" + "\n" + f"货车长度: {car_length}", f"熟悉的路线: \n{routes}"])


def get_person_info(file_in):
    person_data = jsonl_reader(file_in)
    person_info = random.choice(person_data)
    return pre_process(person_info)



if __name__ == "__main__":
    file_in = "/Users/chendongdong/Work/llm/huochebao/driver_agent/resources/drivers.jsonl"
    print(get_person_info(file_in))
