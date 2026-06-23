import requests

def send_wechat_notification():
    # 替换成你自己的 SendKey
    send_key = "SCT359045TapV8Dwo5hF2FAzSRDDKdtkIh"
    url = f"https://sctapi.ftqq.com/{send_key}.send"
    data = {
        "title": "脚本运行完毕",
        "desp": "脚本已经顺利执行结束啦！"
    }
    requests.post(url, data=data, proxies={"http": None, "https": None})

# 主程序最后调用即可
send_wechat_notification()