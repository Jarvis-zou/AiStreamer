import websocket
import threading
import json
import time

room_id = "21704811"  # 这里填写你要爬取的直播间ID

# 弹幕WebSocket地址
url = f"wss://broadcastlv.chat.bilibili.com/sub"

# 发送心跳包，保持WebSocket连接
def send_heartbeat(ws):
    while True:
        ws.send(b'\x00\x02{"type":2}\x00\x00')
        time.sleep(30)

# 处理弹幕消息
def handle_danmu(data):
    # 将消息转换为JSON格式
    try:
        msg = json.loads(data)
    except:
        return
    # 解析JSON格式消息，获取弹幕内容
    if msg["cmd"] == "DANMU_MSG":
        danmu = msg["info"][1]
        print(f"【{msg['info'][0][3]}】{danmu}")

# 建立WebSocket连接，并监听弹幕消息
def connect_websocket():
    ws = websocket.WebSocketApp(url,
                                on_message=handle_danmu,
                                on_close=lambda ws: print("WebSocket已关闭"),
                                on_error=lambda error: print(f"WebSocket发生错误：{error}"))
    ws.on_open = lambda ws: threading.Thread(target=send_heartbeat, args=(ws,), daemon=True).start()
    ws.run_forever()

# 启动弹幕爬取
if __name__ == '__main__':
    connect_websocket()
