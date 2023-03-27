# -*- coding: utf-8 -*-
import asyncio
import random
import src.blivedm as blivedm

async def listen(room_id, normal_danmu_queue, pay_danmu_queue):
    await run_single_client(room_id, normal_danmu_queue, pay_danmu_queue)


async def run_single_client(room_id, normal_danmu_queue, pay_danmu_queue):
    """
    演示监听一个直播间
    """
    # 如果SSL验证失败就把ssl设为False，B站真的有过忘续证书的情况
    client = blivedm.BLiveClient(room_id, ssl=True)
    handler = MyHandler(normal_danmu_queue, pay_danmu_queue)
    client.add_handler(handler)

    client.start()
    try:
        await client.join()
    finally:
        await client.stop_and_close()


class MyHandler(blivedm.BaseHandler):
    def __init__(self, normal_danmu_queue, pay_danmu_queue):
        super().__init__()
        self.normal_danmu = normal_danmu_queue
        self.pay_danmu = pay_danmu_queue

    async def _on_danmaku(self, client: blivedm.BLiveClient, message: blivedm.DanmakuMessage):
        # print(f'[{client.room_id}] {message.uname}：{message.msg}')
        if self.normal_danmu.qsize() < 3:
            self.normal_danmu.put((message.uname, message.msg))

    async def _on_super_chat(self, client: blivedm.BLiveClient, message: blivedm.SuperChatMessage):
        # print(f'[{client.room_id}] 醒目留言 ¥{message.price} {message.uname}：{message.message}')
        if not self.pay_danmu.empty():
            self.pay_danmu.put((message.uname, message.message))
