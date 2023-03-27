# -*- coding: utf-8 -*-
import asyncio
import random
import src.blivedm as blivedm

# 直播间ID的取值看直播间URL
TEST_ROOM_IDS = [
    139,
]


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
    # # 演示如何添加自定义回调
    # _CMD_CALLBACK_DICT = blivedm.BaseHandler._CMD_CALLBACK_DICT.copy()
    #
    # # 入场消息回调
    # async def __interact_word_callback(self, client: blivedm.BLiveClient, command: dict):
    #     print(f"[{client.room_id}] INTERACT_WORD: self_type={type(self).__name__}, room_id={client.room_id},"
    #           f" uname={command['data']['uname']}")
    # _CMD_CALLBACK_DICT['INTERACT_WORD'] = __interact_word_callback  # noqa

    # async def _on_heartbeat(self, client: blivedm.BLiveClient, message: blivedm.HeartbeatMessage):
    #     print(f'[{client.room_id}] 当前人气值：{message.popularity}')

    async def _on_danmaku(self, client: blivedm.BLiveClient, message: blivedm.DanmakuMessage):
        print(f'[{client.room_id}] {message.uname}：{message.msg}')
        if self.normal_danmu.qsize() < 3:
            print(f'当前队列长度{self.normal_danmu.qsize()}')
            self.normal_danmu.put({message.uname: message.msg})


    # async def _on_gift(self, client: blivedm.BLiveClient, message: blivedm.GiftMessage):
    #     print(f'[{client.room_id}] {message.uname} 赠送{message.gift_name}x{message.num}'
    #           f' （{message.coin_type}瓜子x{message.total_coin}）')

    # async def _on_buy_guard(self, client: blivedm.BLiveClient, message: blivedm.GuardBuyMessage):
    #     print(f'[{client.room_id}] {message.username} 购买{message.gift_name}')

    async def _on_super_chat(self, client: blivedm.BLiveClient, message: blivedm.SuperChatMessage):
        print(f'[{client.room_id}] 醒目留言 ¥{message.price} {message.uname}：{message.message}')
        if len(self.pay_danmu) <= 10:
            self.pay_danmu[message.uname] = message.message


if __name__ == '__main__':
    asyncio.run(listen(room_id=139))
