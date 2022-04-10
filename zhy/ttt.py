import asyncio
import time
from PyQt5.QtCore import QTimer


async def hello():
    print("Hello world!")
    # 异步调用asyncio.sleep(1):
    await asyncio.sleep(2)
    print("Hello again!")


# 获取EventLoop:
loop = asyncio.get_event_loop()
# 执行coroutine
loop.run_until_complete(hello())
loop.close()

import threading


class Student:
    def __init__(self):
        self.name = 'xiao ming'

    def loop(self):
        for _ in range(20):
            time.sleep(0.1)
            print(self.name)
    @property
    def ppp(self, a='asd'):
        return a

s = Student()
t = threading.Thread(target=s.loop)

t.start()

print('已经 start')
print('刚 close')


def func():
    print('---')


print(...)
print('===')

timer = QTimer()
timer.timeout.connect(func)
timer.start(100)
print('started')
print(timer.isActive())
# t.join()
timer.stop()
print('stopped')
print(timer.isActive())

import instrument

laser1 = getattr(instrument, 'Laser')
print('1', laser1)
laser2 = getattr(instrument, 'Laser')()
print('2', laser2)

print(s.ppp,s.ppp,s.ppp)