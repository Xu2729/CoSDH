import time  
from threading import Lock  

class TimerManager:  
    _instance = None  
    _lock = Lock()  

    def __new__(cls, *args, **kwargs):  
        if not cls._instance:  
            with cls._lock:  
                if not cls._instance:  
                    cls._instance = super().__new__(cls)  
                    cls._instance._timers = {}  
        return cls._instance  

    def start_timer(self, label):  
        """启动某个标签的计时器"""  
        if label not in self._timers:  
            self._timers[label] = {'start': None, 'elapsed': 0, 'start_count': 0}  
        if self._timers[label]['start'] is None:  
            self._timers[label]['start'] = time.time()  
            self._timers[label]['start_count'] += 1  # 增加启动次数  
        else:  
            raise RuntimeError(f"Timer with label '{label}' is already running.")  

    def stop_timer(self, label):  
        """停止某个标签的计时器"""  
        if label in self._timers and self._timers[label]['start'] is not None:  
            elapsed = time.time() - self._timers[label]['start']  
            self._timers[label]['elapsed'] += elapsed  
            self._timers[label]['start'] = None  
        else:  
            raise RuntimeError(f"Timer with label '{label}' is not running.")  

    def reset_timer(self, label):  
        """重置某个标签的计时器"""  
        if label in self._timers:  
            self._timers[label] = {'start': None, 'elapsed': 0, 'start_count': 0}  
        else:  
            raise RuntimeError(f"Timer with label '{label}' does not exist.")  

    def get_elapsed_time(self, label):  
        """获取某个标签的累计计时时间"""  
        if label in self._timers:  
            if self._timers[label]['start'] is not None:  
                # 如果计时器正在运行，计算当前累计时间  
                current_elapsed = time.time() - self._timers[label]['start']  
                return self._timers[label]['elapsed'] + current_elapsed  
            return self._timers[label]['elapsed']  
        else:  
            raise RuntimeError(f"Timer with label '{label}' does not exist.")  

    def get_start_count(self, label):  
        """获取某个标签计时器的启动次数"""  
        if label in self._timers:  
            return self._timers[label]['start_count']  
        else:  
            raise RuntimeError(f"Timer with label '{label}' does not exist.")  

    def list_timers(self):  
        """列出所有计时器及其累计时间和启动次数"""  
        result = {}  
        for label, data in self._timers.items():  
            elapsed_time = self.get_elapsed_time(label)  
            start_count = data['start_count']  
            result[label] = {'elapsed_time': elapsed_time, 'start_count': start_count}  
        return result  


# # 示例用法  
# if __name__ == "__main__":  
#     manager = TimerManager()  

#     # 启动计时器  
#     manager.start_timer("task1")  
#     time.sleep(1)  # 模拟任务1运行1秒  
#     manager.stop_timer("task1")  

#     manager.start_timer("task1")  # 再次启动任务1  
#     time.sleep(1)  # 模拟任务1运行1秒  
#     manager.stop_timer("task1")  

#     manager.start_timer("task2")  
#     time.sleep(2)  # 模拟任务2运行2秒  
#     manager.stop_timer("task2")  

#     # 获取计时结果  
#     print("Task1 elapsed time:", manager.get_elapsed_time("task1"))  # 约2秒  
#     print("Task1 start count:", manager.get_start_count("task1"))  # 2次  

#     print("Task2 elapsed time:", manager.get_elapsed_time("task2"))  # 约2秒  
#     print("Task2 start count:", manager.get_start_count("task2"))  # 1次  

#     # 列出所有计时器  
#     print("All timers:", manager.list_timers())  

#     # 重置计时器  
#     manager.reset_timer("task1")  
#     print("Task1 after reset:", manager.get_elapsed_time("task1"))  # 0秒  
#     print("Task1 start count after reset:", manager.get_start_count("task1"))  # 0次
