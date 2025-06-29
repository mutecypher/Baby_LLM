import psutil
import os
import logging

def log_memory_usage():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logging.info(f"Memory usage: RSS={mem_info.rss / 1024**2:.2f}MB, VMS={mem_info.vms / 1024**2:.2f}MB")
   