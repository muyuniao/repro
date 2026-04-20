import subprocess
import time
import os
from datetime import datetime

def get_free_gpu():
    try:
        # 获取所有 GPU 的显存占用信息（单位：MB）
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
            encoding='utf-8'
        )
        usages = [int(x.strip()) for x in result.split('\n') if x.strip()]
        
        # 寻找占用极低的空闲显卡（小于 1000 MB 视为无人使用）
        for i, usage in enumerate(usages):
            if usage < 1000:
                return i
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 检查 nvidia-smi 时出错: {e}")
    return None

def main():
    print("=========================================")
    print("⏳ 自动抢卡与训练监控脚本已启动 ⏳")
    print("监控频率：每 10 分钟（600 秒）扫描一次四张显卡状态")
    print("=========================================")
    
    while True:
        free_gpu = get_free_gpu()
        
        if free_gpu is not None:
            print(f"\n🎉 [{datetime.now().strftime('%H:%M:%S')}] 监控到 GPU {free_gpu} 已空闲！准备抢占并启动训练...")
            
            # 注入环境变量，告知 bash 脚本应该把模型挂载到哪张卡
            os.environ["SYS_GPU_ID"] = str(free_gpu)
            
            print("🚀 发射...")
            print("-" * 50)
            
            # 使用 shell 直接执行我们的 bash 脚本
            subprocess.run(["bash", "scripts/finetune_OrdChain_lora.sh"])
            
            print("-" * 50)
            print("训练子进程已结束或已退出，自动抢卡脚本使命完成！")
            break
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 四张卡均被严重占用中，继续潜伏等待...")
        
        # 每隔 600 秒（10分钟）检查一次
        time.sleep(600)

if __name__ == "__main__":
    main()
