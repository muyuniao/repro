import json

config_path = "/home/duomeitinrfx/users/yunhe/models/llava-v1.5-7b/config.json"

try:
    # 1. 读取由于外部网络限制无法下载的默认配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 2. 强制将在线权重替换为你本地下载好的权重绝对路径
    config['mm_vision_tower'] = "/home/duomeitinrfx/users/yunhe/models/clip-vit-large-patch14-336"
    
    # 3. 覆盖写入
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
        
    print(f"✅ 成功! 已经将 {config_path} 中的 mm_vision_tower 替换为本地绝对路径。")
except Exception as e:
    print(f"❌ 修改失败: {e}")
