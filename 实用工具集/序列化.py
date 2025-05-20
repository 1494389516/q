import os
import pickle
import base64
import json  # 新增JSON支持
import yaml  # 新增YAML支持

class Serializer:
    """新增的序列化工具类"""
    def __init__(self):
        self.supported_formats = ['pickle', 'json', 'yaml']
    
    def serialize(self, data, format='pickle'):
        """通用序列化方法"""
        if format not in self.supported_formats:  # 修正拼写错误
            raise ValueError(f"Unsupported format: {format}")   #
        
        try:
            if format == 'pickle':
                return pickle.dumps(data)
            elif format == 'json':
                return json.dumps(data).encode()
            elif format == 'yaml':
                return yaml.dump(data).encode()
        except Exception as e:
            print(f"序列化错误: {str(e)}")
            return None
    
    def deserialize(self, data, format='pickle'):
        """通用反序列化方法"""
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        try:
            if format == 'pickle':
                return pickle.loads(data)
            elif format == 'json':
                return json.loads(data.decode())
            elif format == 'yaml':
                return yaml.load(data.decode(), Loader=yaml.FullLoader)
        except Exception as e:
            print(f"反序列化错误: {str(e)}")
            return None
# 示例用法
if __name__ == "__main__":
    serializer = Serializer()
    
    # 初始化日志配置
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
# 获取用户输入数据
try:
    data_input = input("请输入要序列化的数据（JSON格式）: ")
    test_data = json.loads(data_input)
    selected_fmt = input(f"请选择序列化格式（{serializer.supported_formats}）: ")
    if selected_fmt not in serializer.supported_formats:
        logger.error(f"不支持的格式: {selected_fmt}")
        exit(1)
    fmt = selected_fmt
except json.JSONDecodeError:
    logger.error("输入数据非有效JSON格式")
    exit(1)
except ValueError as e:
    logger.error(f"无效格式选择: {e}")
    exit(1)
    
    # 测试不同格式序列化
    for fmt in [selected_fmt]:  # 仅处理用户选择的格式
        serialized = serializer.serialize(test_data, fmt)
        logger.info(f"\n{fmt.upper()} 序列化结果（Base64前100字符）: {base64.b64encode(serialized).decode()[:100]}...")
        deserialized = serializer.deserialize(serialized, fmt)
        if deserialized is not None:
            logger.info(f"反序列化验证成功: {deserialized}")
        else:
            logger.error("反序列化失败")