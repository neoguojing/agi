import random
from collections import defaultdict

class SearchEngineSelector:
    def __init__(self, search_engines):
        self.search_engines = search_engines
        self.success_stats = defaultdict(lambda: {'success': 0, 'total': 0})
        self.default_engines = list(search_engines.keys())
        
    def record_result(self, engine_name, success):
        """记录搜索引擎的使用结果"""
        self.success_stats[engine_name]['total'] += 1
        if success:
            self.success_stats[engine_name]['success'] += 1
    
    def get_success_rate(self, engine_name):
        """获取搜索引擎的成功率"""
        stats = self.success_stats[engine_name]
        if stats['total'] == 0:
            return 0.5  # 默认成功率
        return stats['success'] / stats['total']
    
    def select_engine(self):
        """根据成功率选择搜索引擎"""
        # 计算每个引擎的权重（成功率 + 小随机值避免完全排除低成功率引擎）
        weights = {
            name: self.get_success_rate(name) + random.uniform(0, 0.1)
            for name in self.default_engines
        }
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight == 0:
            return random.choice(self.default_engines)
            
        normalized_weights = {
            name: weight / total_weight
            for name, weight in weights.items()
        }
        
        # 根据权重随机选择
        return random.choices(
            list(normalized_weights.keys()),
            weights=list(normalized_weights.values()),
            k=1
        )
    
    def get_engine(self, name):
        """获取指定名称的搜索引擎实例"""
        return self.search_engines.get(name)