# zero_trust_security.py
"""
零信任安全架构核心算法实现
包含：多维度信任评分计算、隔离森林异常检测、信任阈值自适应调整
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# ==================== 信任评分函数模块 ====================

@dataclass
class TrustFactors:
    """信任评估四维度因子"""
    identity_score: float  # 身份可信度 S_identity (0-100)
    device_score: float  # 设备健康度 S_device (0-100)
    behavior_score: float  # 行为一致性 S_behavior (0-100)
    context_score: float  # 环境安全度 S_context (0-100)


class TrustScoreCalculator:
    """
    信任评分计算器
    实现公式: TrustScore = w1*S_identity + w2*S_device + w3*S_behavior + w4*S_context
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        初始化权重配置
        默认权重: 身份0.4, 设备0.2, 行为0.3, 环境0.1
        """
        self.weights = weights or {
            'identity': 0.4,
            'device': 0.2,
            'behavior': 0.3,
            'context': 0.1
        }
        # 验证权重和为1
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "权重之和必须为1"

    def calculate_identity_score(self,
                                 mfa_enabled: bool,
                                 password_age_days: int,
                                 has_security_incident: bool,
                                 auth_method: str = "mfa") -> float:
        """
        计算身份可信度 S_identity
        :param mfa_enabled: 是否启用MFA
        :param password_age_days: 密码使用天数
        :param has_security_incident: 是否有安全事件记录
        :param auth_method: 认证方式 (mfa/password)
        """
        score = 100.0

        # MFA加成
        if not mfa_enabled:
            score -= 30
        elif auth_method == "mfa":
            score += 5

        # 密码年龄惩罚
        if password_age_days > 90:
            score -= 20
        elif password_age_days > 60:
            score -= 10
        elif password_age_days > 30:
            score -= 5

        # 安全事件惩罚
        if has_security_incident:
            score -= 40

        return max(0, min(100, score))

    def calculate_device_score(self,
                               is_compliant: bool,
                               patch_level: str,
                               is_jailbroken: bool,
                               has_malware: bool) -> float:
        """
        计算设备健康度 S_device
        :param is_compliant: 是否符合安全基线
        :param patch_level: 补丁状态 (latest/medium/outdated)
        :param is_jailbroken: 是否越狱/root
        :param has_malware: 是否检测到恶意软件
        """
        score = 100.0

        if not is_compliant:
            score -= 30

        if is_jailbroken:
            score -= 50

        if has_malware:
            score -= 60

        if patch_level == "outdated":
            score -= 25
        elif patch_level == "medium":
            score -= 10

        return max(0, min(100, score))

    def calculate_behavior_score(self,
                                 anomaly_score: float,
                                 deviation_level: str = "normal") -> float:
        """
        计算行为一致性 S_behavior
        :param anomaly_score: 隔离森林异常分数 (0-1)
        :param deviation_level: 偏离等级 (normal/slight/severe)
        """
        # 基于异常分数计算
        behavior_score = 100 * (1 - anomaly_score)

        # 基于偏离等级调整
        if deviation_level == "slight":
            behavior_score -= 15
        elif deviation_level == "severe":
            behavior_score -= 40

        return max(0, min(100, behavior_score))

    def calculate_context_score(self,
                                ip_reputation: str = "trusted",
                                is_work_hours: bool = True,
                                location_match: bool = True) -> float:
        """
        计算环境安全度 S_context
        :param ip_reputation: IP信誉 (trusted/unknown/malicious)
        :param is_work_hours: 是否工作时间访问
        :param location_match: 地理位置是否匹配历史常驻地
        """
        score = 100.0

        if ip_reputation == "unknown":
            score -= 30
        elif ip_reputation == "malicious":
            score -= 70

        if not is_work_hours:
            score -= 20

        if not location_match:
            score -= 25

        return max(0, min(100, score))

    def calculate_trust_score(self, factors: TrustFactors) -> float:
        """
        计算综合信任评分
        TrustScore = w1*S_identity + w2*S_device + w3*S_behavior + w4*S_context
        """
        trust_score = (
                self.weights['identity'] * factors.identity_score +
                self.weights['device'] * factors.device_score +
                self.weights['behavior'] * factors.behavior_score +
                self.weights['context'] * factors.context_score
        )
        return round(trust_score, 2)

    def get_decision(self, trust_score: float) -> Tuple[str, str]:
        """
        根据信任评分做出决策
        :param trust_score: 信任评分 (0-100)
        :return: (决策结果, 决策描述)
        """
        if trust_score >= 70:
            return ("ALLOW", "允许访问")
        elif trust_score >= 40:
            return ("CHALLENGE", "触发MFA二次验证")
        else:
            return ("DENY", "拒绝访问并终止会话")


# ==================== 隔离森林异常检测模块 ====================

class BehaviorFeatureExtractor:
    """
    行为特征提取器
    提取六类行为特征: 身份特征、时空特征、操作特征、序列特征、资源特征、失败特征
    """

    def __init__(self):
        self.scaler = StandardScaler()
        # 兼容不同版本的 scikit-learn
        try:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            # 旧版本使用 sparse 参数
            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.is_fitted = False

    def extract_features(self, behavior_log: pd.DataFrame) -> np.ndarray:
        """
        从行为日志中提取特征向量
        """
        features = []

        for _, row in behavior_log.iterrows():
            feature_vector = []

            # 1. 身份特征
            feature_vector.extend(self._extract_identity_features(row))

            # 2. 时空特征
            feature_vector.extend(self._extract_spatial_temporal_features(row))

            # 3. 操作特征
            feature_vector.extend(self._extract_operation_features(row))

            # 4. 序列特征
            feature_vector.extend(self._extract_sequence_features(row))

            # 5. 资源特征
            feature_vector.extend(self._extract_resource_features(row))

            # 6. 失败特征
            feature_vector.extend(self._extract_failure_features(row))

            features.append(feature_vector)

        return np.array(features)

    def _extract_identity_features(self, row) -> List[float]:
        """提取身份特征：用户ID哈希、角色编码、认证方式编码"""
        features = []
        # 简化处理：使用角色和认证方式的哈希
        role = row.get('role', 'user')
        role_hash = hash(role) % 100 / 100.0
        auth_method = row.get('auth_method', 'password')
        auth_hash = 1.0 if auth_method == 'mfa' else 0.0
        features.extend([role_hash, auth_hash])
        return features

    def _extract_spatial_temporal_features(self, row) -> List[float]:
        """提取时空特征：登录时间、地理位置合理性"""
        features = []
        # 登录时间（小时归一化）
        login_hour = row.get('login_hour', 12)
        login_hour_normalized = login_hour / 24.0
        # 地理位置合理性
        location_match = 1.0 if row.get('location_match', True) else 0.0
        features.extend([login_hour_normalized, location_match])
        return features

    def _extract_operation_features(self, row) -> List[float]:
        """提取操作特征：请求频率、数据包大小"""
        features = []
        request_rate = row.get('request_rate', 0)
        request_rate_normalized = min(request_rate / 100, 1.0)
        packet_size = row.get('packet_size', 0)
        packet_size_normalized = min(packet_size / (1024 * 1024), 1.0)
        features.extend([request_rate_normalized, packet_size_normalized])
        return features

    def _extract_sequence_features(self, row) -> List[float]:
        """提取序列特征：操作序列模式编码"""
        # 简化处理：基于操作类型编码
        operation_type = row.get('operation_type', 'read')
        op_map = {'read': 0.1, 'write': 0.3, 'delete': 0.7, 'export': 0.9}
        features = [op_map.get(operation_type, 0.2)]
        return features

    def _extract_resource_features(self, row) -> List[float]:
        """提取资源特征：资源敏感度"""
        sensitivity = row.get('resource_sensitivity', 'low')
        sens_map = {'low': 0.2, 'medium': 0.5, 'high': 0.9}
        features = [sens_map.get(sensitivity, 0.3)]
        return features

    def _extract_failure_features(self, row) -> List[float]:
        """提取失败特征：失败次数"""
        fail_count = row.get('fail_count', 0)
        fail_count_normalized = min(fail_count / 10, 1.0)
        features = [fail_count_normalized]
        return features


class IsolationForestDetector:
    """
    基于隔离森林的异常行为检测器
    实现公式: s(x,n) = 2^(-E(h(x))/c(n))
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_samples: int = 256,
                 max_depth: int = 8,
                 contamination: float = 0.05):
        """
        初始化隔离森林检测器
        :param n_estimators: 隔离树数量
        :param max_samples: 子采样大小
        :param max_depth: 树的最大深度
        :param contamination: 数据集中异常比例
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.contamination = contamination

        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            contamination=contamination,
            random_state=42
        )
        self.feature_extractor = BehaviorFeatureExtractor()
        self.anomaly_threshold = None

    def fit(self, behavior_logs: pd.DataFrame):
        """
        训练隔离森林模型
        :param behavior_logs: 正常行为日志数据
        """
        # 提取特征
        features = self.feature_extractor.extract_features(behavior_logs)

        # 训练模型
        self.model.fit(features)

        # 计算异常阈值（使用95分位数）
        anomaly_scores = -self.model.score_samples(features)
        self.anomaly_threshold = np.percentile(anomaly_scores, 95)

        print(f"模型训练完成 - 异常阈值: {self.anomaly_threshold:.4f}")
        return self

    def detect(self, behavior_logs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测异常行为
        :return: (异常标签, 异常分数)
        """
        features = self.feature_extractor.extract_features(behavior_logs)

        # 预测标签 (1: 正常, -1: 异常)
        labels = self.model.predict(features)

        # 计算异常分数 (归一化到0-1)
        anomaly_scores_raw = -self.model.score_samples(features)
        # 使用sigmoid归一化
        anomaly_scores_normalized = 1 / (1 + np.exp(-anomaly_scores_raw))

        # 转换标签格式
        is_anomaly = (labels == -1).astype(int)

        return is_anomaly, anomaly_scores_normalized

    def evaluate(self, test_logs: pd.DataFrame, true_labels: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        """
        predicted_labels, anomaly_scores = self.detect(test_logs)

        metrics = {
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'recall': recall_score(true_labels, predicted_labels),
            'f1_score': f1_score(true_labels, predicted_labels),
            'auc': roc_auc_score(true_labels, anomaly_scores)
        }

        return metrics


# ==================== 信任阈值自适应调整模块 ====================

class AdaptiveThresholdManager:
    """
    信任阈值自适应调整管理器
    实现公式: Threshold_dynamic = Base_threshold + α*Risk_resource - β*History_score
    """

    def __init__(self,
                 base_threshold: float = 70,
                 alpha: float = 3,
                 beta: float = 0.2):
        """
        初始化阈值管理器
        :param base_threshold: 基础阈值
        :param alpha: 资源敏感度系数
        :param beta: 历史信任系数
        """
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.beta = beta

    def calculate_dynamic_threshold(self,
                                    resource_sensitivity: float,
                                    history_score: float) -> float:
        """
        计算动态阈值
        :param resource_sensitivity: 资源安全敏感度 (0-10)
        :param history_score: 历史信任评分均值 (0-100)
        """
        dynamic_threshold = (
                self.base_threshold +
                self.alpha * resource_sensitivity -
                self.beta * history_score
        )
        # 限制阈值范围 [0, 100]
        return max(0, min(100, dynamic_threshold))

    def make_adaptive_decision(self,
                               trust_score: float,
                               resource_sensitivity: float,
                               history_score: float) -> Tuple[str, str]:
        """
        基于自适应阈值的决策
        """
        dynamic_threshold = self.calculate_dynamic_threshold(
            resource_sensitivity, history_score
        )

        if trust_score >= dynamic_threshold:
            return ("ALLOW", f"允许访问 (阈值: {dynamic_threshold:.1f})")
        elif trust_score >= dynamic_threshold * 0.6:
            return ("CHALLENGE", f"触发MFA验证 (阈值: {dynamic_threshold:.1f})")
        else:
            return ("DENY", f"拒绝访问 (阈值: {dynamic_threshold:.1f})")


# ==================== 完整零信任评估引擎 ====================

class ZeroTrustEvaluator:
    """
    零信任综合评估引擎
    整合信任评分计算、异常检测、自适应决策
    """

    def __init__(self):
        self.trust_calculator = TrustScoreCalculator()
        self.anomaly_detector = IsolationForestDetector()
        self.threshold_manager = AdaptiveThresholdManager()
        self.is_trained = False

    def train_anomaly_detector(self, normal_behavior_logs: pd.DataFrame):
        """
        训练异常检测器
        """
        self.anomaly_detector.fit(normal_behavior_logs)
        self.is_trained = True
        print("异常检测器训练完成")

    def evaluate_access_request(self,
                                request_data: Dict,
                                behavior_log: pd.DataFrame = None) -> Dict:
        """
        评估访问请求
        :param request_data: 请求数据，包含身份、设备、资源等信息
        :param behavior_log: 用户行为日志（用于异常检测）
        :return: 评估结果字典
        """
        # 1. 计算身份可信度
        identity_score = self.trust_calculator.calculate_identity_score(
            mfa_enabled=request_data.get('mfa_enabled', False),
            password_age_days=request_data.get('password_age_days', 0),
            has_security_incident=request_data.get('has_security_incident', False),
            auth_method=request_data.get('auth_method', 'password')
        )

        # 2. 计算设备健康度
        device_score = self.trust_calculator.calculate_device_score(
            is_compliant=request_data.get('is_compliant', True),
            patch_level=request_data.get('patch_level', 'latest'),
            is_jailbroken=request_data.get('is_jailbroken', False),
            has_malware=request_data.get('has_malware', False)
        )

        # 3. 异常行为检测
        anomaly_score = 0.0
        if self.is_trained and behavior_log is not None:
            _, anomaly_scores = self.anomaly_detector.detect(behavior_log)
            anomaly_score = anomaly_scores[-1] if len(anomaly_scores) > 0 else 0.0

        # 4. 计算行为一致性
        deviation_level = request_data.get('deviation_level', 'normal')
        behavior_score = self.trust_calculator.calculate_behavior_score(
            anomaly_score, deviation_level
        )

        # 5. 计算环境安全度
        context_score = self.trust_calculator.calculate_context_score(
            ip_reputation=request_data.get('ip_reputation', 'trusted'),
            is_work_hours=request_data.get('is_work_hours', True),
            location_match=request_data.get('location_match', True)
        )

        # 6. 计算综合信任评分
        factors = TrustFactors(
            identity_score=identity_score,
            device_score=device_score,
            behavior_score=behavior_score,
            context_score=context_score
        )
        trust_score = self.trust_calculator.calculate_trust_score(factors)

        # 7. 基础决策
        base_decision, base_desc = self.trust_calculator.get_decision(trust_score)

        # 8. 自适应决策（考虑资源敏感度和历史信任）
        resource_sensitivity = request_data.get('resource_sensitivity', 0)
        history_score = request_data.get('history_score', 60)
        adaptive_decision, adaptive_desc = self.threshold_manager.make_adaptive_decision(
            trust_score, resource_sensitivity, history_score
        )

        return {
            'trust_score': trust_score,
            'factors': {
                'identity_score': identity_score,
                'device_score': device_score,
                'behavior_score': behavior_score,
                'context_score': context_score,
                'anomaly_score': anomaly_score
            },
            'base_decision': base_decision,
            'base_description': base_desc,
            'adaptive_decision': adaptive_decision,
            'adaptive_description': adaptive_desc,
            'resource_sensitivity': resource_sensitivity
        }


# ==================== 模拟数据生成与测试 ====================

def generate_synthetic_data(n_normal: int = 5000, n_anomaly: int = 500):
    """
    生成模拟的行为日志数据用于测试
    """
    np.random.seed(42)

    # 正常行为数据
    normal_data = {
        'role': np.random.choice(['admin', 'user', 'viewer'], n_normal),
        'auth_method': np.random.choice(['mfa', 'password'], n_normal, p=[0.8, 0.2]),
        'login_hour': np.random.normal(14, 3, n_normal).clip(0, 23),
        'location_match': np.random.choice([True, False], n_normal, p=[0.95, 0.05]),
        'request_rate': np.random.exponential(5, n_normal).clip(0, 50),
        'packet_size': np.random.exponential(1024, n_normal).clip(0, 1024 * 500),
        'operation_type': np.random.choice(['read', 'write', 'delete', 'export'],
                                           n_normal, p=[0.7, 0.2, 0.05, 0.05]),
        'resource_sensitivity': np.random.choice(['low', 'medium', 'high'],
                                                 n_normal, p=[0.6, 0.3, 0.1]),
        'fail_count': np.random.poisson(0.1, n_normal).clip(0, 5)
    }

    # 异常行为数据
    anomaly_data = {
        'role': np.random.choice(['admin', 'user', 'viewer'], n_anomaly, p=[0.5, 0.4, 0.1]),
        'auth_method': np.random.choice(['mfa', 'password'], n_anomaly, p=[0.2, 0.8]),
        'login_hour': np.random.uniform(0, 24, n_anomaly),
        'location_match': np.random.choice([True, False], n_anomaly, p=[0.3, 0.7]),
        'request_rate': np.random.exponential(30, n_anomaly).clip(0, 200),
        'packet_size': np.random.exponential(5120, n_anomaly).clip(0, 1024 * 1024 * 2),
        'operation_type': np.random.choice(['read', 'write', 'delete', 'export'],
                                           n_anomaly, p=[0.2, 0.3, 0.3, 0.2]),
        'resource_sensitivity': np.random.choice(['low', 'medium', 'high'],
                                                 n_anomaly, p=[0.2, 0.3, 0.5]),
        'fail_count': np.random.poisson(3, n_anomaly).clip(0, 10)
    }

    normal_df = pd.DataFrame(normal_data)
    anomaly_df = pd.DataFrame(anomaly_data)

    return normal_df, anomaly_df


def run_experiment():
    """
    运行完整实验，验证论文中的指标
    """
    print("=" * 60)
    print("零信任安全架构核心算法实验")
    print("=" * 60)

    # 1. 生成模拟数据
    print("\n[1] 生成模拟行为日志数据...")
    normal_logs, anomaly_logs = generate_synthetic_data(5000, 500)
    print(f"    正常样本数: {len(normal_logs)}, 异常样本数: {len(anomaly_logs)}")

    # 2. 训练异常检测器
    print("\n[2] 训练隔离森林异常检测器...")
    detector = IsolationForestDetector()
    detector.fit(normal_logs)

    # 3. 评估异常检测性能
    print("\n[3] 评估异常检测模型性能...")
    test_logs = pd.concat([normal_logs.sample(1000), anomaly_logs], ignore_index=True)
    true_labels = np.array([0] * 1000 + [1] * len(anomaly_logs))

    metrics = detector.evaluate(test_logs, true_labels)
    print(f"    准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"    召回率 (Recall): {metrics['recall']:.4f}")
    print(f"    F1-Score: {metrics['f1_score']:.4f}")
    print(f"    AUC: {metrics['auc']:.4f}")

    # 验证论文指标
    print("\n    论文指标对比:")
    print(f"    论文准确率: 94.2%, 实验准确率: {metrics['accuracy'] * 100:.1f}%")
    print(f"    论文召回率: 91.5%, 实验召回率: {metrics['recall'] * 100:.1f}%")
    print(f"    论文F1-Score: 0.928, 实验F1-Score: {metrics['f1_score']:.3f}")
    print(f"    论文AUC: 0.962, 实验AUC: {metrics['auc']:.3f}")

    # 4. 测试信任评分计算
    print("\n[4] 测试信任评分计算...")
    evaluator = ZeroTrustEvaluator()

    # 正常访问请求
    normal_request = {
        'mfa_enabled': True,
        'password_age_days': 15,
        'has_security_incident': False,
        'auth_method': 'mfa',
        'is_compliant': True,
        'patch_level': 'latest',
        'is_jailbroken': False,
        'has_malware': False,
        'deviation_level': 'normal',
        'ip_reputation': 'trusted',
        'is_work_hours': True,
        'location_match': True,
        'resource_sensitivity': 3,
        'history_score': 85
    }

    # 异常访问请求
    anomalous_request = {
        'mfa_enabled': False,
        'password_age_days': 120,
        'has_security_incident': True,
        'auth_method': 'password',
        'is_compliant': False,
        'patch_level': 'outdated',
        'is_jailbroken': True,
        'has_malware': True,
        'deviation_level': 'severe',
        'ip_reputation': 'malicious',
        'is_work_hours': False,
        'location_match': False,
        'resource_sensitivity': 9,
        'history_score': 45
    }

    print("\n    正常访问请求评估:")
    normal_result = evaluator.evaluate_access_request(normal_request)
    print(f"    信任评分: {normal_result['trust_score']}")
    print(f"    各维度分数: {normal_result['factors']}")
    print(f"    基础决策: {normal_result['base_description']}")
    print(f"    自适应决策: {normal_result['adaptive_description']}")

    print("\n    异常访问请求评估:")
    anomalous_result = evaluator.evaluate_access_request(anomalous_request)
    print(f"    信任评分: {anomalous_result['trust_score']}")
    print(f"    各维度分数: {anomalous_result['factors']}")
    print(f"    基础决策: {anomalous_result['base_description']}")
    print(f"    自适应决策: {anomalous_result['adaptive_description']}")

    # 5. 测试动态阈值调整
    print("\n[5] 测试动态阈值自适应调整...")
    threshold_manager = AdaptiveThresholdManager()

    test_cases = [
        (75, 2, 80, "低敏感资源 + 高历史信任"),
        (75, 9, 80, "高敏感资源 + 高历史信任"),
        (55, 9, 50, "高敏感资源 + 低历史信任"),
        (85, 8, 90, "高危资源 + 优秀历史信任"),
    ]

    for trust, sensitivity, history, desc in test_cases:
        threshold = threshold_manager.calculate_dynamic_threshold(sensitivity, history)
        decision, desc_result = threshold_manager.make_adaptive_decision(trust, sensitivity, history)
        print(f"    {desc}: 信任分={trust}, 阈值={threshold:.1f}, 决策={decision}")

    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)

    return metrics


def check_sklearn_version():
    """检查 sklearn 版本"""
    import sklearn
    print(f"scikit-learn 版本: {sklearn.__version__}")


if __name__ == "__main__":
    # 检查版本
    check_sklearn_version()

    # 运行实验
    results = run_experiment()

    # 输出论文公式验证
    print("\n" + "=" * 60)
    print("论文公式实现验证")
    print("=" * 60)
    print("\n[公式3-1] 信任评分函数:")
    print("  TrustScore = 0.4*S_identity + 0.2*S_device + 0.3*S_behavior + 0.1*S_context")
    print("  实现位置: TrustScoreCalculator.calculate_trust_score()")

    print("\n[公式3-2] 隔离森林异常分数:")
    print("  s(x,n) = 2^(-E(h(x))/c(n))")
    print("  实现位置: IsolationForestDetector.detect()")

    print("\n[公式3-3] 动态阈值调整:")
    print("  Threshold_dynamic = Base_threshold + α*Risk_resource - β*History_score")
    print("  实现位置: AdaptiveThresholdManager.calculate_dynamic_threshold()")

    print("\n[公式4-1] 信任评分权重配置:")
    print("  TrustScore = 0.4·S_identity + 0.2·S_device + 0.3·S_behavior + 0.1·S_context")
    print("  实现位置: TrustScoreCalculator.__init__()")