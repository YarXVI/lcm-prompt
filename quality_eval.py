"""
LCM v2 答案质量系统性评估框架
量化 LCM 的"聚焦效应"质量提升
"""
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime


@dataclass
class QualityMetrics:
    """质量评估指标"""
    # 聚焦效应指标
    hallucination_count: int = 0  # 幻觉次数（编造不存在的内容）
    focus_accuracy: float = 0.0  # 聚焦准确度（0-1）
    context_utilization: float = 0.0  # 上下文利用率（0-1）
    
    # 效率指标
    token_efficiency: float = 0.0  # Token 效率（有用 token / 总 token）
    rounds_to_converge: int = 0  # 收敛轮次
    avg_response_length: int = 0  # 平均响应长度
    
    # 质量指标
    answer_completeness: float = 0.0  # 答案完整性（0-1）
    citation_accuracy: float = 0.0  # 引用准确度（0-1）
    instruction_following: float = 0.0  # 指令遵循度（0-1）
    
    # 对比指标（vs 传统方案）
    vs_traditional_quality_delta: float = 0.0  # 质量提升（百分点）
    vs_traditional_latency_delta: float = 0.0  # 延迟变化（毫秒）
    vs_traditional_token_delta: float = 0.0  # Token 节省


@dataclass
class EvaluationResult:
    """单次评估结果"""
    task_id: str
    task_type: str  # "code_review", "security_audit", "refactor", etc.
    model: str
    use_lcm: bool
    metrics: QualityMetrics
    raw_output: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0


class QualityEvaluator:
    """
    质量评估器
    系统性评估 LCM 方案的答案质量
    """

    # 评估任务模板
    EVALUATION_TASKS = {
        "code_review": {
            "description": "代码审查任务",
            "criteria": ["发现所有问题", "不遗漏关键缺陷", "不编造不存在的问题"],
        },
        "security_audit": {
            "description": "安全审计任务",
            "criteria": ["发现所有安全漏洞", "不遗漏高危漏洞", "不编造漏洞"],
        },
        "refactor": {
            "description": "重构建议任务",
            "criteria": ["建议合理", "不破坏功能", "考虑边界情况"],
        },
        "architecture_review": {
            "description": "架构审查任务",
            "criteria": ["识别架构问题", "考虑扩展性", "不遗漏关键设计缺陷"],
        },
    }

    def __init__(self):
        self._results: List[EvaluationResult] = []
        self._baseline_results: Dict[str, EvaluationResult] = {}  # 传统方案基线

    def evaluate_task(
        self,
        task_id: str,
        task_type: str,
        model: str,
        use_lcm: bool,
        output: str,
        ground_truth: Optional[Dict] = None,
        duration_ms: float = 0.0,
    ) -> EvaluationResult:
        """
        评估单次任务输出
        
        Args:
            task_id: 任务 ID
            task_type: 任务类型
            model: 模型名称
            use_lcm: 是否使用 LCM
            output: 模型输出
            ground_truth: 标准答案（可选）
            duration_ms: 执行耗时
        """
        metrics = self._compute_metrics(output, ground_truth)
        
        result = EvaluationResult(
            task_id=task_id,
            task_type=task_type,
            model=model,
            use_lcm=use_lcm,
            metrics=metrics,
            raw_output=output,
            duration_ms=duration_ms,
        )
        
        self._results.append(result)
        
        # 如果是传统方案，记录为基线
        if not use_lcm:
            self._baseline_results[f"{task_type}:{model}"] = result
        
        return result

    def _compute_metrics(
        self,
        output: str,
        ground_truth: Optional[Dict] = None,
    ) -> QualityMetrics:
        """计算质量指标"""
        metrics = QualityMetrics()
        
        # 检测幻觉（编造不存在的 chunk 引用）
        metrics.hallucination_count = self._detect_hallucinations(output)
        
        # 计算聚焦准确度
        metrics.focus_accuracy = self._compute_focus_accuracy(output)
        
        # 计算上下文利用率
        metrics.context_utilization = self._compute_context_utilization(output)
        
        # 计算 token 效率
        metrics.token_efficiency = self._compute_token_efficiency(output)
        
        # 计算答案完整性
        metrics.answer_completeness = self._compute_completeness(output, ground_truth)
        
        # 计算引用准确度
        metrics.citation_accuracy = self._compute_citation_accuracy(output, ground_truth)
        
        # 计算指令遵循度
        metrics.instruction_following = self._compute_instruction_following(output)
        
        return metrics

    def _detect_hallucinations(self, output: str) -> int:
        """检测幻觉（编造不存在的 chunk 引用）"""
        import re
        # 查找 [NEED_CHUNK:xxx] 或引用不存在的 chunk
        hallucination_patterns = [
            r"\[NEED_CHUNK:([A-Za-z0-9_\-]+)\]",  # 未解析的哨兵标记
            r"chunk[\s\w]*does not exist",
            r"未找到.*chunk",
        ]
        
        count = 0
        for pattern in hallucination_patterns:
            count += len(re.findall(pattern, output, re.IGNORECASE))
        
        return count

    def _compute_focus_accuracy(self, output: str) -> float:
        """计算聚焦准确度（输出是否集中在相关主题上）

        基于以下启发式规则：
        1. 填充词密度（越低越好）
        2. 结构化内容比例（代码块、列表、引用越高越好）
        3. 重复内容比例（越低越好）
        """
        if not output.strip():
            return 0.0

        lines = output.split("\n")
        if not lines:
            return 0.0

        # 1. 计算填充词密度
        filler_words = [
            "让我", "让我想想", "我来分析", "首先", "总结", "嗯", "啊", "呢",
            "let me", "i think", "first", "in summary", "well", "so", "okay",
            "实际上", "说实话", "怎么说呢", "那个", "这个", "就是",
        ]
        filler_count = sum(1 for line in lines if any(kw in line.lower() for kw in filler_words))
        filler_ratio = filler_count / len(lines)

        # 2. 计算结构化内容比例（代码块、列表、引用标记）
        structured_patterns = [
            r"^\s*[-*+]\s",  # 列表项
            r"^\s*\d+\.\s",   # 编号列表
            r"^```",          # 代码块
            r"^\s*>",         # 引用
            r"\[Chunk",       # Chunk 引用
            r"`[^`]+`",       # 行内代码
        ]
        import re
        structured_count = sum(1 for line in lines if any(re.search(p, line) for p in structured_patterns))
        structured_ratio = structured_count / len(lines)

        # 3. 计算重复内容比例
        unique_lines = set(line.strip().lower() for line in lines if len(line.strip()) > 10)
        total_meaningful = sum(1 for line in lines if len(line.strip()) > 10)
        uniqueness_ratio = len(unique_lines) / total_meaningful if total_meaningful > 0 else 0

        # 综合评分：结构化内容加分，填充词和重复内容减分
        score = (
            structured_ratio * 0.5 +      # 结构化内容权重 50%
            uniqueness_ratio * 0.3 +       # 唯一性权重 30%
            (1 - filler_ratio) * 0.2       # 非填充词权重 20%
        )

        return min(max(score, 0.0), 1.0)

    def _compute_context_utilization(self, output: str) -> float:
        """计算上下文利用率"""
        # 检查输出中引用了多少 chunk 内容
        import re
        citations = re.findall(r"\[Chunk.*?\]", output)
        return min(len(citations) / 5, 1.0)  # 假设引用 5 个 chunk 为满分

    def _compute_token_efficiency(self, output: str) -> float:
        """计算 token 效率"""
        # 有用内容 / 总内容
        total_chars = len(output)
        if total_chars == 0:
            return 0.0
        
        # 去除填充词后的有效内容
        filler_words = ["嗯", "啊", "呢", "吧", "哦", "呃", "那个", "这个"]
        useful_chars = total_chars
        for word in filler_words:
            useful_chars -= output.count(word) * len(word)
        
        return max(0, useful_chars / total_chars)

    def _compute_completeness(
        self,
        output: str,
        ground_truth: Optional[Dict] = None,
    ) -> float:
        """计算答案完整性"""
        if not ground_truth:
            return 0.5  # 无标准答案时返回中性值
        
        expected_points = ground_truth.get("expected_points", [])
        if not expected_points:
            return 0.5
        
        matched = 0
        for point in expected_points:
            if point.lower() in output.lower():
                matched += 1
        
        return matched / len(expected_points)

    def _compute_citation_accuracy(
        self,
        output: str,
        ground_truth: Optional[Dict] = None,
    ) -> float:
        """计算引用准确度"""
        if not ground_truth:
            return 0.5
        
        expected_citations = ground_truth.get("expected_citations", [])
        if not expected_citations:
            return 0.5
        
        matched = 0
        for citation in expected_citations:
            if citation in output:
                matched += 1
        
        return matched / len(expected_citations)

    def _compute_instruction_following(self, output: str) -> float:
        """计算指令遵循度"""
        score = 1.0
        
        # 检查是否重复内容（违反"不要重复"指令）
        lines = output.split("\n")
        seen = set()
        duplicates = 0
        for line in lines:
            normalized = line.strip().lower()
            if normalized in seen and len(normalized) > 10:
                duplicates += 1
            seen.add(normalized)
        
        if duplicates > 0:
            score -= min(duplicates * 0.1, 0.5)
        
        # 检查是否包含未解析的哨兵标记
        if "[NEED_CHUNK:" in output:
            score -= 0.3
        
        return max(0, score)

    def compute_vs_traditional(self, result: EvaluationResult) -> None:
        """计算与传统方案的对比指标"""
        baseline_key = f"{result.task_type}:{result.model}"
        baseline = self._baseline_results.get(baseline_key)
        
        if not baseline:
            return
        
        result.metrics.vs_traditional_quality_delta = (
            result.metrics.answer_completeness - baseline.metrics.answer_completeness
        ) * 100
        
        result.metrics.vs_traditional_latency_delta = (
            result.duration_ms - baseline.duration_ms
        )
        
        # Token 节省（简化计算）
        result.metrics.vs_traditional_token_delta = 0  # 需要实际 token 计数

    def get_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self._results:
            return {"status": "no_data"}
        
        lcm_results = [r for r in self._results if r.use_lcm]
        traditional_results = [r for r in self._results if not r.use_lcm]
        
        def avg_metrics(results):
            if not results:
                return {}
            return {
                "avg_focus_accuracy": sum(r.metrics.focus_accuracy for r in results) / len(results),
                "avg_completeness": sum(r.metrics.answer_completeness for r in results) / len(results),
                "avg_citation_accuracy": sum(r.metrics.citation_accuracy for r in results) / len(results),
                "avg_instruction_following": sum(r.metrics.instruction_following for r in results) / len(results),
                "avg_hallucination_count": sum(r.metrics.hallucination_count for r in results) / len(results),
                "avg_duration_ms": sum(r.duration_ms for r in results) / len(results),
            }
        
        return {
            "status": "completed",
            "total_tasks": len(self._results),
            "lcm_tasks": len(lcm_results),
            "traditional_tasks": len(traditional_results),
            "lcm_metrics": avg_metrics(lcm_results),
            "traditional_metrics": avg_metrics(traditional_results),
            "improvement": {
                "focus_accuracy_delta": (
                    avg_metrics(lcm_results).get("avg_focus_accuracy", 0) -
                    avg_metrics(traditional_results).get("avg_focus_accuracy", 0)
                ),
                "completeness_delta": (
                    avg_metrics(lcm_results).get("avg_completeness", 0) -
                    avg_metrics(traditional_results).get("avg_completeness", 0)
                ),
            },
        }

    def export_report(self, filepath: str) -> None:
        """导出评估报告"""
        report = {
            "summary": self.get_summary(),
            "results": [
                {
                    "task_id": r.task_id,
                    "task_type": r.task_type,
                    "model": r.model,
                    "use_lcm": r.use_lcm,
                    "duration_ms": r.duration_ms,
                    "metrics": asdict(r.metrics),
                }
                for r in self._results
            ],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


class BenchmarkRunner:
    """
    基准测试运行器
    自动化运行评估任务
    """

    def __init__(self, evaluator: QualityEvaluator):
        self.evaluator = evaluator
        self._tasks: List[Dict] = []

    def add_task(self, task_id: str, task_type: str, model: str, 
                 use_lcm: bool, runner_fn: Callable, ground_truth: Optional[Dict] = None):
        """添加评估任务"""
        self._tasks.append({
            "task_id": task_id,
            "task_type": task_type,
            "model": model,
            "use_lcm": use_lcm,
            "runner_fn": runner_fn,
            "ground_truth": ground_truth,
        })

    def run_all(self) -> List[EvaluationResult]:
        """运行所有评估任务"""
        results = []
        
        for task in self._tasks:
            print(f"Running {task['task_id']}...")
            start = time.time()
            
            try:
                output = task["runner_fn"]()
                duration = (time.time() - start) * 1000
                
                result = self.evaluator.evaluate_task(
                    task_id=task["task_id"],
                    task_type=task["task_type"],
                    model=task["model"],
                    use_lcm=task["use_lcm"],
                    output=output,
                    ground_truth=task.get("ground_truth"),
                    duration_ms=duration,
                )
                
                if task["use_lcm"]:
                    self.evaluator.compute_vs_traditional(result)
                
                results.append(result)
                print(f"  ✓ Completed in {duration:.2f}ms")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        return results
