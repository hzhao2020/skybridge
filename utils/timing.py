"""
时间记录工具类：用于记录workflow、operation和transmission的执行时间
"""
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class TransmissionTiming:
    """传输时间记录"""
    source: str  # 起始地（本地路径或云存储URI）
    destination: str  # 目的地（云存储URI）
    transmission_type: str  # 传输类型：upload_local_to_cloud, transfer_s3_to_s3, transfer_gcs_to_gcs, transfer_s3_to_gcs, transfer_gcs_to_s3
    start_time: float  # 开始时间（Unix时间戳）
    end_time: float  # 结束时间（Unix时间戳）
    duration: float  # 持续时间（秒）
    operation: Optional[str] = None  # 传输发生的operation名称
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "destination": self.destination,
            "transmission_type": self.transmission_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": round(self.duration, 3),
            "start_time_formatted": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "end_time_formatted": datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "operation": self.operation,
        }


@dataclass
class OperationTiming:
    """Operation执行时间记录"""
    operation_name: str  # operation名称（如segment, split, caption）
    operation_pid: str  # operation的pid
    start_time: float  # 开始时间（Unix时间戳）
    end_time: float  # 结束时间（Unix时间戳）
    duration: float  # 持续时间（秒）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_name": self.operation_name,
            "operation_pid": self.operation_pid,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": round(self.duration, 3),
            "start_time_formatted": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "end_time_formatted": datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        }


@dataclass
class WorkflowTiming:
    """Workflow执行时间记录"""
    workflow_name: str  # workflow名称
    start_time: float  # 开始时间（Unix时间戳）
    end_time: float  # 结束时间（Unix时间戳）
    total_duration: float  # 总持续时间（秒）
    operations: List[OperationTiming] = field(default_factory=list)  # operation时间记录列表
    transmissions: List[TransmissionTiming] = field(default_factory=list)  # 传输时间记录列表
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration": round(self.total_duration, 3),
            "start_time_formatted": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "end_time_formatted": datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "operations": [op.to_dict() for op in self.operations],
            "transmissions": [trans.to_dict() for trans in self.transmissions],
        }


class TimingRecorder:
    """时间记录器：单例模式，用于记录workflow执行过程中的所有时间信息"""
    
    _instance: Optional['TimingRecorder'] = None
    _workflow_timing: Optional[WorkflowTiming] = None
    _current_operation: Optional[str] = None  # 当前正在执行的operation名称
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._workflow_timing = None
            cls._instance._current_operation = None
            cls._instance._operation_actual_start_times = {}
        return cls._instance
    
    def start_workflow(self, workflow_name: str):
        """开始记录workflow"""
        self._workflow_timing = WorkflowTiming(
            workflow_name=workflow_name,
            start_time=time.time(),
            end_time=0.0,
            total_duration=0.0,
        )
    
    def end_workflow(self):
        """结束记录workflow"""
        if self._workflow_timing:
            self._workflow_timing.end_time = time.time()
            self._workflow_timing.total_duration = (
                self._workflow_timing.end_time - self._workflow_timing.start_time
            )
    
    def record_operation(self, operation_name: str, operation_pid: str, start_time: float, end_time: float):
        """记录operation执行时间"""
        if self._workflow_timing:
            duration = end_time - start_time
            operation_timing = OperationTiming(
                operation_name=operation_name,
                operation_pid=operation_pid,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
            )
            self._workflow_timing.operations.append(operation_timing)
    
    def record_transmission(self, source: str, destination: str, transmission_type: str, start_time: float, end_time: float, operation: Optional[str] = None):
        """记录传输时间
        
        Args:
            source: 源位置
            destination: 目标位置
            transmission_type: 传输类型
            start_time: 开始时间
            end_time: 结束时间
            operation: 传输发生的operation名称（如果为None，使用当前operation）
        """
        if self._workflow_timing:
            duration = end_time - start_time
            # 如果没有指定operation，使用当前operation
            if operation is None:
                operation = self._current_operation
            
            transmission_timing = TransmissionTiming(
                source=source,
                destination=destination,
                transmission_type=transmission_type,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                operation=operation,
            )
            self._workflow_timing.transmissions.append(transmission_timing)
    
    def set_current_operation(self, operation_name: Optional[str]):
        """设置当前正在执行的operation名称"""
        self._current_operation = operation_name
    
    def get_timing(self) -> Optional[WorkflowTiming]:
        """获取当前的时间记录"""
        return self._workflow_timing
    
    def reset(self):
        """重置记录器"""
        self._workflow_timing = None
    
    def save_to_file(self, filepath: str):
        """将时间记录保存到文件"""
        if not self._workflow_timing:
            raise ValueError("没有可保存的时间记录")
        
        timing_dict = self._workflow_timing.to_dict()
        
        # 保存为JSON格式
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(timing_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n时间记录已保存到: {filepath}")
        
        # 同时打印摘要信息
        self.print_summary()
    
    def print_summary(self):
        """打印时间记录摘要"""
        if not self._workflow_timing:
            return
        
        print("\n" + "="*80)
        print("Workflow 执行时间摘要")
        print("="*80)
        print(f"Workflow名称: {self._workflow_timing.workflow_name}")
        print(f"总执行时间: {self._workflow_timing.total_duration:.3f} 秒")
        print(f"开始时间: {datetime.fromtimestamp(self._workflow_timing.start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print(f"结束时间: {datetime.fromtimestamp(self._workflow_timing.end_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        print(f"\nOperation执行时间 ({len(self._workflow_timing.operations)} 个):")
        for op in self._workflow_timing.operations:
            print(f"  - {op.operation_name} ({op.operation_pid}): {op.duration:.3f} 秒")
        
        print(f"\n传输时间 ({len(self._workflow_timing.transmissions)} 次):")
        for trans in self._workflow_timing.transmissions:
            print(f"  - {trans.transmission_type}:")
            print(f"    源: {trans.source}")
            print(f"    目标: {trans.destination}")
            if trans.operation:
                print(f"    发生operation: {trans.operation}")
            print(f"    耗时: {trans.duration:.3f} 秒")
        
        print("="*80 + "\n")
