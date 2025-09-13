"""
Trend data models and enums for the trend radar application.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List
from dataclasses import dataclass, asdict


class TrendCategory(Enum):
    """Categories for trend classification"""
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"


class TrendImpact(Enum):
    """Impact levels for trends"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrendTimeHorizon(Enum):
    """Time horizons for trend maturity"""
    EMERGING = "emerging"      # 0-6 months
    SHORT_TERM = "short_term"  # 6-18 months
    MEDIUM_TERM = "medium_term"  # 1-3 years
    LONG_TERM = "long_term"    # 3+ years


@dataclass
class Trend:
    """Main trend data structure"""
    id: str
    title: str
    description: str
    category: TrendCategory
    impact: TrendImpact
    time_horizon: TrendTimeHorizon
    confidence_score: float  # 0.0 to 1.0
    sources: List[str]
    keywords: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trend to dictionary with enum values serialized"""
        data = asdict(self)
        data['category'] = self.category.value
        data['impact'] = self.impact.value
        data['time_horizon'] = self.time_horizon.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trend':
        """Create trend from dictionary"""
        return cls(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            category=TrendCategory(data['category']),
            impact=TrendImpact(data['impact']),
            time_horizon=TrendTimeHorizon(data['time_horizon']),
            confidence_score=data['confidence_score'],
            sources=data['sources'],
            keywords=data['keywords'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


@dataclass
class RadarPoint:
    """Data structure for radar chart visualization"""
    id: str
    title: str
    category: str
    x: float  # Time horizon position (1-4)
    y: float  # Impact position (1-4) 
    size: float  # Confidence-based size
    description: str
    keywords: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    