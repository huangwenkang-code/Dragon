"""Hot topic detector — statistical analysis of trending themes and sectors."""

from __future__ import annotations

from collections import Counter

from shared.schemas.agent_state import Event, SentimentScore
from shared.utils.logging import get_logger

logger = get_logger(__name__)

# Sector keywords for A-share market
SECTOR_KEYWORDS = {
    "新能源": ["新能源", "光伏", "风电", "储能", "锂电池", "钠电池", "固态电池"],
    "半导体": ["半导体", "芯片", "光刻机", "晶圆", "封装", "EDA", "GPU", "AI芯片"],
    "人工智能": ["AI", "人工智能", "大模型", "ChatGPT", "AGI", "智能体", "机器人"],
    "医药": ["医药", "创新药", "CXO", "医疗器械", "生物制药", "疫苗", "中药"],
    "消费": ["消费", "白酒", "食品", "家电", "零售", "免税", "预制菜"],
    "金融": ["银行", "券商", "保险", "信托", "AMC", "数字化"],
    "地产": ["地产", "房地产", "基建", "建材", "物业", "城中村"],
    "汽车": ["汽车", "整车", "自动驾驶", "零部件", "一体化压铸", "飞行汽车"],
    "军工": ["军工", "航空航天", "船舶", "弹药", "信息化"],
    "数字经济": ["数据", "信创", "东数西算", "算力", "数据要素", "数字政府"],
    "电力": ["电力", "电网", "火电", "水电", "核电", "虚拟电厂", "特高压"],
    "周期": ["钢铁", "有色金属", "煤炭", "化工", "稀土", "黄金", "铜"],
    "低空经济": ["低空", "无人机", "eVTOL", "通航", "空管"],
    "合成生物": ["合成生物", "基因编辑", "生物制造", "细胞治疗"],
    "量子": ["量子", "量子计算", "量子通信"],
    "机器人": ["机器人", "人形机器人", "灵巧手", "减速器"],
}


def detect_hot_sectors(events: list[Event]) -> list[dict]:
    """Detect hot sectors from event analysis.

    Returns list of {sector, count, events, heat_score} sorted by heat.
    """
    sector_counter: Counter = Counter()
    sector_events: dict[str, list[str]] = {}

    for evt in events:
        text = f"{evt.title} {evt.content} {evt.narrative} {' '.join(evt.keywords)}"
        for sector, keywords in SECTOR_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                sector_counter[sector] += 1
                sector_events.setdefault(sector, []).append(evt.title[:60])

    total = sum(sector_counter.values()) or 1
    result = []
    for sector, count in sector_counter.most_common(15):
        result.append({
            "sector": sector,
            "mention_count": count,
            "heat_score": round(count / total, 3),
            "top_events": sector_events.get(sector, [])[:3],
        })

    logger.info("detect_hot_sectors: %d sectors found", len(result))
    return result


def detect_hot_keywords(
    events: list[Event],
    sentiment_scores: list[SentimentScore],
) -> list[dict]:
    """Aggregate hot keywords across events and sentiment scores.

    Returns top keywords with frequency and average sentiment.
    """
    kw_counter: Counter = Counter()
    kw_sentiments: dict[str, list[float]] = {}

    for evt in events:
        for kw in evt.keywords:
            kw_counter[kw] += 1
            kw_sentiments.setdefault(kw, []).append(evt.strength)

    for score in sentiment_scores:
        for kw in score.keywords:
            kw_counter[kw] += 1
            kw_sentiments.setdefault(kw, []).append(score.sentiment_score)

    result = []
    for kw, count in kw_counter.most_common(30):
        sentiments = kw_sentiments.get(kw, [])
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        result.append({
            "keyword": kw,
            "frequency": count,
            "avg_sentiment": round(avg_sentiment, 3),
        })

    return result


def detect_sector_rotation(
    events: list[Event],
    sentiment_scores: list[SentimentScore],
) -> dict:
    """Detect sector rotation signals.

    Returns hot sectors, keywords, and whether conditions are favorable for
    market speculation (妖股行情).
    """
    hot_sectors = detect_hot_sectors(events)
    hot_keywords = detect_hot_keywords(events, sentiment_scores)

    # Conditions favorable for 妖股:
    # 1. Multiple hot sectors (>3)
    # 2. High speculation keywords
    # 3. Strong narratives
    spec_kw_present = any(
        kw["keyword"] in {"连板", "龙头", "妖股", "打板", "接力", "封板", "涨停"}
        for kw in hot_keywords[:20]
    )

    multi_sector = len(hot_sectors) >= 3
    favorable = multi_sector and spec_kw_present

    return {
        "hot_sectors": hot_sectors[:10],
        "hot_keywords": hot_keywords[:20],
        "sector_count": len(hot_sectors),
        "妖股环境": "有利" if favorable else ("中性" if multi_sector else "不利"),
        "多板块轮动": multi_sector,
        "投机关键词活跃": spec_kw_present,
    }
