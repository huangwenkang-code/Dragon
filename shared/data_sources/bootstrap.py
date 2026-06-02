"""Bootstrap data sources — register all available sources at startup."""
from shared.data_sources.registry import register, is_registered
from shared.utils.logging import get_logger

logger = get_logger(__name__)


def bootstrap_sources() -> list[str]:
    """Register all available data sources. Returns list of registered names."""
    registered = []

    # mootdx
    try:
        if not is_registered("mootdx"):
            from shared.data_sources.sources.mootdx_source import MootdxSource
            register("mootdx", MootdxSource())
            registered.append("mootdx")
    except ImportError as e:
        logger.warning("mootdx not available: %s", e)

    # tx_finance
    try:
        if not is_registered("tx_finance"):
            from shared.data_sources.sources.tx_finance_source import TxFinanceSource
            register("tx_finance", TxFinanceSource())
            registered.append("tx_finance")
    except Exception as e:
        logger.warning("tx_finance not available: %s", e)

    # ths_hot
    try:
        if not is_registered("ths_hot"):
            from shared.data_sources.sources.ths_hot_source import ThsHotSource
            register("ths_hot", ThsHotSource())
            registered.append("ths_hot")
    except Exception as e:
        logger.warning("ths_hot not available: %s", e)

    return registered
