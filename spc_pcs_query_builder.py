"""
SPC PCS 数据查询构建器

支持两种查询模式：
1. 原始数据查询（raw）：返回逐条 SPC 交易记录，按 SPC_TXN_DATE ASC 排序
2. 聚合查询（aggregated）：按 SPC_LOT 和 MODULE 分组，计算 CHART_VALUE 均值

关键设计原则
-----------
聚合模式下，SPC_TXN_DATE 不出现在 SELECT 列表或 GROUP BY 子句中，
因此 ORDER BY SPC_TXN_DATE ASC 在该模式下语法非法（SQL Server 报错）。
本模块通过将两种模式拆分为独立函数来彻底规避此问题：
- build_spc_pcs_raw_query()        → 包含 ORDER BY SPC_TXN_DATE ASC
- build_spc_pcs_aggregated_query() → 不包含 ORDER BY SPC_TXN_DATE ASC
"""

_TABLE = "[Assembly_PCS].[dbo].[dwd_PCS_Data]"


def _build_where_clause(
    modules,
    facilities,
    monitor_set_names,
    area_groupings,
    ct_ids,
    start_work_week,
    end_work_week,
):
    """
    构建通用 WHERE 子句。

    Args:
        modules (list[str]): MODULE 过滤值列表
        facilities (list[str]): SPC_FACILITY 过滤值列表
        monitor_set_names (list[str]): MONITOR_SET_NAME 过滤值列表
        area_groupings (list[str]): AREA_GROUPING 过滤值列表
        ct_ids (list[str]): CT_ID 过滤值列表
        start_work_week (str): 起始工作周，格式 'YYYYWW'（如 '202606'）
        end_work_week (str): 结束工作周，格式 'YYYYWW'（如 '202611'）

    Returns:
        str: WHERE 子句字符串（以 "WHERE 1=1" 开头）
    """

    def _in_list(col, values):
        quoted = ", ".join(f"'{v}'" for v in values)
        return f"{col} in ({quoted})"

    conditions = [
        "1=1",
        _in_list("MODULE", modules),
        _in_list("SPC_FACILITY", facilities),
        _in_list("MONITOR_SET_NAME", monitor_set_names),
        _in_list("AREA_GROUPING", area_groupings),
        _in_list("CT_ID", ct_ids),
        f"SITE_WORK_WEEK>='{start_work_week}'",
        f"SITE_WORK_WEEK<='{end_work_week}'",
        f"MONITOR_SET_NAME='{monitor_set_names[0]}'",
    ]

    return "WHERE " + "\n            AND ".join(conditions)


def build_spc_pcs_raw_query(
    modules,
    facilities,
    monitor_set_names,
    area_groupings,
    ct_ids,
    start_work_week,
    end_work_week,
):
    """
    构建原始 SPC PCS 数据查询。

    返回逐条交易记录，并按 SPC_TXN_DATE ASC 升序排列。
    适用于需要查看每条测量记录的场景。

    Args:
        modules (list[str]): MODULE 过滤值列表
        facilities (list[str]): SPC_FACILITY 过滤值列表
        monitor_set_names (list[str]): MONITOR_SET_NAME 过滤值列表
        area_groupings (list[str]): AREA_GROUPING 过滤值列表
        ct_ids (list[str]): CT_ID 过滤值列表
        start_work_week (str): 起始工作周
        end_work_week (str): 结束工作周

    Returns:
        str: 完整 SQL 查询字符串（包含 ORDER BY SPC_TXN_DATE ASC）
    """
    where_clause = _build_where_clause(
        modules, facilities, monitor_set_names,
        area_groupings, ct_ids, start_work_week, end_work_week,
    )

    return (
        f"SELECT SPC_LOT, SPC_TXN_DATE, CHART_VALUE, MODULE\n"
        f"FROM {_TABLE} WITH(NOLOCK)\n"
        f"            {where_clause}\n"
        f"            ORDER BY SPC_TXN_DATE ASC"
    )


def build_spc_pcs_aggregated_query(
    modules,
    facilities,
    monitor_set_names,
    area_groupings,
    ct_ids,
    start_work_week,
    end_work_week,
):
    """
    构建聚合 SPC PCS 数据查询。

    按 SPC_LOT 和 MODULE 分组，计算每组的 CHART_VALUE 均值。
    **此函数不包含 ORDER BY SPC_TXN_DATE ASC**，原因如下：
        - SPC_TXN_DATE 未出现在 SELECT 列表中
        - SPC_TXN_DATE 未出现在 GROUP BY 子句中
        - SQL Server 不允许在上述条件下将其用于 ORDER BY

    如需对聚合结果排序，请在外层查询或调用处单独指定
    ORDER BY SPC_LOT 等已在 SELECT/GROUP BY 中存在的列。

    Args:
        modules (list[str]): MODULE 过滤值列表
        facilities (list[str]): SPC_FACILITY 过滤值列表
        monitor_set_names (list[str]): MONITOR_SET_NAME 过滤值列表
        area_groupings (list[str]): AREA_GROUPING 过滤值列表
        ct_ids (list[str]): CT_ID 过滤值列表
        start_work_week (str): 起始工作周
        end_work_week (str): 结束工作周

    Returns:
        str: 完整 SQL 查询字符串（不含 ORDER BY SPC_TXN_DATE）
    """
    where_clause = _build_where_clause(
        modules, facilities, monitor_set_names,
        area_groupings, ct_ids, start_work_week, end_work_week,
    )

    return (
        f"SELECT SPC_LOT, AVG(CHART_VALUE) AS CHART_VALUE, MODULE\n"
        f"FROM {_TABLE} WITH(NOLOCK)\n"
        f"            {where_clause}\n"
        f"            GROUP BY SPC_LOT, MODULE"
    )
