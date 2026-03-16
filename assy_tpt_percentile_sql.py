"""
Assembly TPT (Throughput Time) 分位数 SQL 查询构建器

支持两种分位数计算模式：
1. 按周分组（weekly）：按 assy_date 和 week 列分区计算分位数
2. Total（全量）：仅按 assy_date 分区计算分位数，不按时间分组
"""


def get_weekly_percentile_expr(
    p_percentile,
    tpt_col_name,
    assy_date_col_name,
    week_col_name,
    alias="Assy_WW",
):
    """
    生成按周分区的分位数 SQL 表达式。

    使用 PERCENTILE_CONT 窗口函数，按 assy_date 和 week 列进行分区。

    Args:
        p_percentile: 分位数值，范围 0.0 ~ 1.0（如 0.5 表示中位数）
        tpt_col_name: 吞吐时间列名
        assy_date_col_name: 装配日期列名
        week_col_name: 工作周列名
        alias: 输出列别名，默认 "Assy_WW"

    Returns:
        str: SQL 表达式字符串
    """
    return (
        f"cast(PERCENTILE_CONT({p_percentile}) WITHIN GROUP (ORDER BY {tpt_col_name}) "
        f"OVER (partition by {assy_date_col_name}, {week_col_name}) "
        f"as decimal(10,2)) as {alias}"
    )


def get_total_percentile_expr(
    p_percentile,
    tpt_col_name,
    assy_date_col_name,
    alias="Assy",
):
    """
    生成 Total（全量）分区的分位数 SQL 表达式，不按时间分组。

    使用 PERCENTILE_CONT 窗口函数，仅按 assy_date 列进行分区，
    不再按周或其他时间维度进一步细分。

    Args:
        p_percentile: 分位数值，范围 0.0 ~ 1.0（如 0.5 表示中位数）
        tpt_col_name: 吞吐时间列名
        assy_date_col_name: 装配日期列名
        alias: 输出列别名，默认 "Assy"

    Returns:
        str: SQL 表达式字符串
    """
    return (
        f"cast(PERCENTILE_CONT({p_percentile}) WITHIN GROUP (ORDER BY {tpt_col_name}) "
        f"OVER (partition by {assy_date_col_name}) "
        f"as decimal(10,2)) as {alias}"
    )


def build_assy_tpt_percentile_query(
    p_percentile,
    tpt_col_name,
    assy_date_col_name,
    week_col_name,
    base_table_expr,
    extra_select_cols=None,
):
    """
    构建同时包含按周分位和 Total 分位的完整 SQL 查询。

    查询结果中同时包含：
    - Assy_WW：按周分区的分位数
    - Assy：Total（不按时间分组）的分位数

    Args:
        p_percentile: 分位数值，范围 0.0 ~ 1.0
        tpt_col_name: 吞吐时间列名
        assy_date_col_name: 装配日期列名
        week_col_name: 工作周列名
        base_table_expr: 基础表名或子查询表达式
        extra_select_cols: 额外需要 SELECT 的列名列表，默认 None

    Returns:
        str: 完整 SQL 查询字符串
    """
    weekly_expr = get_weekly_percentile_expr(
        p_percentile, tpt_col_name, assy_date_col_name, week_col_name
    )
    total_expr = get_total_percentile_expr(
        p_percentile, tpt_col_name, assy_date_col_name
    )

    select_parts = []
    if extra_select_cols:
        select_parts.extend(extra_select_cols)
    select_parts.append(weekly_expr)
    select_parts.append(total_expr)

    select_clause = ",\n    ".join(select_parts)

    return (
        f"SELECT\n"
        f"    {select_clause}\n"
        f"FROM {base_table_expr}"
    )
