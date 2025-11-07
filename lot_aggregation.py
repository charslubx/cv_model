"""
针对assy/finish/test_fpo/test_atpo segment进行分组聚合统计
按照除segment以外的key分组，计算lot个数和平均值
"""

from django.db.models import Count, Avg, F, Value, CharField


def get_lot_aggregated_data(segment, r_facility, r_owner, r_prodgroup3, request, r_start, r_end, r_timewindow, r_cal=None):
    """
    获取分组聚合后的lot数据
    
    分组字段：
    - prodgroup3 (PG3)
    - out_month (MONTH) 或 out_ww/out_qtr (根据timewindow)
    - resp_area_raw (RESP_AREA_RAW)
    - responsible_org (RESPONSIBLE_ORG)
    - excursion_type (EXCURSION_TYPE)
    - responsible_func_area (RESPONSIBLE_FUNC_AREA)
    - xrb (XRB)
    - xrb_title (XRB_TITLE)
    
    聚合计算：
    - lot_count: lot的个数
    - avg_intel_products: Intel_Products的平均值
    - avg_intel_foundry_non_atm: Intel_Foundry_Non_ATM的平均值
    - avg_intel_foundry_atm: Intel_Foundry_ATM的平均值
    """
    
    # 构建过滤条件
    filters = {}
    filters['prodgroup3__in'] = r_prodgroup3
    filters['facility__in'] = r_facility
    filters['owner__in'] = r_owner

    print('p r_prodgroup3=======', r_prodgroup3, 'length=', len(r_prodgroup3))
    print('p r_period_start=======', r_start)
    print('p r_period_end=======', r_end)

    # 根据时间窗口设置时间过滤条件
    time_field = None
    if r_timewindow.lower() == 'ww':
        time_field = 'out_ww'
        if r_start:
            filters['out_ww__gte'] = r_start
        if r_end:
            filters['out_ww__lte'] = r_end
    elif r_timewindow.lower() == 'month':
        time_field = 'out_month'
        if r_start:
            filters['out_month__gte'] = r_start
        if r_end:
            filters['out_month__lte'] = r_end
    else:
        time_field = 'out_qtr'
        if r_start:
            filters['out_qtr__gte'] = r_start
        if r_end:
            filters['out_qtr__lte'] = r_end

    if r_cal:
        if 'NA' in r_cal:
            r_cal.remove('NA')
            r_cal.append(None)
            r_cal.append('_')
        filters['cal__in'] = r_cal

    # 定义分组字段（不包含segment，segment将作为固定值添加）
    group_by_fields = [
        'prodgroup3',
        time_field,  # 根据timewindow动态选择
        'resp_area_raw',
        'responsible_org',
        'excursion_type',
        'responsible_func_area',
        'xrb',
        'xrb_title'
    ]

    # 定义聚合计算
    aggregations = {
        'lot_count': Count('lot', distinct=True),
        'avg_intel_products': Avg('intel_products'),
        'avg_intel_foundry_non_atm': Avg('intel_foundry_non_atm'),
        'avg_intel_foundry_atm': Avg('intel_foundry_atm')
    }

    qs = None

    # 根据segment选择对应的Model，并添加segment标识
    if segment == 'assy':
        qs = AssyLotModel.objects.filter(**filters).annotate(
            segment=Value('assy', output_field=CharField())
        )
    elif segment == 'finish':
        qs = FinishLotModel.objects.filter(**filters).annotate(
            segment=Value('finish', output_field=CharField())
        )
    elif segment == 'atpo':
        qs = TestAtpoLotModel.objects.filter(**filters).annotate(
            segment=Value('atpo', output_field=CharField())
        )
    elif segment == 'fpo':
        qs = TestFpoLotModel.objects.filter(**filters).annotate(
            segment=Value('fpo', output_field=CharField())
        )
    elif segment == 'assy_ifs':
        qs = IFSAssyLotModel.objects.filter(**filters).annotate(
            segment=Value('assy_ifs', output_field=CharField())
        )
    elif segment == 'finish_ifs':
        qs = IFSFinishLotModel.objects.filter(**filters).annotate(
            segment=Value('finish_ifs', output_field=CharField())
        )
    elif segment == 'atpo_ifs':
        qs = IFSTestAtpoLotModel.objects.filter(**filters).annotate(
            segment=Value('atpo_ifs', output_field=CharField())
        )
    elif segment == 'fpo_ifs':
        qs = IFSTestFpoLotModel.objects.filter(**filters).annotate(
            segment=Value('fpo_ifs', output_field=CharField())
        )

    if qs is None:
        return []

    # 将segment添加到分组字段中
    group_by_fields_with_segment = ['segment'] + group_by_fields

    # 执行分组和聚合
    result = qs.values(*group_by_fields_with_segment).annotate(**aggregations).order_by(*group_by_fields_with_segment)

    result_list = list(result)
    
    return result_list




# 使用示例
"""
# 获取分组聚合后的数据
aggregated_data = get_lot_aggregated_data(
    segment='assy',
    r_facility=['facility1', 'facility2'],
    r_owner=['owner1'],
    r_prodgroup3=['PG1', 'PG2'],
    request=request,
    r_start='202301',
    r_end='202312',
    r_timewindow='month'
)

# 返回的聚合数据格式示例：
[
    {
        'segment': 'assy',  # segment标识
        'prodgroup3': 'PG1',
        'out_month': '202301',
        'resp_area_raw': 'Area1',
        'responsible_org': 'Org1',
        'excursion_type': 'Type1',
        'responsible_func_area': 'FuncArea1',
        'xrb': 'XRB1',
        'xrb_title': 'Title1',
        'lot_count': 150,  # lot的个数
        'avg_intel_products': 85.5,  # Intel_Products的平均值
        'avg_intel_foundry_non_atm': 65.3,  # Intel_Foundry_Non_ATM的平均值
        'avg_intel_foundry_atm': 45.2  # Intel_Foundry_ATM的平均值
    },
    {
        'segment': 'assy',  # 同一segment的另一组数据
        'prodgroup3': 'PG2',
        'out_month': '202301',
        'resp_area_raw': 'Area2',
        'responsible_org': 'Org2',
        'excursion_type': 'Type2',
        'responsible_func_area': 'FuncArea2',
        'xrb': 'XRB2',
        'xrb_title': 'Title2',
        'lot_count': 200,
        'avg_intel_products': 90.2,
        'avg_intel_foundry_non_atm': 70.5,
        'avg_intel_foundry_atm': 50.8
    },
    # ... 更多分组数据
]
"""
