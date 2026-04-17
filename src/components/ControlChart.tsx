import { useEffect, useRef, useImperativeHandle, forwardRef, useMemo, useCallback } from 'react'
import { Chart } from '@antv/g2'

const colorObj = {
  online: '#ff7f0e',
  userDefined: '#2ca02c',
  normalDot: '#1677ff',
  errDot: '#FF0000',
}

const MAX_RENDER_POINTS = 3000

const ControlChart = ({ chartData, tableData }: any, ref: any) => {
  const container = useRef<HTMLDivElement>(null)
  const chart = useRef<any>(null)
  const isRenderingRef = useRef(false)

  // 用 ref 存最新的派生数据，避免事件回调中的闭包陷阱
  const rawDataRef = useRef<any[]>([])
  const referenceRef = useRef<{
    onlineData: any[]
    userDefinedData: any[]
    userMin: number | undefined
    userMax: number | undefined
  }>({ onlineData: [], userDefinedData: [], userMin: undefined, userMax: undefined })

  const rawData = useMemo(
    () => chartData.map(([LOT, VAL]: [string, number]) => ({ LOT, VAL })),
    [chartData],
  )

  const { userMin, userMax, onlineData, userDefinedData } = useMemo(() => {
    let online: any[] = [], user: any[] = []

    tableData.forEach((item: any) => {
      if (item.Reference === 'user defined') user = [item.LCL, item.UCL, item.CL]
      if (item.Reference === 'Online') online = [item.LCL, item.UCL, item.CL]
    })

    const [min, max] = (() => {
      const LCL = parseFloat(user[0]), UCL = parseFloat(user[1])
      if (!isNaN(LCL) && !isNaN(UCL)) return [Math.min(LCL, UCL), Math.max(LCL, UCL)]
      return [isNaN(LCL) ? undefined : LCL, isNaN(UCL) ? undefined : UCL]
    })()

    return { userMin: min, userMax: max, onlineData: online, userDefinedData: user }
  }, [tableData])

  // 每次 memoized 值变化时同步到 ref，供回调闭包读取最新值
  useEffect(() => {
    rawDataRef.current = rawData
  }, [rawData])

  useEffect(() => {
    referenceRef.current = { onlineData, userDefinedData, userMin, userMax }
  }, [onlineData, userDefinedData, userMin, userMax])

  // ----- 颜色判断（读 ref，始终拿最新阈值） -----
  const getColor = useCallback((datum: any) => {
    const { userMin: min, userMax: max } = referenceRef.current
    if (min !== undefined && max !== undefined) {
      return min <= datum.VAL && datum.VAL <= max ? colorObj.normalDot : colorObj.errDot
    }
    if (min !== undefined) return datum.VAL >= min ? colorObj.normalDot : colorObj.errDot
    if (max !== undefined) return datum.VAL <= max ? colorObj.normalDot : colorObj.errDot
    return colorObj.normalDot
  }, [])

  // ----- 参考线绘制（读 ref） -----
  const addReferenceLines = useCallback((chartInstance: any) => {
    const { onlineData: od, userDefinedData: ud } = referenceRef.current

    const drawLine = (val: any, label: string, color: string, dashed?: boolean) => {
      if (!val || isNaN(parseFloat(val))) return
      chartInstance
        .lineY()
        .data([Number(val)])
        .style({
          stroke: color,
          strokeOpacity: 0.8,
          lineWidth: 1.5,
          lineDash: dashed ? [4, 4] : [],
        })
        .label({
          text: label,
          position: label.includes('LCL') ? 'top-right' : 'top-left',
        })
    }

    drawLine(od[0], 'Online LCL', colorObj.online, true)
    drawLine(od[1], 'Online UCL', colorObj.online, true)
    drawLine(od[2], 'Online CL', colorObj.online, true)
    drawLine(ud[0], 'User Defined LCL', colorObj.userDefined)
    drawLine(ud[1], 'User Defined UCL', colorObj.userDefined)
    drawLine(ud[2], 'User Defined CL', colorObj.userDefined)
  }, [])

  // ----- 核心：销毁旧图 + 重新初始化 -----
  const initChart = useCallback(async () => {
    if (!container.current || isRenderingRef.current) return
    isRenderingRef.current = true

    // 先销毁旧实例
    if (chart.current) {
      chart.current.destroy()
      chart.current = null
    }

    const data = rawDataRef.current
    const initialData = data.slice(0, MAX_RENDER_POINTS)
    const initialEndRatio = Math.min(MAX_RENDER_POINTS / Math.max(data.length, 1), 1)

    chart.current = new Chart({
      container: container.current!,
      autoFit: true,
      height: 400,
      animate: false,
    })

    chart.current
      .point()
      .data(initialData)
      .encode('x', 'LOT')
      .encode('y', 'VAL')
      .encode('shape', 'point')
      .encode('size', 3)
      .slider('x', {
        values: [0, initialEndRatio],
      })
      .axis('x', { tickCount: 5, label: { autoHide: true } })
      .scale('y', { nice: true })
      .animate(false)
      .style('stroke', (datum: any) => getColor(datum))
      .style('strokeOpacity', 0.2)
      .style('fill', (datum: any) => getColor(datum))

    addReferenceLines(chart.current)

    // slider 变化时按可视区域动态裁剪数据
    chart.current.on('slider:change', (e: any) => {
      const { value } = e.data
      if (!value) return

      const allData = rawDataRef.current
      const [startRatio, endRatio] = value
      const start = Math.floor(startRatio * allData.length)
      const end = Math.ceil(endRatio * allData.length)

      let visibleData = allData.slice(start, end)
      if (visibleData.length > MAX_RENDER_POINTS) {
        const step = Math.ceil(visibleData.length / MAX_RENDER_POINTS)
        visibleData = visibleData.filter((_: any, i: number) => i % step === 0)
      }

      chart.current?.changeData(visibleData)
    })

    chart.current.render()
    isRenderingRef.current = false
  }, [getColor, addReferenceLines])

  // ----- 首次挂载时初始化 -----
  useEffect(() => {
    initChart()
    return () => {
      chart.current?.destroy()
      chart.current = null
      isRenderingRef.current = false
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ----- 暴露给父组件的方法 -----

  /**
   * refresh：销毁当前图表并使用最新 props 重新渲染
   * 父组件在 Apply 后调用：chartRef.current.refresh()
   */
  const refresh = useCallback(() => {
    initChart()
  }, [initChart])

  /**
   * scrollTo：将 slider 跳转到指定比例区间，同步更新展示数据
   */
  const scrollTo = useCallback((startRatio: number, endRatio: number) => {
    if (!chart.current) return

    const allData = rawDataRef.current
    const start = Math.floor(startRatio * allData.length)
    const end = Math.ceil(endRatio * allData.length)

    let visibleData = allData.slice(start, end)
    if (visibleData.length > MAX_RENDER_POINTS) {
      const step = Math.ceil(visibleData.length / MAX_RENDER_POINTS)
      visibleData = visibleData.filter((_: any, i: number) => i % step === 0)
    }

    chart.current.emit('slider:change', { data: { value: [startRatio, endRatio] } })
    chart.current.changeData(visibleData)
  }, [])

  useImperativeHandle(ref, () => ({ refresh, scrollTo }))

  return <div ref={container} style={{ width: '100%' }} />
}

export default forwardRef(ControlChart)
