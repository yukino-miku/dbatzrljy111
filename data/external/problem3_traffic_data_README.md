# 问题三实际流量数据说明

`problem3_traffic_data_template.csv` 只有字段名，不包含虚构数据。请另存一份数据文件后通过：

```powershell
python problem3_network_routing.py --mode standard --stage traffic --traffic-file <CSV路径>
```

支持三种互斥口径：

1. 平均速率：填写 `region_name,traffic_value,traffic_unit,period,source_name,source_url,notes`，单位为 `Gbps` 或 `Tbps`。
2. 累计流量：填写 `region_name,traffic_value,traffic_unit,period_start,period_end,source_name,source_url,notes`，单位为 `GB/TB/PB/EB`。程序用累计字节数乘 8，再除以统计周期秒数和 `1e9`，换算为平均 Gbps。
3. 时间曲线：填写 `timestamp,traffic_value,traffic_unit,source_name,source_url`，单位为 `Gbps` 或 `Tbps`。

同一文件不要混合速率单位和累计单位。`source_name`、`source_url` 和统计口径应填写可核验的真实来源；程序不会自动爬取网络数据，也不会替用户补造来源。

Standard 或 Full 模式未提供真实 CSV 时，流量阶段写入 `status=skipped_missing_real_data`，但拓扑和路由仍可运行。Quick 模式只有显式添加 `--allow-demo-traffic` 时才使用标为 `synthetic_demo` 的演示流量，该结果仅用于代码测试。
