SecureVision AI Dashboard – Open Instructions
==============================================

1. Unzip SecureVision_AI_Dashboard.zip completely.
2. Open Power BI Desktop (August 2025 / 2026 or later recommended).
3. Enable preview features if needed:
   File > Options and settings > Options > Preview features
   - Power BI Project (.pbip) save option
   - Store semantic model using TMDL format
   - Store reports using enhanced metadata format (PBIR)
4. Double-click: SecureVision_AI_Dashboard/SecureVision_AI_Dashboard.pbip

If data does not load (path error):
- The CSVs live in: SecureVision_AI_Dashboard/Data/
- In Power BI: Home > Transform data > Data source settings
  (or right-click each table in Model view > Edit query)
- Point each query to the matching CSV under the Data folder
  on your machine (full Windows path).

Pages included:
  OVERVIEW     – model accuracy + latency/throughput
  PERFORMANCE  – per-class precision/recall/F1 + F1 chart
  ANALYTICS    – cost summary + optimal threshold savings
  MONITORING   – latency detail, anomalies, data quality

All metrics come only from the original secure-vision-ai.zip.
