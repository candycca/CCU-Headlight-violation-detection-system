# 即時車輛違規系統

## 簡介
  傳統的車燈違規偵測方式需要警方投入大量人力來監控車輛是否違規，這不僅耗費了大量的時間和人力成本，還容易導致誤判。為了減少警方的人力負擔，並提升交通執法的效率，我們研發一套基於人工智慧的自動化車燈違規偵測系統，取代傳統的人工方式，提高執法的效率及準確率。

## 使用環境
- **作業系統**: Linux 
- **開發環境**: 本地開發 (localhost)
- **伺服器環境**: PHP server with Port Forward (端口轉發)
- **程式語言**: 
  - PHP 
  - Python (使用 Anaconda 管理環境)
- **資料庫**: SQLite3

## 前置準備

1. **安裝所需套件:**
    - python 3.8
    - numpy
    - opencv
    - torch
    - ultralytics

## 訓練與測試(編輯中

### 訓練模型
1. 執行訓練指令：
    - 轉彎模型
      ```bash
      python ResNet_train/test.py
      ```
    - 車燈辨識模型
      ```bash
      python CNNLSTM_train/test.py
      ```
    
3. 訓練過程中，記錄模型的訓練與測試指標（準確率、損失值等），保存至 log 文件。

### 測試模型
1. 執行測試指令：
    ```bash
     python ccc.py --name filefolder
    ```
    - filefolder：測試集的資料夾

## 功能說明
簡單描述專案的主要功能：
1. **違規檢測系統**: 輸入車輛資訊，檢測車輛是否有違規行為。
2. **結果輸出**: 系統將輸出車輛違規結果及相關數據。

## 模型架構
![image](https://github.com/candycca/CCU-Headlight-violation-detection-system/blob/main/docs/%E7%B3%BB%E7%B5%B1%E6%9E%B6%E6%A7%8B%E5%9C%96.png)

## 文件結構(編輯中

```bash
|-- my_project/
    |-- app/               # PHP app
    |-- database/          # SQLite3 資料庫
    |-- public/            # 前端文件
        |-- style.css      # 前端樣式
    |-- scripts/           # 訓練與測試的 Python 腳本
        |-- train_model.py
        |-- test_model.py
    |-- README.md          # 專案說明文件
