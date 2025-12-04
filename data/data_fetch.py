import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 配置参数
# ==========================================
COUNTRIES = ['CN', 'US', 'UK', 'JP']  # 节点列表 (N=4)
START_DATE = '2006-01-01'
END_DATE = '2023-12-31'
FREQ = 'M' # 月度频率

# 特征列表 (根据文档描述)
# 注意：'ExRate_Close' 是原始价格，用于计算 'ExRate_LogRet'
RAW_FEATURES = ['CPI', 'PolicyRate', 'RealGDP', 'Equity_Close', 'BondYield_10Y', 'ExRate_Close']
FINAL_FEATURES = ['CPI', 'PolicyRate', 'RealGDP', 'Equity_Ret', 'BondYield_10Y', 'ExRate_LogRet']

# ==========================================
# 2. 模拟数据生成 (实际使用时请替换为读取 Excel/CSV)
# ==========================================
def generate_mock_data():
    """
    生成模拟的原始数据。
    实际项目中，请用 pd.read_csv() 或 pd.read_excel() 替换此部分。
    """
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='M')
    quarterly_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='Q')
    
    data_dict = {}
    
    for country in COUNTRIES:
        # 创建月度数据框架
        df = pd.DataFrame(index=dates)
        
        # 模拟月度数据
        df['CPI'] = np.random.normal(100, 2, len(dates)) # 模拟CPI
        df['PolicyRate'] = np.random.uniform(0, 5, len(dates)) # 模拟利率
        df['BondYield_10Y'] = np.random.uniform(1, 4, len(dates)) # 模拟国债
        
        # 模拟金融市场价格 (用于计算收益率)
        # 股票价格 (随机游走)
        df['Equity_Close'] = 1000 * np.exp(np.cumsum(np.random.normal(0, 0.02, len(dates))))
        # 汇率价格 (随机游走)
        df['ExRate_Close'] = 7.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, len(dates))))
        
        # 模拟季度数据 (Real GDP)
        # 先生成季度数据，再合并
        df_q = pd.DataFrame(index=quarterly_dates)
        df_q['RealGDP'] = np.random.uniform(2, 8, len(quarterly_dates)) # 模拟GDP增长率
        
        # 将季度数据合并到月度索引中 (此时非季度末的月份会有 NaN)
        df = df.join(df_q)
        
        data_dict[country] = df
        
    return data_dict

# ==========================================
# 3. 数据预处理核心逻辑
# ==========================================
def process_data(raw_data_dict):
    """
    处理原始数据：对齐频率、计算收益率、标准化
    """
    processed_data = []
    
    # 初始化标准化器 (对每个特征独立标准化)
    # 注意：严格来说，Scaler应该只fit训练集，这里为了演示方便fit全量数据
    scalers = {feat: StandardScaler() for feat in FINAL_FEATURES}
    
    # 临时存储所有国家的数据以进行统一标准化fit
    all_countries_df = pd.DataFrame()

    # --- 第一步：单国家内部处理 ---
    country_dfs = {}
    for country in COUNTRIES:
        df = raw_data_dict[country].copy()
        
        # 1. 处理季度数据 (Real GDP)
        # 文档要求："repeated within months" -> 使用前向填充 (ffill)
        df['RealGDP'] = df['RealGDP'].ffill()
        # 如果开头有NaN (因为第一个季度数据还没出)，用后向填充兜底
        df['RealGDP'] = df['RealGDP'].bfill()
        
        # 2. 计算金融特征 (收益率)
        # 股票收益率
        df['Equity_Ret'] = df['Equity_Close'].pct_change()
        # 汇率对数收益率 (Target) -> ln(Pt / Pt-1)
        df['ExRate_LogRet'] = np.log(df['ExRate_Close'] / df['ExRate_Close'].shift(1))
        
        # 3. 去除因计算差分产生的 NaN (通常是第一行)
        df = df.dropna()
        
        # 仅保留最终需要的特征列
        df = df[FINAL_FEATURES]
        
        country_dfs[country] = df
        
        # 堆叠用于Scaler训练
        df_with_label = df.copy()
        df_with_label['Country'] = country
        all_countries_df = pd.concat([all_countries_df, df_with_label])

    # --- 第二步：标准化 (Z-Score) ---
    # 对每一列特征进行标准化，消除量纲差异 (如 GDP是百分比，CPI是指数)
    for feat in FINAL_FEATURES:
        # Fit
        scalers[feat].fit(all_countries_df[[feat]])
        
        # Transform 每个国家
        for country in COUNTRIES:
            country_dfs[country][feat] = scalers[feat].transform(country_dfs[country][[feat]])

    # --- 第三步：构建节点特征矩阵 X_t ---
    # 最终形状: (Time_Steps, Nodes, Features)
    # 确保所有国家的时间索引完全一致
    common_index = country_dfs[COUNTRIES[0]].index
    
    # 将字典转换为 3D 数组
    # 维度 0: 时间
    # 维度 1: 国家 (CN, US, UK, JP)
    # 维度 2: 特征
    X_list = []
    for t in range(len(common_index)):
        node_features_at_t = []
        for country in COUNTRIES:
            # 获取该国家在 t 时刻的特征向量
            vals = country_dfs[country].iloc[t].values
            node_features_at_t.append(vals)
        X_list.append(node_features_at_t)
    
    X_t = np.array(X_list, dtype=np.float32)
    
    return X_t, common_index

# ==========================================
# 4. 构建静态邻接矩阵 (Fixed Trade Graph)
# ==========================================
def build_adjacency_matrix():
    """
    构建基于贸易权重的静态邻接矩阵 A_trade。
    形状: (N, N) = (4, 4)
    """
    # 这里使用硬编码的示例权重。
    # 实际操作中：下载 IMF DOTS 数据 -> 计算双边贸易额 -> 归一化
    # 顺序对应: CN, US, UK, JP
    
    # 假设矩阵 A_ij 表示 j 对 i 的影响 (或 i 与 j 的连接强度)
    # 行归一化通常用于 GCN
    
    raw_weights = np.array([
        # CN,  US,   UK,   JP  (列)
        [0.0,  0.50, 0.10, 0.20], # CN (行)
        [0.40, 0.0,  0.15, 0.10], # US
        [0.10, 0.20, 0.0,  0.05], # UK
        [0.25, 0.15, 0.05, 0.0 ]  # JP
    ])
    
    # 添加自环 (Self-loop)，即国家受自身历史影响，通常设为 1
    np.fill_diagonal(raw_weights, 1.0)
    
    # 行归一化 (Row Normalization): 使得每一行的和为 1
    # D^{-1} A
    row_sums = raw_weights.sum(axis=1)
    adj_matrix = raw_weights / row_sums[:, np.newaxis]
    
    return adj_matrix

# ==========================================
# 5. 主执行流程
# ==========================================
if __name__ == "__main__":
    print("1. 正在生成模拟数据...")
    raw_data = generate_mock_data()
    print(f"   - 获取了 {len(raw_data)} 个国家的数据")
    print(f"   - 原始特征示例: {raw_data['CN'].columns.tolist()}")

    print("\n2. 正在处理数据 (对齐、差分、标准化)...")
    X_tensor_np, time_index = process_data(raw_data)
    
    print("\n3. 正在构建静态贸易图...")
    A_matrix_np = build_adjacency_matrix()

    print("\n4. 转换为 PyTorch 张量 (准备输入模型)...")
    # 转换为 PyTorch Tensor
    X = torch.tensor(X_tensor_np) # 特征矩阵
    A = torch.tensor(A_matrix_np, dtype=torch.float32) # 邻接矩阵

    # ==========================================
    # 输出最终数据维度检查
    # ==========================================
    # 保存数据到文件，A与X
    torch.save(A, "A_trade.pt")
    torch.save(X, "X_tensor.pt")
    print("-" * 30)
    print("数据准备完成 (Data Ready for ST-GCN-Base)")
    print("-" * 30)
    print(f"时间步数量 (T): {X.shape[0]} (从 {time_index[0].date()} 到 {time_index[-1].date()})")
    print(f"节点数量 (N):   {X.shape[1]} (CN, US, UK, JP)")
    print(f"特征数量 (F):   {X.shape[2]} {FINAL_FEATURES}")
    print(f"特征矩阵 X 形状: {X.shape} -> (Batch/Time, Nodes, Features)")
    print(f"邻接矩阵 A 形状: {A.shape} -> (Nodes, Nodes)")
    print("-" * 30)
    
    print("\n邻接矩阵内容示例 (A_trade):")
    print(np.round(A.numpy(), 2))