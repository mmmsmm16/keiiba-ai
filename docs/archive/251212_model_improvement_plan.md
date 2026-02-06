# モデル精度向上計画書 (Model v13+ Development Plan)

**作成日**: 2025-12-12  
**現行最良モデル**: v12_tabnet_revival (ROI 87.62%, 的中率 28.8%)  
**目標**: 根本的なモデル精度向上によるROI・的中率の同時改善

---

## 現状分析

### v12モデルの構成
- **アーキテクチャ**: LightGBM + CatBoost + TabNet → Linear Regression (Meta)
- **特徴量数**: 165個（Entity Embedding 32次元含む）
- **学習データ**: 12年分（2013-2024）、NAR含む
- **損失関数**: LambdaRank (NDCG最適化)

### 既存の強み
- Entity Embedding（horse_id, jockey_id, trainer_id, sire_id）導入済み
- v12専用戦略で2025年ROI 100%超え達成
- 複勝圏率 60.9% という高い安定性

### 課題
- 全レース対象ROIは87.62%（損益分岐点100%未達）
- 損失関数がランキング最適化であり、回収率直接最適化ではない

---

## 改善施策

### 1. 損失関数の改良【優先度: S】

現在のNDCG/LambdaRankは「順位予測」に特化しているが、回収率最大化には「高オッズ馬の正確な予測」がより重要。

#### 1.1 オッズ加重損失関数
```python
def odds_weighted_logloss(y_true, y_pred, odds):
    """高オッズの正解を重視する損失関数"""
    weight = np.log1p(odds) * (1 + y_true * 2)
    return -np.mean(weight * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
```

#### 1.2 期待値最大化損失 (EV Loss)
```python
def ev_loss(y_true, y_pred, odds):
    """期待値を直接最大化"""
    ev = y_pred * odds
    return -torch.mean(ev * y_true)  # 正解の期待値を最大化
```

**実装箇所**: `src/model/lgbm.py`, `src/model/catboost_model.py`

---

### 2. Meta-Modelの高度化【優先度: A】

現在のLinear RegressionをAttentionベースに置換し、「どの基盤モデルをどの条件で信頼するか」を動的に学習。

```python
class AttentionMetaModel(nn.Module):
    def __init__(self, n_base_models=3, context_dim=16):
        self.attention = nn.MultiheadAttention(embed_dim=n_base_models, num_heads=1)
        self.context_encoder = nn.Linear(context_dim, n_base_models)
        
    def forward(self, base_preds, context):
        # context: [n_horses, race_level, distance, etc.]
        weights = F.softmax(self.context_encoder(context), dim=-1)
        return (base_preds * weights).sum(dim=-1)
```

**期待効果**: 
- 荒れるレース → CatBoost重視
- 堅いレース → LightGBM重視
のような動的切り替え

---

### 3. Time Decay重み付け【優先度: A】

構造変化（2023年斤量改定、京都改修）への適応。

```python
def calc_sample_weight(date, reference_date, half_life_years=2):
    """直近データを重視する時間減衰重み"""
    days_ago = (reference_date - date).days
    half_life_days = half_life_years * 365
    return np.exp(-np.log(2) * days_ago / half_life_days)
```

**適用方法**:
- LightGBM: `sample_weight` パラメータ
- CatBoost: `sample_weight` パラメータ
- TabNet: DataLoader内で重み付きサンプリング

---

### 4. 新規特徴量の追加【優先度: B】

#### 4.1 グラフ特徴量（対戦ネットワーク）
```python
import networkx as nx

def build_matchup_graph(race_results):
    """馬同士の勝敗グラフを構築"""
    G = nx.DiGraph()
    for race in race_results:
        horses = sorted(race, key=lambda x: x['rank'])
        for i, winner in enumerate(horses):
            for loser in horses[i+1:]:
                G.add_edge(winner['horse_id'], loser['horse_id'])
    return G

def calc_pagerank_features(G, horse_id):
    """PageRank中心性を特徴量化"""
    pagerank = nx.pagerank(G)
    return pagerank.get(horse_id, 0.0)
```

#### 4.2 血統深化特徴量
- 母父(BMS) × 距離カテゴリ 勝率
- 血統類似度ベクトル（Embedding間のコサイン類似度）

#### 4.3 時系列スペクトル特徴量
```python
from scipy.fft import fft

def calc_performance_spectrum(past_ranks, n_components=3):
    """過去着順のFFTスペクトル（好不調の周期性）"""
    spectrum = np.abs(fft(past_ranks))[:n_components]
    return spectrum
```

---

### 5. FT-Transformer導入【優先度: B】

TabNetの代替または補完として、Feature Tokenizer + Transformerアーキテクチャを検討。

```python
# pytorch-tabular ライブラリ使用
from pytorch_tabular.models import FTTransformerConfig

config = FTTransformerConfig(
    task="regression",
    num_attn_blocks=3,
    num_heads=8,
    ffn_hidden_multiplier=2,
    attn_dropout=0.2,
)
```

---

### 6. 検証方法の強化【優先度: A】

#### 6.1 Walk-Forward Validation
```python
splits = [
    {'train': [2018, 2019, 2020, 2021], 'test': 2022},
    {'train': [2019, 2020, 2021, 2022], 'test': 2023},
    {'train': [2020, 2021, 2022, 2023], 'test': 2024},
]
# 各splitで訓練・評価し、全体の安定性を確認
```

#### 6.2 Expected Calibration Error (ECE)
```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """予測確率と実際の勝率の乖離を測定"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            ece += mask.sum() * abs(bin_accuracy - bin_confidence)
    return ece / len(y_true)
```

---

## 実装ロードマップ

| Phase | 期間 | 施策 | 成果物 |
|-------|------|------|--------|
| **Phase 1** | 1週間 | オッズ加重損失関数 | `src/model/custom_loss.py` |
| **Phase 2** | 3日 | Attention Meta-Model | `src/model/attention_meta.py` |
| **Phase 3** | 1日 | Time Decay導入 | `src/preprocessing/sample_weight.py` |
| **Phase 4** | 1週間 | グラフ特徴量 | `src/preprocessing/graph_features.py` |
| **Phase 5** | 2週間 | FT-Transformer | `src/model/ft_transformer.py` |
| **Eval** | 随時 | Walk-Forward検証 | `src/model/walk_forward_eval.py` |

---

## 成功指標

| 指標 | 現状 (v12) | 目標 (v13+) |
|-----|-----------|------------|
| 全レースROI | 87.62% | **95%+** |
| 的中率 | 28.8% | **30%+** |
| 複勝圏率 | 60.9% | **65%+** |
| 条件付きROI | 100%+ | **110%+** |
| ECE (較正誤差) | 未計測 | **0.05以下** |

---

## 追加検討事項

### 7. データ品質・リーク再検証【優先度: A】

推論時に使用できない特徴量の混入チェック。

```python
# リーク候補の自動検出
leak_candidates = ['rank', 'time', 'odds', 'popularity', 'passing_rank']
for col in feature_cols:
    if any(leak in col for leak in leak_candidates):
        print(f"⚠️ リーク可能性: {col}")
```

**確認ポイント**:
- lag特徴量の時点ずれリスク
- race_内集計特徴量で当該馬自身を含んでいないか

---

### 8. NARデータ分離実験【優先度: A】

JRAとNARの構造差異による負の転移リスク：
- 馬場特性・コース形状の違い
- オッズ構造（控除率）の違い
- 騎手・調教師の分布差異

**実験計画**:
```python
# 実験A: JRA専用モデル
train_jra = df[df['venue'].isin(jra_venues)]

# 実験B: NAR含む混合モデル（現行）
train_all = df

# 比較: JRA検証データでのROI差分
```

---

### 9. 特徴量選択・削減【優先度: A】

165特徴量からノイズを削減：

```python
from sklearn.inspection import permutation_importance

# Permutation Importanceで重要度測定
perm_imp = permutation_importance(model, X_val, y_val, n_repeats=10)

# 重要度が負または極めて低い特徴量を削除候補に
drop_candidates = [f for f, imp in zip(features, perm_imp.importances_mean) if imp < 0.001]
```

---

### 10. Adversarial Validation【優先度: B】

Train/Testデータの分布ドリフトを検出：

```python
# Train=0, Test=1としてモデルが区別できるか
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

X_combined = pd.concat([X_train, X_test])
y_is_test = np.array([0]*len(X_train) + [1]*len(X_test))

clf = LGBMClassifier()
auc = cross_val_score(clf, X_combined, y_is_test, scoring='roc_auc', cv=5).mean()
# AUC > 0.6 ならドリフトあり
```

---

### 11. Ensemble多様性の確保【優先度: B】

現在LightGBM/CatBoost/TabNetは同一特徴量で学習 → 予測相関が高い可能性。

**Feature Bagging**:
```python
# 異なる特徴量サブセットで学習
lgbm_features = random.sample(all_features, int(len(all_features) * 0.8))
catboost_features = random.sample(all_features, int(len(all_features) * 0.8))
```

---

### 12. オッズ加重損失の過学習対策【優先度: B】

高オッズ馬はサンプル数が少なく過学習リスクあり：

```python
# Focal Loss併用で少数クラスへの注力を制御
def focal_loss(y_true, y_pred, gamma=2.0):
    pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    return -((1 - pt) ** gamma) * torch.log(pt)
```

---

## 追加推奨施策サマリー

| 施策 | 優先度 | 期待効果 | 工数 |
|------|--------|---------|------|
| **データリーク再検証** | A | 致命的バグ防止 | 1日 |
| **NARデータ分離実験** | A | 負の転移リスク排除 | 3日 |
| **Permutation Importance特徴量削減** | A | ノイズ削減 | 2日 |
| **Adversarial Validation** | B | ドリフト検出 | 1日 |
| **Feature Bagging** | B | Ensemble多様性向上 | 2日 |
| **Focal Loss併用** | B | 過学習防止 | 1日 |

---

## 参考文献

- `reference/feature_analyze.md` - 高度技術レポート
- `reference/racehorse_AI_analyze.md` - 学習データ期間の最適化
- `docs/feature_engineering_v6_plan.md` - 特徴量計画


