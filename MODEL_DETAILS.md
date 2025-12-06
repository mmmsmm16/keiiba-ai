# モデルおよびアルゴリズム詳細

本プロジェクトで使用している機械学習モデルと、評価に使用する指標の詳細について解説します。
新しいモデルを追加する際は、このドキュメントにセクションを追加してください。

## 1. 使用モデルの概要
現在、決定木ベースの勾配ブースティング（GBDT）2種と、Deep Learningモデル1種の計3モデルによるアンサンブル学習（Stacking/Blending）を採用しています。

1.  **LightGBM** (Ranking: LambdaRank)
2.  **CatBoost** (Ranking: YetiRank)
3.  **TabNet** (Deep Learning: Attentive Transformer)

これらを **Linear Regression** (メタモデル) で重み付け統合し、最終的なスコアを算出します。

---

## 2. LightGBM (LambdaRank)
[LightGBM](https://lightgbm.readthedocs.io/en/latest/) は、葉（Leaf）単位で成長させるGBDTアルゴリズムです。
本プロジェクトでは、**Learning to Rank (ランキング学習)** の手法である **LambdaRank** を使用しています。

### 数理的背景
通常の勾配ブースティングは損失関数 $L$ の勾配を学習しますが、ランキング指標（NDCGなど）は不連続で微分不可能です。
LambdaRankは、「あるアイテムペア $(i, j)$ の順位を入れ替えたときに、NDCGがどれだけ変化するか ($\Delta NDCG$)」を勾配の代わり（$\lambda$）として利用します。

$$
\lambda_{ij} = \frac{\partial C_{ij}}{\partial s_i} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} |\Delta NDCG_{ij}|
$$

ここで、$s_i, s_j$ はモデルのスコア、$\sigma$ はハイパーパラメータです。
これにより、「上位に来るべき順位のペア」の間違いをより強く修正するように学習が進みます。

### 設定
*   **Objective**: `lambdarank`
*   **Metric**: `ndcg` (NDCG@1, 3, 5)
*   **Group**: レースIDごとにグループ化

---

## 3. CatBoost (YetiRank)
[CatBoost](https://catboost.ai/) は、カテゴリ変数の扱いに長けたGBDTライブラリです。
本プロジェクトでは、ランキング学習のために **YetiRank** ロスを採用しています。

### 数理的背景 (YetiRank)
YetiRankは、LambdaRankやPairwiseアプローチを拡張したもので、**Softmax分布の不確実性** を考慮に入れています。
各アイテムのスコアが正規分布に従うと仮定し、予想されるランキングの分布全体に対する期待NDCGを最適化しようとします。これにより、従来のPairwiseな手法よりもノイズに対して頑健であるとされています。

### 設定
*   **Loss Function**: `YetiRank`
*   **Metric**: `NDCG`
*   **Task Type**: `CPU` (TabNetとの競合回避のため)

---

## 4. TabNet (Attentive Interpretable Tabular Learning)
[TabNet](https://arxiv.org/abs/1908.07442) は、ニューラルネットワークでありながら、決定木のような「特徴量選択」の機能を持つアーキテクチャです。

### アーキテクチャの特徴
1.  **Sequential Attention**:
    各決定ステップ ($i$) で、どの特徴量を見るべきかを選択するマスク $M[i]$ を学習します。
    $$
    M[i] = \text{Sparsemax}(P[i-1] \cdot h_i(a[i-1]))
    $$
    Sparsemax関数により、不要な特徴量の重みが完全にゼロになり、解釈性が向上します。

2.  **Feature Transformer**:
    選択された特徴量を処理する深い層（GLU: Gated Linear Unitを使用）です。

### 本プロジェクトでの扱い
TabNetは本来Classification/Regression用ですが、ここでは **回帰タスク (Regression)** として扱っています。
ターゲット値 $y$ は、着順 $rank$ に対して以下の変換を行ったものを使用します。

$$
y = \frac{1}{\text{rank}}
$$

*   1着 $\to 1.0$
*   2着 $\to 0.5$
*   ...
*   18着 $\to 0.055...$

これにより、「値が大きいほど良い」というスコアを学習させます（正規化Rankに近い扱い）。

---

## 5. 評価指標 (Metrics)

学習時および最終評価時に使用される主要な指標です。

### NDCG (Normalized Discounted Cumulative Gain)
ランキングの質を評価する指標です。上位に正解が含まれているほどスコアが高くなります。

$$
DCG_p = \sum_{i=1}^{p} \frac{rel_i}{\log_2(i+1)}
$$

$$
NDCG_p = \frac{DCG_p}{IDCG_p}
$$

*   $p$: 評価する上位件数（@1, @3, @5など）
*   $rel_i$: $i$番目のアイテムの関連度（本プロジェクトでは1着=1、他=0などのバイナリ等で使用）
*   $IDCG$: 理想的な順位（Ideal）だった場合のDCG

### 回収率シミュレーション
モデルの実用性を測るため、検証データ（2024年）に対して以下のシミュレーションを行っています。

1.  モデルの出力スコア $s$ を、レース内でのソフトマックス関数に通して勝率確率 $P$ に変換します。
    $$
    P_i = \frac{e^{s_i}}{\sum_{j \in \text{Race}} e^{s_j}}
    $$
2.  **期待値 (Expected Value)** を計算します。
    $$
    EV_i = P_i \times \text{Odds}_i
    $$
3.  期待値が最大の馬（またはスコア最大の馬）の単勝を100円購入したと仮定し、ROI (Return On Investment) を算出します。

$$
\text{ROI} = \frac{\text{総払戻金額}}{\text{総投資金額}} \times 100 (\%)
$$
