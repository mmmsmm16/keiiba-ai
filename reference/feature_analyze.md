次世代競馬予測AIのための高度技術・戦略的包括レポート
1. 序論：競馬予測におけるパラダイムシフトと現代的課題
競馬予測の領域は、過去数十年の間に劇的な変貌を遂げました。かつては専門家の経験則や単純な線形回帰モデルに依存していたこの分野は、現在、計算機科学、統計的学習理論、そして行動経済学が交差する最先端のデータサイエンスの実験場となっています。現代の競馬予測AI開発において、単に過去のレース結果を学習させるだけのモデルでは、市場の効率性（Efficient Market Hypothesis）を打ち破り、長期的な収益を上げることは極めて困難です。
本レポートは、既存の成功事例、学術論文、およびKaggle等のデータ分析コンペティションにおける上位解法を網羅的に調査・分析し、競馬予測AIの精度を極限まで高めるための技術的枠組みを提示するものです。特に、従来のテーブルデータ処理にとどまらず、グラフニューラルネットワーク（GNN）による血統・対戦関係のモデリング、Transformerを用いた時系列・軌跡データの解析、コンピュータビジョンによるパドック映像診断、そして予測スコアを収益に変換するための強化学習（RL）およびケリー基準の最適化手法に焦点を当てます。
これら高度な技術要素を統合し、単なる着順予測（Classification/Regression）から、相対的な優劣を決定するランキング学習（Learning to Rank）、そして最終的な意思決定を行うベッティングエージェントへと昇華させることが、現代の競馬AIにおける勝利の方程式であることが、調査データから明らかになっています。
2. 特徴量エンジニアリングの深化：予測精度の源泉
機械学習モデルの性能は、入力されるデータの質と表現力に依存します。多くのKaggle上位ソリューションや成功したAIモデルにおいて、モデルのアルゴリズム選定以上に重要視されているのが特徴量エンジニアリングです。ここでは、基本的な統計量を超えた、予測精度に直結する高度な特徴量生成手法について詳述します。
2.1. カテゴリカル変数の表現学習：Entity Embeddings
競馬データは、馬ID、騎手ID、調教師ID、種牡馬IDなど、極めて高次元かつスパース（疎）なカテゴリカル変数によって構成されています。これらを従来のOne-Hot Encodingで処理すると、次元の呪いに陥り、メモリ効率が悪化するだけでなく、カテゴリ間の潜在的な関係性を捉えることができません。
これに対し、Entity Embeddings（エンティティ埋め込み） は、カテゴリ変数を低次元の密ベクトル（例：10〜50次元）に圧縮して表現する手法です。これは自然言語処理における単語埋め込み（Word2Vecなど）と同様の概念ですが、教師あり学習の過程でニューラルネットワークによって獲得されます 1。
メカニズムと利点: ニューラルネットワークの中間層で学習された埋め込みベクトルは、各騎手や馬の「特性」を数学空間上に配置します。例えば、「逃げが得意な騎手」や「重馬場に強い種牡馬」同士は、ベクトル空間上で近い距離に配置されるようになります。
実装と効果: 研究によれば、学習済みニューラルネットワークから抽出したEntity Embeddingsを、LightGBMやXGBoostなどの勾配ブースティング決定木（GBDT）の入力特徴量として使用することで、One-Hot Encodingと比較して大幅な精度向上と学習時間の短縮が報告されています 1。特にRossmann Store Salesコンペティションなどの事例では、Entity Embeddingsの導入がモデルの汎化性能を飛躍的に高めることが実証されており、これを競馬の騎手・調教師データに応用することは極めて有効です 4。
2.2. グラフベース特徴量：対戦関係と血統のネットワーク化
競馬は個々の馬が独立して走るのではなく、特定のライバルたちと競い合う「関係性」のゲームです。また、血統という巨大な有向非巡回グラフ（DAG）が背後に存在します。これらの構造を捉えるために、グラフ理論に基づいた特徴量が導入されています。
競走馬のグローバル有向グラフ: 全レース結果から、「馬Aが馬Bに先着した」という事実を有向エッジ（A $\rightarrow$ B）として表現し、巨大な勝敗グラフを構築します。このグラフに対してPageRankアルゴリズムなどを適用することで、単なる勝率ではなく、「強い相手に勝った馬」を高く評価する中心性スコアを算出できます 5。
条件付きサブグラフ: さらに、「雨の日のレースのみ」「距離2000m以上のみ」といった条件でエッジをフィルタリングし、特定の条件下での中心性を計算することで、レース当日の条件への適性を定量化する強力な特徴量となります 5。学術的な検証において、これらのグラフ特徴量を追加したモデルは、基本特徴量のみのモデルと比較して有意に予測精度が向上することが確認されています 6。
血統ネットワークとGNN: 従来のモデルでは血統を単なるラベルとして扱ってきましたが、Graph Neural Networks（GNN）を用いることで、父、母、母父といった血縁関係のグラフ構造上での情報の伝播（Message Passing）をモデル化できます。これにより、直接の親だけでなく、数世代前の祖先からの遺伝的特徴（スタミナ、スピード適性など）を、現在の競走馬の埋め込みベクトルに集約することが可能です 7。
2.3. 時系列・周波数解析：Nishikaコンペ等の知見
日本のデータ分析コンペティションプラットフォームNishikaで開催された競馬予測コンペの優勝解法などから、時系列データの高度な処理方法が明らかになっています。
周波数領域の特徴量 (FFT): 多くのモデルが過去の戦績（着順やタイム）をそのまま時系列データとして扱いますが、一部の先進的なアプローチでは、これを周波数領域に変換しています。具体的には、過去のレース結果の時系列に対して短時間フーリエ変換（STFT） を適用し、スペクトルパワー、ピーク周波数、スペクトル重心、エントロピーなどを抽出します。これにより、馬の好不調の周期性や、パフォーマンスの安定性を捉えることが可能となります 8。
マルチ解像度特徴抽出: 過去3走、5走といった固定窓だけでなく、長期的（過去数年）、中期的（過去半年）、短期的（直近）といった複数の時間枠（マルチ解像度）で集計した統計量（平均、分散、歪度、尖度）を組み合わせることで、馬の能力のベースラインと現在の勢い（モメンタム）の両方をモデルに学習させます 8。
2.4. 自然言語処理 (NLP) による定性情報の定量化
騎手コメント、調教師の戦前インタビュー、競馬新聞の短評などの非構造化テキストデータには、数値データには現れない重要なシグナルが含まれています。
感情分析 (Sentiment Analysis): 「手応えは良かったが前が詰まった」「まだ太め残り」といったコメントを、BERTなどの事前学習済みモデルを用いてベクトル化、あるいは感情極性辞書を用いてスコアリングします。例えば、「敗因：不利」を示すネガティブな単語と着順の乖離を分析することで、実力負けではないケースを特定し、次走での過小評価（妙味）を見抜く特徴量として機能します 9。
専門用語の構造化: 競馬特有の表現（「出遅れ」「掛かる」「折り合い」）を抽出し、これらをカテゴリカル特徴量としてタグ付けすることで、レース展開の不利や馬の気性難を明示的にモデルに教え込むことができます 10。
2.5. コンピュータビジョン (CV)：パドック・歩様解析
近年のAI技術の進展により、レース直前のパドック映像や調教映像から、馬の身体的コンディションを直接解析する試みが実用化されつつあります。
姿勢推定 (Pose Estimation): OpenPoseや独自の姿勢推定モデルを用いて、パドックを周回する馬の骨格点（関節位置）を検出します。これにより、歩幅（ストライド）、踏み込みの深さ、リズムの一定性、四肢の可動域などを定量的な時系列データとして抽出します 12。
異常検知と調子判定: 過去の好走時の歩様データと当日のデータを比較し、その乖離度を特徴量とします。また、発汗量やチャカつき（興奮状態）などの行動特徴を映像から抽出することで、数値データだけでは分からない当日の「気配」をモデルに組み込むことが可能です。AlphaImpactなどの商用AIやJRA-VANの実験的プロジェクトでも、この映像解析技術の導入が進められています 14。
3. 機械学習・深層学習モデルアーキテクチャの選定と最適化
特徴量が予測の「燃料」であるならば、モデルアーキテクチャは「エンジン」です。現在、競馬予測において支配的な地位を占める勾配ブースティング決定木（GBDT）と、台頭する深層学習モデル、そしてそれらを統合するアンサンブル戦略について詳述します。
3.1. GBDTの覇権：LightGBM, XGBoost, CatBoostの比較と使い分け
テーブルデータに対する予測性能において、GBDTは依然として最強のアルゴリズムです。しかし、ライブラリごとの特性を理解し、適切に使い分けることが重要です。

アルゴリズム
特徴と競馬予測における利点
弱点・注意点
LightGBM
高速性とRanking学習。Leaf-wise（葉ごとの）成長戦略を採用しており、大規模データでも高速に学習可能。lambdarankなどのランキング学習用目的関数が充実しており、着順予想に最適 17。
過学習しやすいため、num_leavesやmax_depthの調整が必須。小規模データには不向きな場合がある。
CatBoost
カテゴリ変数の自動処理。Ordered Target Statisticsにより、事前のエンコーディングなしでカテゴリ変数を高精度に処理できる。騎手IDや馬IDが多い競馬データで強力な威力を発揮する 19。
学習速度がLightGBMに比べて遅い傾向がある。パラメータチューニングが独特。
XGBoost
安定性と正則化。強力なL1/L2正則化項を持ち、汎化性能が高い。歴史が長く、知見が豊富。Kaggleの上位解法では、LightGBMとのアンサンブルの基盤として頻繁に使用される 19。
カテゴリ変数の処理にはOne-Hot Encodingなどが必要（最新版では対応が進んでいるがCatBoostほどではない）。

推奨戦略: 単一のモデルに頼るのではなく、LightGBM（速度とRanking）、CatBoost（カテゴリ処理）、XGBoost（安定性）の3つをそれぞれ学習させ、後述するスタッキング（Stacking）によって統合するのが最も確実な精度向上策です 21。
3.2. 学習の枠組みの転換：分類・回帰から「ランキング学習」へ
多くの初学者が陥る罠は、競馬予測を「1着になるか否か（2値分類）」や「走破タイムの予測（回帰）」として定式化してしまうことです。しかし、競馬の本質は「他馬との相対的な順序」にあります。レースのタイムは天候や馬場状態によって大きく変動するため、絶対値の予測（回帰）はノイズの影響を受けやすくなります。
Learning to Rank (LTR): 検索エンジンのページランク付けなどで使われる技術を応用します。LightGBMの rank_xendcg や lambdarank などの目的関数を使用し、レースIDをクエリグループとして、その中での馬の順序（Ranking）を正しく並べ替えることを学習させます。これにより、1着と2着の微妙な差を、タイムの二乗誤差よりも敏感に捉えることができます 17。
Pairwise Ranking: 馬Aと馬Bのペアを入力とし、「どちらが先着するか」を予測するモデルを構築します。このペアワイズな比較を全組み合わせで行うことで、全体の順位を構成します。計算コストは高いですが、相対評価に特化しているため高い精度が期待できます 26。
3.3. 深層学習の活用：TransformerとTabNet
GBDTが支配的なテーブルデータ領域においても、近年の深層学習アーキテクチャの進化により、GBDTに匹敵または凌駕する成果が出始めています。
Transformer for Tabular Data (TabTransformer, FT-Transformer): 自然言語処理で革命を起こしたTransformerのSelf-Attention機構をテーブルデータに応用したものです。特徴量間の相互作用（例：特定の騎手と特定のコースの相性）をAttentionによって動的に学習できる点が強みです。特に、過去のレース履歴をシーケンス（時系列）として入力し、Transformerでエンコードする手法は、長期的な依存関係を捉えるのに有効です 27。
軌跡予測のためのTransformer: GPSトラッキングデータ（Trakusなど）を用いる場合、各時点の $(x, y)$ 座標や速度ベクトルのシーケンスをTransformerに入力し、レース展開や最終的な位置取りを予測する研究が行われています。これにより、「第3コーナーでの位置取りが最終結果にどう影響するか」といった時空間的なダイナミクスをモデル化できます 29。
3.4. アンサンブル学習：Kaggle 1st Placeの手法
Kaggleの "Hong Kong Horse Racing" コンペティションや、類似の金融予測コンペ（Amex Default Predictionなど）の優勝解法 23 において、スタッキング（Stacking） は必須の技術です。
多層スタッキング:
Level 1: 多様なモデル（LightGBM, XGBoost, CatBoost, Neural Network, TabNet）を、異なる特徴量セットやハイパーパラメータで学習させ、予測値（Out-of-Fold Predictions）を出力させます。
Level 2 (Meta-Learner): Level 1のモデル出力値を「新たな特徴量」として入力し、ロジスティック回帰や浅い決定木、あるいはNeural Networkで最終的な予測を行います。これにより、各モデルの強みを活かしつつ弱点を補完し、予測の分散を抑えて安定した精度を実現します 23。
多様性の確保: アンサンブルの効果を最大化するためには、構成するモデルの「相関が低い」ことが重要です。決定木ベースのモデルだけでなく、ニューラルネットワークやk近傍法（KNN）などを混ぜることで、異なる決定境界を持つモデル群を構築し、モデル全体のロバスト性を高めます 30。
4. ベッティング戦略の最適化：予測を利益に変える数理
高精度な予測モデルが完成しても、それだけで利益が出るわけではありません。競馬はパリミュチュエル方式（オッズが投票率に連動する）であるため、大衆もまたある程度の予測能力を持っています。利益を上げるためには、モデルの予測確率と市場オッズの乖離（非効率性）を見つけ出し、適切な資金配分を行う必要があります。
4.1. ケリー基準 (Kelly Criterion) の適用と実践
ケリー基準は、複利運用において長期的な資産成長率を最大化するための数学的に証明された資金配分式です。
基本公式: 最適な投資比率 $f^*$ は以下の式で表されます。

$$f^* = \frac{bp - q}{b}$$

ここで、$p$ はモデルが予測した勝率、$q$ は負ける確率 ($1-p$)、$b$ はオッズ（純オッズ、例えば3.0倍なら $b=2$）です。この式は、期待値がプラスの場合にのみ正の値を返し、エッジ（優位性）の大きさに応じて賭け金を調整します 32。
Fractional Kelly (部分的ケリー): 理論上のケリー基準は、確率 $p$ が正確であることを前提としていますが、実際のモデル予測には誤差が含まれます。過大評価による破産リスク（Ruin Risk）を避けるため、実務的には算出された $f^*$ の20%〜50%程度（ハーフケリー、クォーターケリー）を賭けるのが一般的かつ安全な戦略とされています 35。
複数の馬への同時ベット: 1つのレースで複数の馬に賭ける場合（例：単勝多点買い）、相互排反事象を考慮した多変量ケリー基準や、二次計画法（Quadratic Programming）を用いた最適化が必要になります 35。
4.2. 強化学習 (Reinforcement Learning) による自動ベッティング
静的な数式であるケリー基準に対し、強化学習 (RL) は環境との相互作用を通じて動的なベッティング戦略を獲得します。香港中文大学（CUHK）の論文や関連研究 37 では、以下のようなRLフレームワークが提案されています。
エージェント設計:
状態 (State): 現在の資金、各馬のモデル予測勝率、現在のオッズ、レースの特性（頭数、グレードなど）。
行動 (Action): どの馬券種（単勝、複勝、馬連など）に、いくら賭けるか（あるいは見送るか）。
報酬 (Reward): 資金の増減（損益）。ただし、単に利益を最大化するだけでなく、ドローダウン（最大資産からの下落幅）をペナルティとして組み込むことで、リスク回避的な安定したエージェントを育成することが重要です 37。
Deep Q-Network (DQN) の活用: XGBoostなどで予測した勝率をDQNの入力として使い、最終的な賭けの意思決定を行わせる「モデルベース強化学習」のアプローチが有効です。実験結果では、単純なルールベースの賭け方よりも、RLエージェントの方が高い回収率を示すケースが報告されていますが、報酬関数の設計（Reward Shaping）が極めて難しいという課題もあります 37。
4.3. 回収率シミュレーションとバックテスト
AlphaImpactなどの成功事例では、単に的中率を追うのではなく、「回収率」に特化したシミュレーションを行っています。例えば、「AI指数が高い馬」を無条件に買うのではなく、「AIの評価が高いが、オッズが10倍以上ついている（過小評価されている）馬」を狙う穴狙い戦略や、期待値（勝率 $\times$ オッズ）が1.0を大きく超える馬のみを抽出するフィルタリングが、実際の運用成績を大きく左右します 38。
この検証には、過去のデータを用いて、当時のオッズで賭けた場合にどうなっていたかを厳密に再現するバックテストが不可欠です。これには、Pythonの backtesting.py や sports-betting などのライブラリを活用し、手数料（控除率）やオッズの変動（自分の投票によるオッズ低下）も考慮したシミュレーション環境を構築する必要があります 41。
5. 技術スタックと実装：成功事例の分析
5.1. データ収集基盤 (Scraping & Database)
日本の競馬においては netkeiba.com が主要なデータソースとなります。GitHub上で公開されている keibascraper や nkparser 43 などのライブラリを利用し、レース情報、馬柱、オッズ、結果を定期的に収集・DB化するパイプラインの構築が第一歩です。SQLiteやPostgreSQLなどのRDBにデータを蓄積し、特徴量生成のためのSQLクエリを最適化することが、実験サイクルの高速化に繋がります 17。
5.2. 成功したKaggle解法とGitHubリポジトリ
Kaggle Hong Kong Horse Racing: 1st Place Solutionでは、わずかなデータ量（数千レース）に対し、LightGBMを用いた徹底的な特徴量エンジニアリング（オッズの変動、馬体重の増減率、騎手の連対率など）を行い、特に「偽陽性（勝つと予測して負ける）」を最小化するような適合率（Precision）重視のチューニングが行われました 23。
Dominic Plouffe's Repository: サポートベクター回帰（SVR）を用いたアプローチで、各馬の「着順」を直接予測するのではなく、スピード指数や過去の勝率などの20のコア特徴量を用いて、レースごとの相対的なスコアを算出する方法が公開されています 46。
6. 結論と提言
競馬予測AIの精度を限界突破させるためには、以下の3つのレイヤーでの高度化が必要です。
データの多次元化 (Data Multi-modality):
単なるテーブルデータにとどまらず、Entity Embeddingsによるカテゴリ変数の密ベクトル化、GNNによる血統・対戦ネットワークの取り込み、そしてコンピュータビジョンやNLPによる非構造化データの活用を行うこと。これらは、従来の統計モデルでは捉えきれなかった「馬の潜在能力」や「当日の気配」を数値化します。
学習目的の再定義 (Objective Redefinition):
分類や回帰の問題設定から脱却し、LambdaRankなどのランキング学習を導入して、レース内の相対順位を直接最適化すること。さらに、GBDTとNeural Network（Transformer等）のアンサンブル（Stacking） を前提としたモデルパイプラインを構築すること。
意思決定の数理化 (Decision Science):
予測確率をそのまま信じるのではなく、ケリー基準や強化学習を用いて、リスクとリターンのバランスを数学的に最適化したベッティング戦略を実行すること。予測精度（Accuracy）ではなく、最終的な資産増加（ROI）を目的関数としたエージェントの育成が、実運用における成功の鍵となります。
これらの技術要素を、堅牢なデータパイプラインとバックテスト環境の上に統合することで、現代の競馬市場においても持続的な優位性を持つAIシステムの構築が可能となります。
補遺：主要技術要素の構造化データ
表1: 競馬予測における推奨モデルアーキテクチャ比較

モデル種別
アルゴリズム例
推奨される用途・役割
特記事項・強み
GBDT
LightGBM
メインモデル（ランキング）
高速学習、lambdarankによる順位学習、大規模データ適性 17
GBDT
CatBoost
メインモデル（カテゴリ処理）
騎手・馬ID等のカテゴリ変数処理に特化、過学習への耐性 19
Neural Network
Transformer
時系列・軌跡解析
過去レース履歴やGPS軌跡データのシーケンス学習、長期依存性の獲得 27
Neural Network
GNN
関係性・血統解析
血統図や対戦グラフからの特徴抽出（Node Embedding） 5
Neural Network
MLP
埋め込み生成
Entity Embeddingsの生成（前処理）、GBDTとのスタッキング素材 1
Meta-Model
Logistic Regression
アンサンブル（Stacking）
各モデルの予測値を統合し、最終確率を算出・補正 23

表2: 効果的な特徴量カテゴリと具体例

カテゴリ
具体的な特徴量例
生成手法・技術
ソース
エンティティ
騎手ベクトル、馬ベクトル、調教師ベクトル
Entity Embeddings (NNによる学習)
1
グラフ・ネットワーク
PageRank中心性、In-Degree/Out-Degree（対戦優位度）
ネットワーク分析 (NetworkX)
5
時系列・信号処理
過去走タイムのFFTスペクトル、移動平均、勢い（Momentum）
短時間フーリエ変換、窓関数処理
8
物理・生体力学
ストライド長、ピッチ、走行軌跡の効率性、ドラフティング時間
姿勢推定 (OpenPose)、GPS解析
12
テキスト・感情
騎手コメントの感情スコア、敗因タグ（「詰まり」「出遅れ」）
BERT, 感情極性辞書
10
ドメイン知識
スピード指数（トラック・馬場差補正済み）、ペース判断
統計的補正、ルールベースロジック
48

表3: ベッティング戦略の数理

戦略名
概要
適用場面・メリット
注意点
ケリー基準
幾何平均リターンを最大化する資金配分式
エッジ（期待値）がある場合の最適解
過大評価時の破産リスク。Fractional（1/2など）での運用が必須 33
強化学習 (DQN)
エージェントが試行錯誤で最適な賭け方を学習
複雑な状況（見送り、多点買い）への適応
報酬設計が難関。ドローダウンへのペナルティが必要 37
穴狙い (Longshot)
高オッズかつAI評価が高い馬のみを狙う
回収率100%超えを狙う現実的アプローチ
的中率が低くなるため、忍耐と十分な試行回数が必要 38

引用文献
(PDF) Entity Embeddings of Categorical Variables - ResearchGate, 12月 9, 2025にアクセス、 https://www.researchgate.net/publication/301878003_Entity_Embeddings_of_Categorical_Variables
Entity Embeddings of Categorical Variables - Kaggle, 12月 9, 2025にアクセス、 https://www.kaggle.com/code/lucapapariello/entity-embeddings-of-categorical-variables/notebook
Entity Embeddings for ML | Towards Data Science, 12月 9, 2025にアクセス、 https://towardsdatascience.com/entity-embeddings-for-ml-2387eb68e49/
Entity Embeddings for Categorical Data - Emergent Mind, 12月 9, 2025にアクセス、 https://www.emergentmind.com/papers/1604.06737
Horse racing prediction using graph-based features. - ThinkIR, 12月 9, 2025にアクセス、 https://ir.library.louisville.edu/cgi/viewcontent.cgi?article=4083&context=etd
"Horse racing prediction using graph-based features." by Mehmet ..., 12月 9, 2025にアクセス、 https://ir.library.louisville.edu/etd/2953/
The future of pedigree research - The Owner Breeder, 12月 9, 2025にアクセス、 https://theownerbreeder.com/stories/the-future-of-pedigree-research/
Nishikaの睡眠コンペで1位になった解法の紹介 | TECH | NRI Digital, 12月 9, 2025にアクセス、 https://www.nri-digital.jp/tech/20230627-14038/
【自然言語処理】BERTなるもの...｜Non - note, 12月 9, 2025にアクセス、 https://note.com/noa813/n/n69584fe07b5c
Analysing in-running comments - Betwise Blog, 12月 9, 2025にアクセス、 https://blog.betwise.net/2010/05/01/analysing-in-running-comments-in-this-months-racing-ahead/
【自然言語処理】感情分析の進め方＆ハマりやすいポイント - Qiita, 12月 9, 2025にアクセス、 https://qiita.com/toshiyuki_tsutsui/items/604f92dbe6e20a18a17e
馬別パドック動画 デモ公開中｜AIパドックプロジェクト｜競馬情報ならJRA-VAN, 12月 9, 2025にアクセス、 https://jra-van.jp/lp/paddock/
深層学習による馬の姿勢推定システム「パドックAI 解析」U.I デザイン | MEGALO WORKS, 12月 9, 2025にアクセス、 https://works.megalab.tokyo/750/
JRAシステムサービス株式会社、AIでパドック動画を分割する新サービス「パドックアイ」を公開, 12月 9, 2025にアクセス、 https://prtimes.jp/main/html/rd/p/000000003.000126077.html
競馬×AIで、新しい競馬の楽しみ方を提供 「馬別パドック分割動画」試験公開開始 - AIsmiley, 12月 9, 2025にアクセス、 https://aismiley.co.jp/ai_news/jra-ghelia-keiba/
馬の姿勢推定技術｜AIで競走馬のパフォーマンス向上を実現する方法 - Hakky Handbook, 12月 9, 2025にアクセス、 https://book.st-hakky.com/purpose/racehorse-posture-estimation-using-ai-technology
LightGBMによるAI競馬予想(ランキング学習) - PC-KEIBA, 12月 9, 2025にアクセス、 https://pc-keiba.com/wp/lambdarank/
LightGBM でかんたん Learning to Rank - 霧でも食ってろ, 12月 9, 2025にアクセス、 https://knuu.github.io/ltr_by_lightgbm.html
XGBoost vs. CatBoost vs. LightGBM: A Guide to Boosting Algorithms | by Kishan A - Medium, 12月 9, 2025にアクセス、 https://kishanakbari.medium.com/xgboost-vs-catboost-vs-lightgbm-a-guide-to-boosting-algorithms-47d40d944dab
catboost/benchmarks: Comparison tools - GitHub, 12月 9, 2025にアクセス、 https://github.com/catboost/benchmarks
XGBoost vs LightGBM vs CatBoost vs AdaBoost - Kaggle, 12月 9, 2025にアクセス、 https://www.kaggle.com/code/faressayah/xgboost-vs-lightgbm-vs-catboost-vs-adaboost
[2401.06086] XGBoost Learning of Dynamic Wager Placement for In-Play Betting on an Agent-Based Model of a Sports Betting Exchange - arXiv, 12月 9, 2025にアクセス、 https://arxiv.org/abs/2401.06086
1st solution(update github code) - Kaggle, 12月 9, 2025にアクセス、 https://www.kaggle.com/competitions/amex-default-prediction/writeups/lucky-shake-1st-solution-update-github-code
Horse race rank prediction using learning-to-rank approaches - ResearchGate, 12月 9, 2025にアクセス、 https://www.researchgate.net/publication/380208045_Horse_race_rank_prediction_using_learning-to-rank_approaches
LightGBMによるAI競馬予想(チューニング編) - Qiita, 12月 9, 2025にアクセス、 https://qiita.com/PC-KEIBA/items/0f6a48dcc6bfdb86118f
Predicting Horse Racing Results with Machine Learning - CUHK CSE, 12月 9, 2025にアクセス、 https://www.cse.cuhk.edu.hk/lyu/_media/thesis/presentation-1703-2.pdf?id=students%3Afyp&cache=cache
A Simple Baseline for Predicting Events with Auto-Regressive Tabular Transformers - arXiv, 12月 9, 2025にアクセス、 https://arxiv.org/html/2410.10648v3
lamhungphu/TransformerForStockPricePrediction: Using Transformer approach to optimize stock price prediction results for 10 stocks - GitHub, 12月 9, 2025にアクセス、 https://github.com/lamhungphu/TransformerForStockPricePrediction
Trajectory Prediction Attempt using Transformer - Kaggle, 12月 9, 2025にアクセス、 https://www.kaggle.com/code/ryanglaspey/trajectory-prediction-attempt-using-transformer
機械学習で競馬の回収率140%超を達成：開発までの話 - Qiita, 12月 9, 2025にアクセス、 https://qiita.com/umaro_ai/items/d1e0b61f90098ee7fbcb
【v2.1.1】競馬予想AI v2の概要 - Zenn, 12月 9, 2025にアクセス、 https://zenn.dev/dijzpeb/books/848d4d8e47001193f3fb/viewer/08_about_v2
AIアルゴリズムトレードにおけるリスク管理：ケリー基準によるポジションサイズの最適化 - Qiita, 12月 9, 2025にアクセス、 https://qiita.com/tikeda123/items/6cc0d2c508d79ce96bbc
Kelly System for Investing and Kuhn-Tucker Conditions In this addendum to Section 5.8.4 of the book Operati, 12月 9, 2025にアクセス、 https://www.utwente.nl/en/eemcs/sor/boucherie/Operations%20Research/584operationsresearchkellybetting.pdf
The Gambler Who Cracked the Horse-Racing Code: A Story of Skill, Data, and a Little Bit of Luck | by Adel Basli | Medium, 12月 9, 2025にアクセス、 https://medium.com/@adelbasli/the-gambler-who-cracked-the-horse-racing-code-a-story-of-skill-data-and-a-little-bit-of-luck-c227c7e7207b
Kelly betting on horse races with uncertainty in probability estimates - arXiv, 12月 9, 2025にアクセス、 https://arxiv.org/pdf/1701.02814
The Kelly criterion for mutually exclusive markets. : r/algobetting - Reddit, 12月 9, 2025にアクセス、 https://www.reddit.com/r/algobetting/comments/1o0t6yo/the_kelly_criterion_for_mutually_exclusive_markets/
The Chinese University of Hong Kong - CUHK CSE, 12月 9, 2025にアクセス、 https://www.cse.cuhk.edu.hk/lyu/_media/thesis/report-2003-1.pdf?id=students%3Afyp&cache=cache
AlphaImpactとは - AIで大井競馬を攻略せよ - TCK × netkeiba, 12月 9, 2025にアクセス、 https://tck.sp.netkeiba.com/ai_yoso/about_aiyoso.html
競馬のAI予想をさらに改良！単勝専用モデルで長期間でも回収率160%！｜pakara - note, 12月 9, 2025にアクセス、 https://note.com/pakara_keiba/n/n283046881232
AI予想を活用した複勝買いのススメ ー回収率100%以上をめざしてー｜とりまる - note, 12月 9, 2025にアクセス、 https://note.com/dataij/n/n29353bdc05c6
sports-betting - PyPI, 12月 9, 2025にアクセス、 https://pypi.org/project/sports-betting/
Backtesting.py - Backtest trading strategies in Python, 12月 9, 2025にアクセス、 https://kernc.github.io/backtesting.py/
new-village/KeibaScraper: nkparser is a simple scraping library for netkeiba.com - GitHub, 12月 9, 2025にアクセス、 https://github.com/new-village/KeibaScraper
Releases · new-village/KeibaScraper - GitHub, 12月 9, 2025にアクセス、 https://github.com/new-village/nkparser/releases
Predict the Winning Horse(100% on small test data) - Kaggle, 12月 9, 2025にアクセス、 https://www.kaggle.com/code/yyzz1010/predict-the-winning-horse-100-on-small-test-data
dominicplouffe/HorseRacingPrediction: Using Support Vector regression algorithm to predict horse racing results - GitHub, 12月 9, 2025にアクセス、 https://github.com/dominicplouffe/HorseRacingPrediction
Bayesian Velocity Models for Horse Race Simulation - Kaggle, 12月 9, 2025にアクセス、 https://www.kaggle.com/code/bkumagai/bayesian-velocity-models-for-horse-race-simulation
[競馬予想AI] 特徴量の選択と作成で精度向上を目指そう｜とりまる, 12月 9, 2025にアクセス、 https://note.com/dataij/n/n9ff213453ce9
ＡＬＧＯ ＳＰＥＥＤＥＲ｜競馬ソフト使い放題の会員サービス DataLab.（データラボ） - JRA-VAN, 12月 9, 2025にアクセス、 https://jra-van.jp/dlb/sft/lib/algospeeder.html
